"""Real int4 quantization path: pack weights and runtime quantized Linear."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantize import get_blocks, get_named_linears

_EPS = 1e-9
_FORMAT_VERSION = "asdq_int4_v1"


def _set_submodule(root: nn.Module, qualified_name: str, new_module: nn.Module) -> None:
    parts = qualified_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def _pack_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack uint8 values in [0, 15] into uint8 nibbles along last dim."""
    if values.shape[-1] % 2 != 0:
        pad = torch.zeros(values.shape[:-1] + (1,), dtype=values.dtype, device=values.device)
        values = torch.cat([values, pad], dim=-1)
    low = values[..., 0::2]
    high = values[..., 1::2]
    return (low | (high << 4)).contiguous()


def _unpack_int4(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    out = torch.empty((packed.shape[0], packed.shape[1] * 2), dtype=torch.uint8, device=packed.device)
    out[:, 0::2] = low
    out[:, 1::2] = high
    return out[:, :in_features]


@dataclass
class PackedLinearState:
    qweight: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    bias: torch.Tensor | None
    hp_indices: torch.Tensor
    hp_weight: torch.Tensor
    in_features: int
    out_features: int
    group_size: int

    def to_dict(self) -> Dict:
        return {
            "qweight": self.qweight.cpu(),
            "scales": self.scales.cpu(),
            "zeros": self.zeros.cpu(),
            "bias": None if self.bias is None else self.bias.cpu(),
            "hp_indices": self.hp_indices.cpu(),
            "hp_weight": self.hp_weight.cpu(),
            "in_features": int(self.in_features),
            "out_features": int(self.out_features),
            "group_size": int(self.group_size),
        }


def _pack_linear_weight_int4(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    layer_key: str,
    high_precision_columns: Set[Tuple[str, int]] | None,
) -> PackedLinearState:
    assert weight.dim() == 2, "Expected (out_features, in_features) weight"
    out_features, in_features = weight.shape
    assert group_size > 0 and in_features % group_size == 0, "in_features must be divisible by group_size"

    hp_set = high_precision_columns or set()
    hp_indices: List[int] = sorted([idx for (k, idx) in hp_set if k == layer_key])
    hp_index_tensor = torch.tensor(hp_indices, dtype=torch.long, device=weight.device)
    hp_weight = (
        weight.index_select(dim=1, index=hp_index_tensor).contiguous()
        if hp_index_tensor.numel() > 0
        else weight.new_zeros((out_features, 0))
    )

    quant_source = weight.clone()
    if hp_index_tensor.numel() > 0:
        quant_source[:, hp_index_tensor] = 0

    n_groups = in_features // group_size
    grouped = quant_source.reshape(out_features, n_groups, group_size).float()
    g_min = grouped.amin(dim=-1, keepdim=True)
    g_max = grouped.amax(dim=-1, keepdim=True)
    scales = (g_max - g_min).clamp(min=1e-5) / 15.0
    zeros = torch.clamp(torch.round(-g_min / scales), 0, 15)

    q = torch.clamp(torch.round(grouped / scales) + zeros, 0, 15).to(torch.uint8)
    q_flat = q.reshape(out_features, in_features)
    qweight = _pack_int4(q_flat)

    return PackedLinearState(
        qweight=qweight,
        scales=scales.squeeze(-1).to(weight.dtype),
        zeros=zeros.squeeze(-1).to(weight.dtype),
        bias=None if bias is None else bias.detach().clone(),
        hp_indices=hp_index_tensor,
        hp_weight=hp_weight,
        in_features=in_features,
        out_features=out_features,
        group_size=group_size,
    )


def dequantize_int4_python(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int,
    in_features: int,
) -> torch.Tensor:
    q = _unpack_int4(qweight, in_features=in_features).float()
    out_features = q.shape[0]
    n_groups = scales.shape[1]
    q = q.reshape(out_features, n_groups, group_size)
    s = scales.float().unsqueeze(-1).clamp(min=_EPS)
    z = zeros.float().unsqueeze(-1)
    w = (q - z) * s
    return w.reshape(out_features, in_features)


def _try_get_cuda_kernel():
    try:
        import asdq_cuda  # type: ignore
        return asdq_cuda
    except Exception:
        return None


class Int4QuantLinear(nn.Module):
    """Runtime int4 Linear with optional CUDA dequant kernel."""

    def __init__(self, packed: PackedLinearState):
        super().__init__()
        self.in_features = packed.in_features
        self.out_features = packed.out_features
        self.group_size = packed.group_size

        self.register_buffer("qweight", packed.qweight.contiguous().to(torch.uint8), persistent=True)
        self.register_buffer("scales", packed.scales.contiguous(), persistent=True)
        self.register_buffer("zeros", packed.zeros.contiguous(), persistent=True)
        self.register_buffer("hp_indices", packed.hp_indices.contiguous(), persistent=True)
        self.register_buffer("hp_weight", packed.hp_weight.contiguous(), persistent=True)

        if packed.bias is not None:
            self.bias = nn.Parameter(packed.bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None

    def _dequant_weight(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        qweight = self.qweight.to(device=device, non_blocking=True)
        scales = self.scales.to(device=device, dtype=dtype, non_blocking=True)
        zeros = self.zeros.to(device=device, dtype=dtype, non_blocking=True)

        kernel = _try_get_cuda_kernel()
        if kernel is not None and qweight.is_cuda:
            w = kernel.dequantize_4bit(qweight, scales, zeros, int(self.group_size), int(self.in_features))
            return w.to(dtype=dtype)

        return dequantize_int4_python(
            qweight=qweight,
            scales=scales,
            zeros=zeros,
            group_size=self.group_size,
            in_features=self.in_features,
        ).to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        weight = self._dequant_weight(device=x.device, dtype=x_dtype)
        bias = self.bias
        if bias is not None:
            bias = bias.to(device=x.device, dtype=x_dtype)
        out = F.linear(x, weight, bias)

        if self.hp_indices.numel() > 0:
            hp_x = x.index_select(dim=-1, index=self.hp_indices.to(x.device))
            hp_w = self.hp_weight.to(device=x.device, dtype=x_dtype)
            out = out + F.linear(hp_x, hp_w, None)
        return out


@torch.no_grad()
def quantize_model_to_int4(
    model: nn.Module,
    q_group_size: int = 128,
    high_precision_columns: Set[Tuple[str, int]] | None = None,
) -> Dict:
    """Replace language-block nn.Linear with Int4QuantLinear and return serialized payload."""
    layers = get_blocks(model)
    payload = {
        "format": _FORMAT_VERSION,
        "layers": {},
    }

    for i, block in enumerate(layers):
        named_linears = get_named_linears(block)
        for linear_name, linear in named_linears.items():
            key = f"layers.{i}.{linear_name}"
            packed = _pack_linear_weight_int4(
                weight=linear.weight.data,
                bias=linear.bias.data if linear.bias is not None else None,
                group_size=q_group_size,
                layer_key=key,
                high_precision_columns=high_precision_columns,
            )
            quant_mod = Int4QuantLinear(packed)
            _set_submodule(block, linear_name, quant_mod)
            payload["layers"][key] = packed.to_dict()
    return payload


@torch.no_grad()
def apply_quantized_payload(model: nn.Module, payload: Dict) -> bool:
    if not isinstance(payload, dict) or payload.get("format") != _FORMAT_VERSION:
        return False
    layer_payload = payload.get("layers", {})
    if not isinstance(layer_payload, dict):
        return False

    layers = get_blocks(model)
    for i, block in enumerate(layers):
        named_linears = get_named_linears(block)
        for linear_name in list(named_linears.keys()):
            key = f"layers.{i}.{linear_name}"
            state = layer_payload.get(key)
            if state is None:
                continue
            packed = PackedLinearState(
                qweight=state["qweight"],
                scales=state["scales"],
                zeros=state["zeros"],
                bias=state["bias"],
                hp_indices=state["hp_indices"],
                hp_weight=state["hp_weight"],
                in_features=int(state["in_features"]),
                out_features=int(state["out_features"]),
                group_size=int(state["group_size"]),
            )
            _set_submodule(block, linear_name, Int4QuantLinear(packed))
    return True
