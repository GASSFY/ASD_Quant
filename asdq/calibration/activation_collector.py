"""
校准阶段：跑前向并收集每层 Linear 的输入激活，用于后续算 ASD 与混合精度。

与 rtn 约定一致：layer_key = "layers.{block_idx}.{linear_name}"，
以便 select_high_precision_columns 得到的 (layer_key, channel_idx) 能直接用于量化。
"""
from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn

from asdq.quantization.rtn import get_blocks, get_named_linears, _linear_layer_key


def collect_layer_activations(
    model: nn.Module,
    forward_fn,
    forward_kwargs_list: list[dict],
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """
    对多轮前向收集每层 Linear 的输入激活，按 layer_key 拼接为 [N, C]。

    model: 带 .model.layers 的根模型（如 process_model.model）。
    forward_fn: 执行一次前向的可调用对象，签名为 forward_fn(**kwargs) -> Any；
                通常为 process_model.forward，kwargs 为 prompt_inputs | prompt_kwargs。
    forward_kwargs_list: 每轮前向的 kwargs 列表（如多批校准数据）。

    Returns:
        {layer_key: tensor of shape (N_total, C)}，C 为该层输入通道数。
    """
    storage: dict[str, list[torch.Tensor]] = defaultdict(list)
    handles = []

    def _make_hook(key: str):
        def hook(_module, args, _kwargs):
            x = args[0]
            if isinstance(x, torch.Tensor):
                # (batch, seq, in_features) -> (N, C)
                x = x.detach().float().view(-1, x.shape[-1])
                if x.numel() > 0:
                    storage[key].append(x)
        return hook

    layers = get_blocks(model)
    for i in range(len(layers)):
        for name, linear in get_named_linears(layers[i]).items():
            key = _linear_layer_key(i, name)
            handles.append(linear.register_forward_hook(_make_hook(key)))

    try:
        for kwargs in forward_kwargs_list:
            # 确保在正确设备上
            if device is not None:
                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor) and v.device.type != torch.device(device).type:
                        kwargs[k] = v.to(device)
            forward_fn(**kwargs)
    finally:
        for h in handles:
            h.remove()

    out = {}
    for key, parts in storage.items():
        if parts:
            out[key] = torch.cat(parts, dim=0)
    return out
