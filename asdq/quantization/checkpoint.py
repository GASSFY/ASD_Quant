"""Save/load ASDQ checkpoints with compact real-quant (int4) format."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch

from .real_quant import apply_quantized_payload

CHECKPOINT_FORMAT_V2 = "asdq_int4_v2"

_QUANT_BUFFER_SUFFIXES = (
    ".qweight",
    ".scales",
    ".zeros",
    ".hp_indices",
    ".hp_weight",
)


def is_quant_state_key(key: str) -> bool:
    return any(key.endswith(suffix) for suffix in _QUANT_BUFFER_SUFFIXES)


def split_base_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    base: Dict[str, torch.Tensor] = {}
    quant: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if is_quant_state_key(key):
            quant[key] = value
        else:
            base[key] = value
    return base, quant


def tensor_bytes(obj: Any) -> int:
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    return 0


def summarize_checkpoint(path: str) -> Dict[str, Any]:
    """Return byte breakdown for a saved .pt file (for debugging size)."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    summary: Dict[str, Any] = {"path": path, "format": None, "total_bytes": 0, "groups": {}}

    def add_group(name: str, blob: Dict[str, Any]) -> None:
        nbytes = 0
        if isinstance(blob, dict):
            for v in blob.values():
                if isinstance(v, dict):
                    for t in v.values():
                        nbytes += tensor_bytes(t)
                else:
                    nbytes += tensor_bytes(v)
        summary["groups"][name] = nbytes
        summary["total_bytes"] += nbytes

    if not isinstance(state, dict):
        summary["format"] = "raw_state_dict"
        add_group("raw", {"all": state})
        return summary

    summary["format"] = state.get("format", "legacy")
    if state.get("format") == CHECKPOINT_FORMAT_V2:
        add_group("base_state_dict", state.get("base_state_dict", {}))
        add_group("quant_payload", state.get("quant_payload", {}))
    elif "quant_payload" in state and "state_dict" in state:
        add_group("state_dict", state.get("state_dict", {}))
        add_group("quant_payload", state.get("quant_payload", {}))
    elif "state_dict" in state:
        base, quant = split_base_state_dict(state["state_dict"])
        add_group("state_dict_base", base)
        add_group("state_dict_quant", quant)
    else:
        add_group("root", state)
    return summary


def save_checkpoint(
    model: torch.nn.Module,
    path: str,
    *,
    quant_payload: Optional[Dict] = None,
    w_group: int = 128,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state_dict = model.state_dict()

    if quant_payload is not None:
        base_state_dict, _ = split_base_state_dict(state_dict)
        save_obj = {
            "format": CHECKPOINT_FORMAT_V2,
            "base_state_dict": base_state_dict,
            "quant_payload": quant_payload,
            "w_group": int(w_group),
        }
    else:
        save_obj = {"state_dict": state_dict}

    torch.save(save_obj, path)


def load_checkpoint(model: torch.nn.Module, path: str) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)

    if not isinstance(state, dict):
        model.load_state_dict(state, strict=False)
        print(f"[ASDQ] Loaded checkpoint from {path}")
        return

    fmt = state.get("format")
    if fmt == CHECKPOINT_FORMAT_V2:
        base_state_dict = state.get("base_state_dict")
        if base_state_dict:
            model.load_state_dict(base_state_dict, strict=False)
        ok = apply_quantized_payload(model, state.get("quant_payload", {}))
        print(f"[ASDQ] Loaded v2 int4 checkpoint from {path} (quant_payload applied={ok})")
        return

    if "quant_payload" in state and "state_dict" in state:
        sd = state["state_dict"]
        if any(is_quant_state_key(k) for k in sd):
            model.load_state_dict(sd, strict=False)
            print(f"[ASDQ] Loaded legacy checkpoint from {path} (packed state_dict only)")
        else:
            model.load_state_dict(sd, strict=False)
            ok = apply_quantized_payload(model, state["quant_payload"])
            print(f"[ASDQ] Loaded legacy checkpoint from {path} (base + quant_payload, applied={ok})")
        return

    if "quant_payload" in state:
        base_state_dict = state.get("base_state_dict")
        if base_state_dict:
            model.load_state_dict(base_state_dict, strict=False)
        ok = apply_quantized_payload(model, state["quant_payload"])
        print(f"[ASDQ] Loaded checkpoint from {path} (quant_payload applied={ok})")
        return

    if "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=False)
        print(f"[ASDQ] Loaded checkpoint from {path}")
        return

    model.load_state_dict(state, strict=False)
    print(f"[ASDQ] Loaded checkpoint from {path}")
