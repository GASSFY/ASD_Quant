"""Evaluation-time model loading: FP16 baseline, pseudo-quant, or real int4 (v2)."""
from __future__ import annotations

import gc
import os
from typing import Any, Optional, Type

import torch
import torch.nn as nn

from .checkpoint import load_checkpoint, peek_checkpoint_format
from .quantize import get_blocks, get_named_linears
from .real_quant import Int4QuantLinear


def resolve_eval_load_mode(
    scale_path: Optional[str],
    *,
    real_quant: bool = False,
    pseudo_quant: bool = False,
) -> str:
    """Return ``fp16``, ``real_quant``, or ``pseudo_quant``."""
    if not scale_path or not os.path.exists(scale_path):
        return "fp16"
    if real_quant and pseudo_quant:
        raise ValueError("real_quant and pseudo_quant are mutually exclusive.")
    if real_quant:
        return "real_quant"
    # v2 int4 checkpoints must use the real-quant load path even if pseudo_quant
    # is true in a shared yaml (main_quant default).
    if peek_checkpoint_format(scale_path) == "v2":
        return "real_quant"
    if pseudo_quant:
        return "pseudo_quant"
    return "pseudo_quant"


def _target_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def log_real_quant_stats(model: nn.Module) -> None:
    layers = get_blocks(model)
    n_int4 = 0
    n_linear = 0
    for block in layers:
        for _, mod in get_named_linears(block).items():
            n_linear += 1
            if isinstance(mod, Int4QuantLinear):
                n_int4 += 1
    print(f"[ASDQ] Int4QuantLinear layers: {n_int4}/{n_linear}")
    if torch.cuda.is_available():
        alloc_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"[ASDQ] GPU memory: allocated={alloc_gb:.2f} GiB, reserved={reserved_gb:.2f} GiB")


def load_model_for_eval(ModelClass: Type[Any], args: Any) -> Any:
    """Load lmms-eval model wrapper for evaluation with the correct quant branch."""
    scale_path = getattr(args, "scale_path", None)
    mode = resolve_eval_load_mode(
        scale_path,
        real_quant=getattr(args, "real_quant", False),
        pseudo_quant=getattr(args, "pseudo_quant", False),
    )
    print(f"[ASDQ] Eval load mode: {mode}")

    model_args = getattr(args, "model_args", "") or ""
    batch_size = getattr(args, "batch_size", "1")
    device = getattr(args, "device", None)

    if mode == "fp16":
        return ModelClass.create_from_arg_string(
            model_args,
            {"batch_size": batch_size, "device": device},
        )

    if mode == "pseudo_quant":
        lm = ModelClass.create_from_arg_string(
            model_args,
            {"batch_size": batch_size, "device": device},
        )
        load_checkpoint(lm._model, scale_path)
        return lm

    # real_quant: load on CPU first so GPU never holds full FP16 LLM weights.
    lm = ModelClass.create_from_arg_string(
        model_args,
        {"batch_size": batch_size, "device": "cpu"},
    )
    load_checkpoint(lm._model, scale_path)
    target = _target_device(device)
    if target != "cpu":
        lm._model.to(target)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    log_real_quant_stats(lm._model)
    return lm
