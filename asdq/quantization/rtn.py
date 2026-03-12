"""Minimal RTN-style weight pseudo quantization for LLaVA-like models."""
from __future__ import annotations

from typing import Set, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from .quant_funcs import (
    pseudo_quantize_tensor,
    pseudo_quantize_weight_per_column,
    pseudo_quantize_weight_spqr_style,
)


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_blocks(model):
    """Return list of block modules (language part) for LLaVA / Qwen2-VL style."""
    cls_name = model.__class__.__name__
    if cls_name in ("LlavaLlamaForCausalLM", "LlavaQwenForCausalLM", "LlavaLlamaModel"):
        return model.model.layers
    if "Llama" in cls_name and hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if "Qwen2" in cls_name and hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if "InternVL" in cls_name and hasattr(model, "language_model"):
        return model.language_model.model.layers
    raise NotImplementedError(f"get_blocks not implemented for {cls_name}")


def _linear_layer_key(block_idx: int, linear_name: str) -> str:
    """与 mixed_precision 中 layer_activations 的 key 一致，便于查找高精度列。"""
    return f"layers.{block_idx}.{linear_name}"


@torch.no_grad()
def pseudo_quantize_model_weight(
    model,
    w_bit: int = 4,
    q_group_size: int = -1,
    zero_point: bool = True,
    high_precision_columns: Set[Tuple[str, int]] | None = None,
    high_w_bit: int = 8,
    low_w_bit: int = 4,
):
    """
    In-place pseudo quantize Linear weights in model (language blocks only).

    若提供 high_precision_columns，则采用 SpQR 风格混合精度：
    分组 = 一行×一坨列（q_group_size）；组内保存精度列剔除后用剩余位置算 scale/zero（填均值再 fit），
    量化后保存位置用原权重重写；输出权重已是「量化部分 + outlier」合并结果，推理时直接 matmul 即可。
    """
    layers = get_blocks(model)
    q_config = {"zero_point": zero_point, "q_group_size": q_group_size}
    use_mixed = high_precision_columns is not None
    if use_mixed and q_group_size <= 0:
        q_group_size = 128

    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            w = m.weight.data  # (out_features, in_features)
            if use_mixed and high_precision_columns is not None:
                key = _linear_layer_key(i, n)
                m.weight.data = pseudo_quantize_weight_spqr_style(
                    w,
                    q_group_size=q_group_size,
                    high_precision_columns=high_precision_columns,
                    layer_key=key,
                    n_bits=low_w_bit,
                    zero_point=zero_point,
                )
            else:
                m.weight.data = pseudo_quantize_tensor(w, n_bits=w_bit, **q_config)
