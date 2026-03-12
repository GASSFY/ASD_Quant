"""Minimal RTN-style weight pseudo quantization for LLaVA-like models."""
import torch
import torch.nn as nn
from tqdm import tqdm

from .quant_funcs import pseudo_quantize_tensor


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


@torch.no_grad()
def pseudo_quantize_model_weight(model, w_bit: int = 4, q_group_size: int = -1, zero_point: bool = True):
    """In-place pseudo quantize Linear weights in model (language blocks only)."""
    layers = get_blocks(model)
    q_config = {"zero_point": zero_point, "q_group_size": q_group_size}
    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bits=w_bit, **q_config
            )
