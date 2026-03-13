"""
Streaming Hessian diagonal collection for ASD importance ranking.

Collects diag(H)_c = E[x_c^2] per Linear layer via forward hooks, without
storing raw activations. Memory cost: O(layers * C) instead of O(layers * N * C).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from asdq.quantization.quantize import get_blocks, get_named_linears, _linear_layer_key


def collect_hessian_diag(
    model: nn.Module,
    forward_fn,
    forward_kwargs_list: list[dict],
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """
    Run calibration forward passes and collect the Hessian diagonal for each
    Linear layer via streaming (no raw activation storage).

    model: root model with .model.layers (e.g. process_model.model).
    forward_fn: callable that executes one forward pass, signature forward_fn(**kwargs).
    forward_kwargs_list: list of kwargs dicts for each calibration batch.

    Returns:
        {layer_key: diag_H} where diag_H is shape [C] = E[x_c^2].
    """
    sum_x2: dict[str, torch.Tensor] = {}
    count: dict[str, int] = {}
    handles = []

    def _make_hook(key: str):
        def hook(_module, args, _kwargs):
            x = args[0]
            if not isinstance(x, torch.Tensor):
                return
            x = x.detach().float().view(-1, x.shape[-1])  # (N_tokens, C)
            if x.numel() == 0:
                return
            x2 = x.pow(2).sum(dim=0)  # (C,)
            n = x.shape[0]
            if key not in sum_x2:
                sum_x2[key] = x2
                count[key] = n
            else:
                sum_x2[key] = sum_x2[key] + x2
                count[key] += n
        return hook

    layers = get_blocks(model)
    for i in range(len(layers)):
        for name, linear in get_named_linears(layers[i]).items():
            key = _linear_layer_key(i, name)
            handles.append(linear.register_forward_hook(_make_hook(key)))

    try:
        for kwargs in forward_kwargs_list:
            if device is not None:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in kwargs.items()
                }
            else:
                batch = kwargs
            forward_fn(**batch)
            del batch
            torch.cuda.empty_cache()
    finally:
        for h in handles:
            h.remove()

    return {key: sum_x2[key] / count[key] for key in sum_x2}
