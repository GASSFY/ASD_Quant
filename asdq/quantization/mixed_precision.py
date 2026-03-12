"""
混合精度：基于全局 ASD 排序 + 全局比例上限。

共识：
- 对所有要量化的层，用校准激活算每通道 ASD → (层, 通道, ASD)。
- 全模型所有通道的 ASD 一起排序，取前 ratio（如 0.1）标为「高精度」。
- 量化时：高精度通道对应权重列用 high_w_bit，其余用 low_w_bit。
"""
from __future__ import annotations

from typing import Any

import torch

from asdq.metrics import compute_ASD, get_ASD_kwargs


def compute_global_asd_list(
    layer_activations: dict[str, torch.Tensor],
    asd_kwargs: dict[str, Any] | None = None,
) -> list[tuple[str, int, float]]:
    """
    对所有层的激活算每通道 ASD，汇总为全局列表。

    layer_activations: 层名 -> 激活张量 [N, C]，C 为输入通道数。
    asd_kwargs: 传给 compute_ASD 的参数（theta1, theta2, k_method, psi_method, normalize 等）。

    Returns:
        [(layer_key, channel_idx, asd_value), ...]，未排序。
    """
    asd_kwargs = asd_kwargs or {}
    out: list[tuple[str, int, float]] = []
    for layer_key, x in layer_activations.items():
        if x.numel() == 0 or x.shape[-1] == 0:
            continue
        x_2d = x.float().view(-1, x.shape[-1])
        asd = compute_ASD(x_2d, **asd_kwargs)  # [C]
        for c in range(asd.shape[0]):
            out.append((layer_key, c, asd[c].item()))
    return out


def select_high_precision_columns(
    global_asd_list: list[tuple[str, int, float]],
    ratio: float,
) -> set[tuple[str, int]]:
    """
    按 ASD 从高到低排序，取前 ratio 比例的 (layer_key, channel_idx) 为高精度。

    ratio: 全模型高精度通道比例，例如 0.1 表示最多 10% 通道高精度。
    Returns:
        set of (layer_key, channel_idx) 享受高精度。
    """
    if ratio <= 0 or not global_asd_list:
        return set()
    if ratio >= 1.0:
        return {(k, c) for k, c, _ in global_asd_list}

    sorted_list = sorted(global_asd_list, key=lambda t: t[2], reverse=True)
    n_total = len(sorted_list)
    n_high = max(1, int(round(n_total * ratio)))
    return {(t[0], t[1]) for t in sorted_list[:n_high]}
