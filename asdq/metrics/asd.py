"""
ASD (Activation-aware Statistical sensitivity for quantization) metrics.

Per-channel K (absolute significance) and Psi (relative significance) with
multiple computation schemes. ASD_c = theta1 * K_c + theta2 * Psi_c.

References: MBQ (get_act_scale, get_weak_column_mask), SmoothQuant (gen_act_scales).
"""
from __future__ import annotations

from typing import Any

import torch

_EPS = 1e-8

# Supported method names for config
K_METHODS = ("mean", "max", "l2")
PSI_METHODS = ("mean", "max", "zscore")

# Preset combinations (plan section 四). Default: K2 + Psi1.
ASD_PRESETS = {
    "default": {"k_method": "mean", "psi_method": "mean"},   # K2 + Psi1, most stable
    "max_max": {"k_method": "max", "psi_method": "max"},     # K1 + Psi2, SmoothQuant/OWQ style
    "mean_max": {"k_method": "mean", "psi_method": "max"},   # K2 + Psi2, single peak emphasis
}


def get_ASD_kwargs(
    preset: str | None = None,
    k_method: str | None = None,
    psi_method: str | None = None,
    theta1: float = 0.5,
    theta2: float = 0.5,
    normalize: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Build kwargs for compute_ASD. Config switch: use preset name or explicit k_method/psi_method.

    preset: "default" (K2+Psi1), "max_max" (K1+Psi2), "mean_max" (K2+Psi2), or None.
    k_method / psi_method: override preset when given.
    """
    out = {"theta1": theta1, "theta2": theta2, "normalize": normalize, **kwargs}
    if preset is not None:
        if preset not in ASD_PRESETS:
            raise ValueError(f"Unknown ASD preset: {preset}. Use one of {list(ASD_PRESETS)}.")
        out.update(ASD_PRESETS[preset])
    if k_method is not None:
        out["k_method"] = k_method
    if psi_method is not None:
        out["psi_method"] = psi_method
    return out


def asd_kwargs_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Build compute_ASD kwargs from a config dict (e.g. from yaml or argparse).

    Expected keys: asd_preset, asd_k_method, asd_psi_method, asd_theta1, asd_theta2, asd_normalize.
    Missing keys are omitted (get_ASD_kwargs defaults apply).
    """
    return get_ASD_kwargs(
        preset=config.get("asd_preset"),
        k_method=config.get("asd_k_method"),
        psi_method=config.get("asd_psi_method"),
        theta1=config.get("asd_theta1", 0.5),
        theta2=config.get("asd_theta2", 0.5),
        normalize=config.get("asd_normalize", True),
    )


@torch.no_grad()
def compute_K(x: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """
    Per-channel absolute significance K_c (how significant this input channel is).

    x: [N, C] activation tensor (e.g. Linear layer input).
    method: "mean" (K2), "max" (K1), "l2" (K3).
    Returns: [C] tensor.
    """
    x = x.float()
    if x.dim() != 2:
        x = x.view(-1, x.shape[-1])
    n, c = x.shape

    if method == "max":
        # K1: SmoothQuant / OWQ style — max over N
        K = x.abs().amax(dim=0)
    elif method == "mean":
        # K2: MBQ / AWQ style — mean abs over N
        K = x.abs().view(-1, c).mean(dim=0)
    elif method == "l2":
        # K3: L2 norm over N
        K = x.pow(2).sum(dim=0).clamp(min=_EPS).sqrt()
    else:
        raise ValueError(f"Unknown K method: {method}. Use one of {K_METHODS}.")

    return K


@torch.no_grad()
def compute_Psi(s: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """
    Per-channel relative significance Psi_c (how significant this channel is vs others).

    s: [C] channel-wise significance (e.g. K from compute_K).
    method: "mean" (Psi1), "max" (Psi2), "zscore" (Psi3).
    Returns: [C] tensor.
    """
    s = s.float()
    if s.dim() != 1:
        s = s.view(-1)

    if method == "mean":
        # Psi1: s_c / mean(s)
        s_mean = s.mean().clamp(min=_EPS)
        Psi = s / s_mean
    elif method == "max":
        # Psi2: s_c / max(s)
        s_max = s.amax().clamp(min=_EPS)
        Psi = s / s_max
    elif method == "zscore":
        # Psi3: (s - mean(s)) / std(s), then clamp to non-negative
        s_mean = s.mean()
        s_std = s.std().clamp(min=_EPS)
        Psi = ((s - s_mean) / s_std).clamp(min=0.0)
    else:
        raise ValueError(f"Unknown Psi method: {method}. Use one of {PSI_METHODS}.")

    return Psi


@torch.no_grad()
def compute_ASD(
    x: torch.Tensor,
    theta1: float = 0.5,
    theta2: float = 0.5,
    k_method: str = "mean",
    psi_method: str = "mean",
    normalize: bool = True,
) -> torch.Tensor:
    """
    Per-channel ASD score: ASD_c = theta1 * K_c + theta2 * Psi_c.

    x: [N, C] activation tensor.
    theta1, theta2: weights for K and Psi (e.g. 0.5, 0.5).
    k_method: "mean" (K2), "max" (K1), "l2" (K3).
    psi_method: "mean" (Psi1), "max" (Psi2), "zscore" (Psi3).
    normalize: if True, scale K and Psi to [0, 1] per tensor so theta1/theta2 are comparable.
    Returns: [C] tensor. Higher ASD_c -> preserve precision for weight column c.
    """
    K = compute_K(x, method=k_method)
    # Use same K as base for Psi (s = K) as in the plan
    Psi = compute_Psi(K, method=psi_method)

    if normalize:
        k_max = K.amax().clamp(min=_EPS)
        K = K / k_max
        p_max = Psi.amax().clamp(min=_EPS)
        Psi = Psi / p_max

    ASD = theta1 * K + theta2 * Psi
    return ASD
