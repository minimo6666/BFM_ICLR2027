"""Explicit context/self decomposition for the direct-X0 BFM.

The Transformer predicts only ``c_theta``. This module reconstructs the full
clean-X0 logit

    z_theta = c_theta + (2 * X_t - 1) * log((2T - t) / t)

before the existing expectation-consistent BFM loss or reverse kernel sees it.
The analytic term has no parameters and physical ``t=0`` is rejected.
"""

from __future__ import annotations

import torch
from torch import nn

from models.binarylatent_flow_expectation_consistent_retrain_tminus1 import (
    BinaryDiffusionFlowDecouple,
)


def analytic_self_evidence(
    physical_time: torch.Tensor,
    *,
    total_steps: int = 64,
    reference: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return broadcastable ``d_t = log((2T-t)/t)`` for physical t in [1,T]."""
    if not torch.is_tensor(physical_time):
        physical_time = torch.as_tensor(physical_time)
    if physical_time.ndim == 0:
        physical_time = physical_time.unsqueeze(0)
    if torch.any(physical_time < 1) or torch.any(physical_time > total_steps):
        raise ValueError(
            f"analytic self evidence requires physical t in [1,{total_steps}], "
            f"got range [{int(physical_time.min())},{int(physical_time.max())}]"
        )
    device = reference.device if reference is not None else physical_time.device
    t = physical_time.to(device=device, dtype=torch.float32)
    d = torch.log((2.0 * float(total_steps) - t) / t)
    if not torch.isfinite(d).all():
        raise FloatingPointError("non-finite analytic self evidence")
    ndim = reference.ndim if reference is not None else 3
    return d.view(-1, *([1] * (ndim - 1)))


class ContextSelfDecomposedDenoiser(nn.Module):
    """Expose full logits while retaining direct access to context logits."""

    def __init__(self, context_denoiser: nn.Module, total_steps: int):
        super().__init__()
        self.context_denoiser = context_denoiser
        self.total_steps = int(total_steps)
        self.last_decomposition: dict[str, torch.Tensor | bool] = {}

    def context_logits(self, idx, label=None, time_steps=None):
        return self.context_denoiser(idx, label=label, time_steps=time_steps)

    def decompose(self, idx, label=None, time_steps=None):
        if time_steps is None:
            raise ValueError("CS-BFM requires the baseline t-1 network timestep")
        c = self.context_logits(idx, label=label, time_steps=time_steps)
        physical_time = time_steps.to(dtype=torch.long) + 1
        d = analytic_self_evidence(
            physical_time, total_steps=self.total_steps, reference=c
        )
        s = 2.0 * idx.float() - 1.0
        z = c.float() + s * d
        return c, d, z

    def forward(self, idx, label=None, time_steps=None, **kwargs):
        del kwargs
        c, d, z = self.decompose(idx, label=label, time_steps=time_steps)
        self.last_decomposition = {
            "c": c.detach(),
            "d": d.detach(),
            "z": z.detach(),
            "d_requires_grad": bool(d.requires_grad),
        }
        return z


class BinaryDiffusionFlowCSDecomposed(BinaryDiffusionFlowDecouple):
    """Original BFM with only its clean-X0 logit parameterization changed."""

    cs_decomposed = True

    def __init__(self, H, denoise_fn, mask_id):
        if bool(H.p_flip):
            raise ValueError("CS-BFM requires p_flip=False")
        if float(H.focal) >= 0.0:
            raise ValueError("CS-BFM requires focal=-1 (plain BCE)")
        if float(H.aux) != 0.0:
            raise ValueError("CS-BFM requires aux=0")
        if str(H.loss_final) != "mean":
            raise ValueError("CS-BFM requires unweighted mean BCE")
        if int(H.total_steps) != 64:
            raise ValueError("This controlled CS-BFM experiment requires T=64")
        wrapped = ContextSelfDecomposedDenoiser(denoise_fn, H.total_steps)
        super().__init__(H, wrapped, mask_id)
        if float(self.barrier_retention_scale) != 1.0:
            raise ValueError("CS-BFM requires the original scale=1 Bernoulli path")
        if any(name == "d" or name.endswith(".d") for name, _ in self.named_parameters()):
            raise RuntimeError("analytic d_t must not be an optimizer parameter")

    @property
    def context_denoiser(self) -> nn.Module:
        return self._denoise_fn.context_denoiser

    def context_logits(self, x_t, physical_time, label=None):
        return self._denoise_fn.context_logits(
            x_t, label=label, time_steps=physical_time.long() - 1
        )

    def decomposed_logits(self, x_t, physical_time, label=None):
        return self._denoise_fn.decompose(
            x_t, label=label, time_steps=physical_time.long() - 1
        )

    def _train_loss(self, x_0, label=None, x_ct=None):
        stats = super()._train_loss(x_0, label=label, x_ct=x_ct)
        decomposition = self._denoise_fn.last_decomposition
        c = decomposition["c"].float()
        d = decomposition["d"].float()
        z = decomposition["z"].float()
        stats.update(
            {
                "mean_c": c.mean(),
                "std_c": c.std(unbiased=False),
                "mean_abs_c": c.abs().mean(),
                "mean_d": d.mean(),
                "mean_abs_d": d.abs().mean(),
                "mean_z": z.mean(),
                "std_z": z.std(unbiased=False),
                "d_requires_grad": float(bool(decomposition["d_requires_grad"])),
            }
        )
        return stats

