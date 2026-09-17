"""BFM + BitDance joint endpoint predictor + V4 multi-NFE sensitivity regularizer.

This is the controlled sensitivity arm for the BitDance-joint experiment.

What is kept EXACTLY from ``BinaryDiffusionFlowBitDanceJoint``:
  * outer BFM Bernoulli corruption and t-1 Transformer conditioning,
  * 24-layer Transformer features,
  * BitDance p=1 64-D joint DiffHead architecture,
  * BitDance native RF/velocity-MSE base objective,
  * BitDance inner-time distribution,
  * 20-step Euler-Maruyama + final Euler inner sampler,
  * hard joint X0 endpoint sampling and the exact BFM endpoint bridge.

The ONLY training-objective addition is the V4-style outer sampling-risk term:

    L = L_RF + lambda * W_multi(t) * MSE_x0_equiv

where ``W_multi(t)`` is copied exactly from the existing BFM V4 construction:
it aggregates normalized S_{s,t}^2 over the actual np.linspace sampler grids of
the requested deployment NFEs.  By default V4 supports 64,32,16,8; the launcher
for the present 64-step controlled comparison sets ``BFM_V4_NFES=64``.

The BitDance head predicts signed clean endpoints y0=2*x0-1.  To keep the
auxiliary on the same 0/1 squared-error scale as the original V4 Brier/MSE,
we use

    MSE_x0_equiv = 1/4 * mean((y0_hat - y0)^2).

This is an adaptation of the V4 allocation rule to the joint RF predictor.  It
is not claimed that the original scalar Bernoulli SRC excess-risk identity is
literally the RF objective; the experiment tests whether the analytically
known outer BFM sensitivity profile improves training once the endpoint model
can represent within-cell joint bit structure.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import torch

from models.binarylatent_flow_bitdance_joint import BinaryDiffusionFlowBitDanceJoint


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    return float(value) if value else float(default)


class BinaryDiffusionFlowBitDanceJointSRCV4(BinaryDiffusionFlowBitDanceJoint):
    """BitDance-joint BFM with the original V4 multi-NFE S^2 allocation table."""

    def __init__(self, H, denoise_fn, mask_id):
        super().__init__(H, denoise_fn, mask_id)

        if float(self.aux) != 0.0:
            raise ValueError(
                "BitDance-joint SRC-V4 requires --aux 0. Do not mix the old "
                "posterior auxiliary into this controlled experiment."
            )

        self.v4_lambda = _env_float("BFM_V4_LAMBDA", 0.25)
        if self.v4_lambda < 0.0:
            raise ValueError("BFM_V4_LAMBDA must be non-negative.")

        nfe_text = os.environ.get("BFM_V4_NFES", "64,32,16,8")
        nfe_values = [int(x.strip()) for x in nfe_text.split(",") if x.strip()]
        if not nfe_values:
            raise ValueError("BFM_V4_NFES must contain at least one NFE value.")
        if len(set(nfe_values)) != len(nfe_values):
            raise ValueError("BFM_V4_NFES must not contain duplicates.")
        for k in nfe_values:
            if k < 2 or k > self.num_timesteps:
                raise ValueError(
                    f"Each V4 NFE must be in [2,{self.num_timesteps}], got {k}."
                )
        self.v4_nfes = tuple(nfe_values)

        pi_text = os.environ.get("BFM_V4_NFE_WEIGHTS", "")
        if pi_text.strip():
            pi = [float(x.strip()) for x in pi_text.split(",") if x.strip()]
            if len(pi) != len(self.v4_nfes):
                raise ValueError(
                    "BFM_V4_NFE_WEIGHTS must have the same number of entries as "
                    "BFM_V4_NFES."
                )
            if any(x < 0.0 for x in pi) or sum(pi) <= 0.0:
                raise ValueError(
                    "BFM_V4_NFE_WEIGHTS must be non-negative and sum > 0."
                )
            pi_array = np.asarray(pi, dtype=np.float64)
            pi_array = pi_array / pi_array.sum()
        else:
            pi_array = np.full(
                len(self.v4_nfes), 1.0 / len(self.v4_nfes), dtype=np.float64
            )
        self.v4_nfe_weights = tuple(float(x) for x in pi_array.tolist())

        weight_table = torch.zeros(self.num_timesteps + 1, dtype=torch.float64)
        mean_s2_by_nfe: Dict[int, float] = {}
        grids_by_nfe: Dict[int, tuple[int, ...]] = {}

        # This block intentionally mirrors BinaryDiffusionFlowMultiNFESRC V4.
        for k, pi_k in zip(self.v4_nfes, self.v4_nfe_weights):
            grid = self._sampling_grid_v4(k)
            intervals = [
                (int(grid[i]), int(grid[i + 1]))
                for i in range(len(grid) - 1)
            ]
            if not intervals:
                raise RuntimeError(f"NFE={k} produced no probabilistic intervals.")

            s2_values = [
                self._sensitivity_squared_v4(t_target, t_current)
                for t_current, t_target in intervals
            ]
            mean_s2 = float(np.mean(s2_values))
            if (not np.isfinite(mean_s2)) or mean_s2 <= 0.0:
                raise RuntimeError(f"Invalid mean S^2 for NFE={k}: {mean_s2}")

            mean_s2_by_nfe[k] = mean_s2
            grids_by_nfe[k] = tuple(int(x) for x in grid.tolist())

            m_k = float(len(intervals))
            for (t_current, _), raw_s2 in zip(intervals, s2_values):
                normalized_s2 = raw_s2 / mean_s2
                weight_table[t_current] += (
                    self.num_timesteps * float(pi_k) / m_k * normalized_s2
                )

        # Same V4 convention: t=1 is the hard final source and gets no SRC aux.
        weight_table[1] = 0.0
        mean_weight = float(weight_table[1:].mean().item())
        if not np.isfinite(mean_weight) or abs(mean_weight - 1.0) > 1e-8:
            raise RuntimeError(
                "V4 weight-table normalization failed: expected uniform-t mean 1, "
                f"got {mean_weight:.12f}."
            )

        self.v4_mean_weight = mean_weight
        self.v4_max_weight = float(weight_table[1:].max().item())
        self.v4_mean_s2_by_nfe = mean_s2_by_nfe
        self.v4_grids_by_nfe = grids_by_nfe
        self.register_buffer(
            "v4_weight_table", weight_table.float(), persistent=False
        )

    def _sampling_grid_v4(self, sample_steps: int) -> np.ndarray:
        """Mirror the current BitDance-BFM/BFM np.linspace outer grid exactly."""
        sampling_steps = np.arange(1, self.num_timesteps + 1)
        if int(sample_steps) != self.num_timesteps:
            idx = np.linspace(
                0.0, self.num_timesteps - 1, int(sample_steps)
            ).astype(np.int64)
            sampling_steps = sampling_steps[idx]
        return sampling_steps[::-1]

    def _sensitivity_squared_v4(self, s: int, t: int) -> float:
        """Exact BFM S_{s,t}^2 used by the original V4 SRC implementation."""
        if not (1 <= int(s) < int(t) <= self.num_timesteps):
            raise ValueError(f"Need 1 <= s < t <= T, got s={s}, t={t}.")
        rho_s = float(self.interpolation_t[int(s)].item())
        rho_t = float(self.interpolation_t[int(t)].item())
        denom = rho_s * (1.0 - rho_t * rho_t)
        if abs(denom) <= 1e-15:
            raise RuntimeError(
                f"Degenerate sensitivity denominator for s={s}, t={t}."
            )
        sensitivity = (rho_s * rho_s - rho_t * rho_t) / denom
        return float(sensitivity * sensitivity)

    def _train_loss(self, x_0, label=None, x_ct=None):
        if x_ct is not None:
            raise NotImplementedError(
                "x_ct-conditioned training is not implemented in BitDance-joint SRC-V4."
            )

        x_0 = x_0.float()
        b, device = x_0.shape[0], x_0.device
        t = self.sample_time(b, device)
        x_t_prob = self.q_sample(x_0, t)
        x_t_in = torch.bernoulli(x_t_prob)

        if label is not None and self.guidance and np.random.random() < 0.1:
            label = None

        features = self._global_features(x_t_in, t, label=label)
        target_signed = x_0.mul(2.0).sub(1.0)

        # [B,N,64] -> [B*N,64], preserving the 64-D within-cell joint target.
        target_flat = target_signed.reshape(-1, target_signed.shape[-1])
        cond_flat = features.reshape(-1, features.shape[-1])

        # One outer-t weight belongs to every spatial cell of that image.
        n_cells = int(target_signed.shape[1])
        image_weight = self.v4_weight_table[t].to(
            device=device, dtype=torch.float32
        )
        row_weight = (
            image_weight[:, None]
            .expand(b, n_cells)
            .reshape(-1)
            .contiguous()
        )

        if self.diff_batch_mul > 1:
            target_flat = target_flat.repeat(self.diff_batch_mul, 1)
            cond_flat = cond_flat.repeat(self.diff_batch_mul, 1)
            row_weight = row_weight.repeat(self.diff_batch_mul)

        # Same RF forward as the pure BitDance-joint arm.  We only request the
        # differentiable per-row endpoint MSE from that same forward; no second
        # network evaluation is introduced.
        stats = self.diff_head.training_loss(
            target_flat,
            cond_flat,
            return_per_row=True,
        )
        base_rf_loss = stats["loss"]
        per_row_endpoint_mse_signed = stats.pop("_per_row_endpoint_mse")
        stats.pop("_per_row_rf_loss", None)

        # y=2x-1 => (y_hat-y)^2 / 4 is the corresponding 0/1-scale squared error.
        per_row_x0_mse_equiv = 0.25 * per_row_endpoint_mse_signed
        v4_src_loss = (row_weight * per_row_x0_mse_equiv).mean()
        total_loss = base_rf_loss + self.v4_lambda * v4_src_loss

        stats["loss"] = total_loss
        # Keep rf_loss semantically equal to the untouched BitDance base RF loss.
        stats["rf_loss"] = base_rf_loss.detach()
        stats["v4_src_loss"] = v4_src_loss.detach()
        stats["v4_plain_x0_mse"] = per_row_x0_mse_equiv.mean().detach()
        stats["v4_weight_mean_batch"] = row_weight.mean().detach()
        stats["v4_weight_max_batch"] = row_weight.max().detach()
        stats["v4_active_fraction"] = (row_weight > 0).float().mean().detach()
        stats["outer_t_mean"] = t.float().mean().detach()
        stats["diff_batch_mul"] = torch.tensor(
            float(self.diff_batch_mul), device=device
        )
        return stats
