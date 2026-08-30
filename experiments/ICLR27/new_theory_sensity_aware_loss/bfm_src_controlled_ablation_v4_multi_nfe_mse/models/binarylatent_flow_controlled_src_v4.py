"""
Controlled BFM variants for the first real sensitivity-loss ablation.

Scientific design
-----------------
Variant 0: current public BFM (unchanged)
    train time conditioning: t
    sample time conditioning: t
    loss: original X0/flip objective
    posterior: expectation-consistent

Variant 1: time-aligned BFM
    ONLY conceptual change from Variant 0:
        denoiser time conditioning t -> t-1 in BOTH train and sample
    plus the corresponding weighted-loss index uses t-1, matching BLD.
    For loss_final='mean' this weight change is inactive.

Variant 2/3 inherit Variant 1's sampler EXACTLY.
They differ ONLY in the training loss:
    brier control:
        L = L_base + lambda * Brier
    SRC:
        L = L_base + lambda * normalized[S(t-1,t)^2] * Brier

No extra forward pass is used.
The same t, same X_t, same network output m_theta are used by base loss and
the auxiliary loss. Thus Variant 2 vs 3 isolates sensitivity weighting.

Primary SRC theory in this file is ONE-STEP:
    s = t-1, t>=2.
This deliberately matches the unchanged 64-NFE sampler, where every noisy
time is a node. The final 1->0 hard projection is excluded from SRC and remains
trained only by the original base objective.

This is the clean first experiment. Low-NFE interval-matched SRC should be a
separate later experiment, not mixed into this one.
"""

from __future__ import annotations

import os
import pdb
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from models.binarylatent_flow_expectation_consistent_retrain import (
    BinaryDiffusionFlowDecouple,
    focal_loss,
)


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


class BinaryDiffusionFlowTimeAligned(BinaryDiffusionFlowDecouple):
    """
    Current corrected BFM with ONLY the time convention aligned to BLD:
        physical X_t -> Transformer time_steps=t-1
    in both training and sampling.
    """

    @staticmethod
    def _model_time(t: torch.Tensor) -> torch.Tensor:
        return t - 1

    def _prepare_train_state(self, x_0, label=None, x_ct=None) -> Dict[str, torch.Tensor]:
        x_0 = x_0.float()
        b, device = x_0.size(0), x_0.device

        # Identical physical-time sampling and corruption as current BFM.
        t = self.sample_time(b, device)
        if x_ct is None:
            x_t_prob = self.q_sample(x_0, t)
        else:
            raise NotImplementedError(
                "x_ct-conditioned training is not implemented in this sampler."
            )
        x_t_in = torch.bernoulli(x_t_prob)

        local_label = label
        if local_label is not None:
            if self.guidance and np.random.random() < 0.1:
                local_label = None
            raw_logits = self._denoise_fn(
                idx=x_t_in,
                label=local_label,
                time_steps=self._model_time(t),
            )
        else:
            raw_logits = self._denoise_fn(
                x_t_in,
                time_steps=self._model_time(t),
            )

        # Preserve exact current BFM p_flip semantics.
        if self.p_flip:
            clean_logits = (
                x_t_in * (-raw_logits)
                + (1.0 - x_t_in) * raw_logits
            )
            if self.focal >= 0:
                flip_target = torch.logical_xor(
                    x_0.bool(), x_t_in.bool()
                ).float()
                per_bit_base = focal_loss(
                    raw_logits, flip_target, gamma=self.focal
                )
            else:
                per_bit_base = F.binary_cross_entropy_with_logits(
                    clean_logits, x_0, reduction="none"
                )
        else:
            clean_logits = raw_logits
            if self.focal >= 0:
                per_bit_base = focal_loss(
                    clean_logits,
                    x_0,
                    alpha=self.focal,
                    gamma=self.focal,
                )
            else:
                per_bit_base = F.binary_cross_entropy_with_logits(
                    clean_logits, x_0, reduction="none"
                )

        if torch.isinf(per_bit_base).any() or torch.isnan(per_bit_base).any():
            pdb.set_trace()

        # Align weighted convention with released BLD.
        # With the user's current --loss_final mean this is simply 1.0.
        if self.loss_final == "weighted":
            base_weight = (
                1.0
                - ((t - 1).float() / self.num_timesteps)
            ).view(-1, 1, 1)
        elif self.loss_final == "mean":
            base_weight = 1.0
        else:
            raise NotImplementedError(
                f"Unknown loss_final={self.loss_final!r}"
            )

        base_loss = (base_weight * per_bit_base).mean()

        with torch.no_grad():
            acc = (
                ((clean_logits > 0.0).float() == x_0)
                .float()
                .mean()
            )

        return {
            "x_0": x_0,
            "t": t,
            "x_t_in": x_t_in,
            "raw_logits": raw_logits,
            "clean_logits": clean_logits,
            "per_bit_base": per_bit_base,
            "base_loss": base_loss,
            "base_loss_unweighted": per_bit_base.mean(),
            "acc": acc,
        }

    def _train_loss(self, x_0, label=None, x_ct=None):
        st = self._prepare_train_state(x_0, label=label, x_ct=x_ct)
        return {
            "loss": st["base_loss"],
            "bce_loss": st["base_loss_unweighted"],
            "acc": st["acc"],
        }

    @torch.no_grad()
    def sample(
        self,
        temp=1.0,
        sample_steps=None,
        b=8,
        shape=None,
        return_all=False,
        label=None,
        mask=None,
        guidance=None,
        full=False,
    ):
        """
        EXACT current corrected-BFM sampler except denoiser time_steps=t-1.
        Posterior construction, grid, Bernoulli draws and hard final are unchanged.
        """
        del full
        device = next(self._denoise_fn.parameters()).device

        if shape is not None:
            x_t = torch.bernoulli(
                0.5 * torch.ones(shape, device=device, dtype=torch.float32)
            )
            b = shape[0]
        else:
            x_t = torch.bernoulli(
                0.5 * torch.ones(
                    (b, np.prod(self.shape), self.codebook_size),
                    device=device,
                    dtype=torch.float32,
                )
            )

        if mask is not None:
            mask_tensor = mask["mask"].unsqueeze(0).to(device)
            latent = mask["latent"].unsqueeze(0).to(device)
            x_t = latent * mask_tensor + x_t * (1.0 - mask_tensor)

        if sample_steps is None:
            sample_steps = self.num_timesteps
        sample_steps = int(sample_steps)
        if sample_steps < 1 or sample_steps > self.num_timesteps:
            raise ValueError(
                f"sample_steps must be in [1,{self.num_timesteps}], got {sample_steps}"
            )

        sampling_steps = np.arange(1, self.num_timesteps + 1)
        if sample_steps != self.num_timesteps:
            idx = np.linspace(
                0.0, self.num_timesteps - 1, sample_steps
            ).astype(np.int64)
            sampling_steps = sampling_steps[idx]
        sampling_steps = sampling_steps[::-1]

        if return_all:
            x_all = [x_t]

        if self.dataset == "imagenet":
            if label is None:
                label = (torch.arange(b, device=device) * 100).long()
            else:
                label = torch.full(
                    (b,), label, device=device, dtype=torch.long
                )

        for i, step_value in enumerate(sampling_steps):
            t = torch.full(
                (b,), int(step_value), device=device, dtype=torch.long
            )
            model_t = self._model_time(t)

            if (
                self.dataset.startswith("imagenet")
                or self.dataset.startswith("laion")
                or self.dataset.startswith("ising")
            ):
                raw_logits = self._denoise_fn(
                    x_t, time_steps=model_t, label=label
                )
                raw_logits = raw_logits / temp
                if guidance is not None:
                    raw_logits_uncond = self._denoise_fn(
                        x_t, time_steps=model_t, y=None
                    )
                    raw_logits_uncond = raw_logits_uncond / temp
                    raw_logits = (
                        (1.0 + guidance) * raw_logits
                        - guidance * raw_logits_uncond
                    )
            else:
                raw_logits = self._denoise_fn(
                    x_t, time_steps=model_t
                )
                raw_logits = raw_logits / temp

            if self.p_flip:
                clean_logits = (
                    x_t * (-raw_logits)
                    + (1.0 - x_t) * raw_logits
                )
            else:
                clean_logits = raw_logits
            clean_prob = torch.sigmoid(clean_logits)

            if int(step_value) != 1:
                next_step_value = int(sampling_steps[i + 1])
                t_target = torch.full(
                    (b,),
                    next_step_value,
                    device=device,
                    dtype=torch.long,
                )
                x_target_prob = self._reverse_probability(
                    clean_prob=clean_prob,
                    x_t=x_t,
                    t_current=t,
                    t_target=t_target,
                )
                x_next = torch.bernoulli(x_target_prob)
            else:
                # Keep the current controlled hard projection.
                if self.hard_final:
                    x_next = (clean_prob > 0.5).float()
                else:
                    x_next = torch.bernoulli(clean_prob)

            x_t = x_next

            if mask is not None:
                x_t = (
                    latent * mask_tensor
                    + x_t * (1.0 - mask_tensor)
                )

            if return_all:
                x_all.append(x_t)

        if return_all:
            return torch.cat(x_all, dim=0)
        return x_t


class BinaryDiffusionFlowOneStepLossAblation(BinaryDiffusionFlowTimeAligned):
    """
    Same exact aligned sampler as BinaryDiffusionFlowTimeAligned.

    Only training loss changes.

    SRC_MODE=unweighted:
        L = L_base + lambda * Brier

    SRC_MODE=sensitivity:
        L = L_base + lambda * [S(t-1,t)^2 / E(S^2)] * Brier

    Same t, same X_t, same forward, same m_theta are used for all terms.
    No second model forward and no altered timestep sampling.
    """

    def __init__(self, H, denoise_fn, mask_id):
        super().__init__(H, denoise_fn, mask_id)

        if float(self.aux) != 0.0:
            raise ValueError(
                "Controlled SRC ablation requires --aux 0. "
                "Do not mix the old posterior-CE auxiliary into this experiment."
            )

        self.src_mode = os.environ.get(
            "BFM_SRC_MODE", "sensitivity"
        ).strip().lower()
        if self.src_mode not in {"unweighted", "sensitivity"}:
            raise ValueError(
                "BFM_SRC_MODE must be 'unweighted' or 'sensitivity'."
            )

        self.src_lambda = _env_float("BFM_SRC_LAMBDA", 1.0)

        # Exact one-step S(t-1,t)^2 table for physical t=1..T.
        # t=1 -> s=0 is excluded because the current sampler hard-thresholds
        # there, while our SRC risk identity is Bernoulli-probabilistic.
        s2 = torch.zeros(self.num_timesteps + 1, dtype=torch.float32)
        for t in range(2, self.num_timesteps + 1):
            s = t - 1
            rho_s = float(self.interpolation_t[s].item())
            rho_t = float(self.interpolation_t[t].item())
            S = (
                (rho_s * rho_s - rho_t * rho_t)
                / (rho_s * (1.0 - rho_t * rho_t))
            )
            s2[t] = float(S * S)

        # Training samples t uniformly from 1..T. t=1 has zero SRC weight.
        mean_s2 = float(s2[1:].mean().item())
        if mean_s2 <= 0.0:
            raise RuntimeError("Invalid mean S^2.")
        self.src_mean_s2 = mean_s2
        self.src_scale = 1.0 / mean_s2
        self.register_buffer(
            "src_s2_table", s2, persistent=False
        )

    def _train_loss(self, x_0, label=None, x_ct=None):
        st = self._prepare_train_state(x_0, label=label, x_ct=x_ct)

        # IMPORTANT: no second forward pass.
        clean_prob = torch.sigmoid(st["clean_logits"])

        # Per-example Brier over all latent bits.
        per_example_brier = (
            (clean_prob.float() - st["x_0"].float())
            .square()
            .flatten(1)
            .mean(dim=1)
        )

        if self.src_mode == "unweighted":
            # Control for "maybe any extra Brier regularizer helps".
            weight = torch.ones_like(per_example_brier)
            # Exclude t=1 here too so Brier-control and SRC have exactly the
            # same support; only relative sensitivity weighting differs.
            weight = weight * (st["t"] > 1).float()
        else:
            raw_s2 = self.src_s2_table[st["t"]].to(
                per_example_brier.device
            )
            weight = raw_s2 * float(self.src_scale)

        aux_brier = (weight * per_example_brier).mean()
        total_loss = st["base_loss"] + self.src_lambda * aux_brier

        with torch.no_grad():
            raw_s2_for_log = self.src_s2_table[st["t"]].to(
                per_example_brier.device
            )

        return {
            "loss": total_loss,
            "bce_loss": st["base_loss_unweighted"],
            "acc": st["acc"],
            "src_aux_loss": aux_brier,
            "src_plain_brier": per_example_brier.mean(),
            "src_weight_mean": weight.mean(),
            "src_raw_s2_mean": raw_s2_for_log.mean(),
        }


# ============================================================================
# V4: Multi-deployment SRC using the ORIGINAL Brier/MSE surrogate
# ============================================================================

class BinaryDiffusionFlowMultiNFESRC(BinaryDiffusionFlowTimeAligned):
    """
    V4: multi-deployment sampling-risk regularization.

    IMPORTANT: this keeps the exact aligned BFM base objective and the exact
    Brier/MSE surrogate derived in the SRC theory.

        L_v4 = L_base + lambda * W_t * (m_theta(X_t,t) - X0)^2

    W_t aggregates the analytically known S_{s,t}^2 over the *actual* sampler
    grids for the target NFE set (default 64,32,16,8).  The network remains
    m_theta(X_t,t): s is NOT an input and there is no second forward pass.

    For each deployment K we first normalize S^2 by that deployment's mean
    S^2, so 64/32/16/8 receive equal auxiliary budget by default rather than
    letting aggressive low-NFE grids dominate solely because their raw S is
    larger.  The resulting W_t is scaled so E_{t~Uniform{1..T}}[W_t] = 1.

    The final hard 1->0 projection is excluded from SRC because the exact
    Bernoulli reverse-risk identity applies to probabilistic reverse intervals.
    The original base objective still trains t=1 normally.
    """

    def __init__(self, H, denoise_fn, mask_id):
        super().__init__(H, denoise_fn, mask_id)

        if float(self.aux) != 0.0:
            raise ValueError(
                "V4 controlled SRC requires --aux 0. Do not mix the historical "
                "posterior auxiliary into this experiment."
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
                raise ValueError("BFM_V4_NFE_WEIGHTS must be non-negative and sum > 0.")
            pi = np.asarray(pi, dtype=np.float64)
            pi = pi / pi.sum()
        else:
            pi = np.full(len(self.v4_nfes), 1.0 / len(self.v4_nfes), dtype=np.float64)
        self.v4_nfe_weights = tuple(float(x) for x in pi.tolist())

        weight_table = torch.zeros(self.num_timesteps + 1, dtype=torch.float64)
        raw_s2_by_nfe = {}
        mean_s2_by_nfe = {}
        grids_by_nfe = {}

        for k, pi_k in zip(self.v4_nfes, self.v4_nfe_weights):
            grid = self._sampling_grid(k)
            intervals = [(int(grid[i]), int(grid[i + 1])) for i in range(len(grid) - 1)]
            if not intervals:
                raise RuntimeError(f"NFE={k} produced no probabilistic intervals.")

            s2_values = []
            for t_current, t_target in intervals:
                s2_values.append(self._sensitivity_squared(t_target, t_current))

            mean_s2 = float(np.mean(s2_values))
            if (not np.isfinite(mean_s2)) or mean_s2 <= 0.0:
                raise RuntimeError(f"Invalid mean S^2 for NFE={k}: {mean_s2}")

            raw_s2_by_nfe[k] = tuple(float(x) for x in s2_values)
            mean_s2_by_nfe[k] = mean_s2
            grids_by_nfe[k] = tuple(int(x) for x in grid.tolist())

            # Define normalized deployment risk:
            #   R_K = (1/M_K) sum_[t->s in I_K] (S^2 / mean_K[S^2]) * error_t.
            # Since training samples physical t uniformly from 1..T, multiply by
            # T/M_K so E_t[W_t * error_t] equals the desired deployment mixture.
            m_k = float(len(intervals))
            for (t_current, _), raw_s2 in zip(intervals, s2_values):
                normalized_s2 = raw_s2 / mean_s2
                weight_table[t_current] += (
                    self.num_timesteps * float(pi_k) / m_k * normalized_s2
                )

        # t=1 is the hard final source and should have zero SRC weight.
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

    def _sampling_grid(self, sample_steps: int) -> np.ndarray:
        """Mirror BinaryDiffusionFlowTimeAligned.sample() exactly."""
        sampling_steps = np.arange(1, self.num_timesteps + 1)
        if int(sample_steps) != self.num_timesteps:
            idx = np.linspace(
                0.0, self.num_timesteps - 1, int(sample_steps)
            ).astype(np.int64)
            sampling_steps = sampling_steps[idx]
        return sampling_steps[::-1]

    def _sensitivity_squared(self, s: int, t: int) -> float:
        """Exact S_{s,t}^2 for 1 <= s < t <= T on the current Bernoulli path."""
        if not (1 <= int(s) < int(t) <= self.num_timesteps):
            raise ValueError(f"Need 1 <= s < t <= T, got s={s}, t={t}.")
        rho_s = float(self.interpolation_t[int(s)].item())
        rho_t = float(self.interpolation_t[int(t)].item())
        denom = rho_s * (1.0 - rho_t * rho_t)
        if abs(denom) <= 1e-15:
            raise RuntimeError(f"Degenerate sensitivity denominator for s={s}, t={t}.")
        sensitivity = (
            (rho_s * rho_s - rho_t * rho_t) / denom
        )
        return float(sensitivity * sensitivity)

    def _train_loss(self, x_0, label=None, x_ct=None):
        st = self._prepare_train_state(x_0, label=label, x_ct=x_ct)

        # The original BFM base objective is preserved exactly.
        base_loss = st["base_loss"]

        # Same network output, same t, same X_t, no second forward pass.
        # For Bernoulli clean-posterior prediction this is the Brier/MSE surrogate
        # from the exact SRC excess-risk derivation.
        clean_prob = torch.sigmoid(st["clean_logits"])
        per_example_brier = (
            (clean_prob.float() - st["x_0"].float())
            .square()
            .flatten(1)
            .mean(dim=1)
        )

        weight = self.v4_weight_table[st["t"]].to(per_example_brier.device)
        multi_nfe_src = (weight * per_example_brier).mean()
        total_loss = base_loss + self.v4_lambda * multi_nfe_src

        with torch.no_grad():
            active = (weight > 0).float()

        return {
            "loss": total_loss,
            "bce_loss": st["base_loss_unweighted"],
            "bfm_base_loss": base_loss,
            "acc": st["acc"],
            "v4_src_loss": multi_nfe_src,
            "v4_plain_brier": per_example_brier.mean(),
            "v4_weight_mean_batch": weight.mean(),
            "v4_weight_max_batch": weight.max(),
            "v4_active_fraction": active.mean(),
        }
