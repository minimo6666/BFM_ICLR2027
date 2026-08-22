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
