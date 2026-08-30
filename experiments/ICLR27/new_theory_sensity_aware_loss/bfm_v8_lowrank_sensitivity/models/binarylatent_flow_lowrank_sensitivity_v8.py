"""V8: Baseline-Preserving Low-Rank Sensitivity Residual BFM.

This method is intentionally built from the V4 sensitivity construction while
removing the mechanism that repeatedly hurt 64/32 NFE:

  V4: sensitivity auxiliary -> shared BFM parameters
  V8: base BFM loss         -> shared BFM parameters ONLY
      sensitivity auxiliary -> tiny low-rank residual adapter ONLY

The adapter is interval-aware through (t,s,S) but the analytic BFM reverse
posterior remains unchanged.  At inference, the raw denoiser logits are

    logits_final = logits_base + delta_logits_lowrank(t,s,S)

and the corrected expectation-consistent BFM posterior is then applied exactly
as before.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from models.binarylatent_flow_controlled_src_v4 import BinaryDiffusionFlowTimeAligned


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


class BinaryDiffusionFlowLowRankSensitivityV8(BinaryDiffusionFlowTimeAligned):
    def __init__(self, H, denoise_fn, mask_id):
        super().__init__(H, denoise_fn, mask_id)
        if float(self.aux) != 0.0:
            raise ValueError("V8 requires --aux 0 for a controlled experiment.")
        if not hasattr(self._denoise_fn, "adapter_residual"):
            raise TypeError("V8 requires TransformerBDLowRankSensitivityV8 as denoiser.")

        self.lr_lambda = _env_float("BFM_LR_LAMBDA", 1.0)
        if self.lr_lambda < 0.0:
            raise ValueError("BFM_LR_LAMBDA must be non-negative.")

        nfe_text = os.environ.get("BFM_LR_NFES", "64,32,16,8")
        nfes = [int(x.strip()) for x in nfe_text.split(",") if x.strip()]
        if not nfes:
            raise ValueError("BFM_LR_NFES must not be empty.")
        if len(set(nfes)) != len(nfes):
            raise ValueError("BFM_LR_NFES must not contain duplicates.")
        for k in nfes:
            if k < 2 or k > self.num_timesteps:
                raise ValueError(f"Each NFE must be in [2,{self.num_timesteps}], got {k}.")
        self.lr_nfes = tuple(nfes)

        pi_text = os.environ.get("BFM_LR_NFE_WEIGHTS", "")
        if pi_text.strip():
            pi = np.asarray([float(x.strip()) for x in pi_text.split(",") if x.strip()], dtype=np.float64)
            if len(pi) != len(self.lr_nfes) or np.any(pi < 0.0) or pi.sum() <= 0.0:
                raise ValueError("Invalid BFM_LR_NFE_WEIGHTS.")
            pi = pi / pi.sum()
        else:
            pi = np.full(len(self.lr_nfes), 1.0 / len(self.lr_nfes), dtype=np.float64)
        self.lr_nfe_weights = tuple(float(x) for x in pi.tolist())

        # Build EXACT V4 deployment interval banks and their per-NFE mean S^2.
        self.lr_intervals_by_nfe: Dict[int, Tuple[Tuple[int, int], ...]] = {}
        self.lr_mean_s2_by_nfe: Dict[int, float] = {}
        self.lr_grids_by_nfe: Dict[int, Tuple[int, ...]] = {}
        for k in self.lr_nfes:
            grid = self._sampling_grid(k)
            intervals = tuple((int(grid[i]), int(grid[i + 1])) for i in range(len(grid) - 1))
            if not intervals:
                raise RuntimeError(f"NFE={k} produced no probabilistic intervals.")
            s2 = [self._sensitivity_squared(s=s, t=t) for t, s in intervals]
            mu = float(np.mean(s2))
            if (not np.isfinite(mu)) or mu <= 0.0:
                raise RuntimeError(f"Invalid mean S^2 for NFE={k}: {mu}")
            self.lr_intervals_by_nfe[k] = intervals
            self.lr_mean_s2_by_nfe[k] = mu
            self.lr_grids_by_nfe[k] = tuple(int(v) for v in grid.tolist())

        # CDF for per-example deployment sampling; no manual mini-batch partition.
        cdf = np.cumsum(np.asarray(self.lr_nfe_weights, dtype=np.float64))
        cdf[-1] = 1.0
        self.register_buffer("lr_nfe_cdf", torch.tensor(cdf, dtype=torch.float32), persistent=False)

        # Keep all adapter-branch randomness on an independent generator so the
        # original BFM branch sees the same global RNG stream it would have seen
        # without V8.  This makes the base path as close as possible to a matched
        # baseline training trajectory.
        self._adapter_seed = _env_int("BFM_LR_ADAPTER_SEED", _env_int("EXPERIMENT_SEED", 20260821) + 1000003)
        self._adapter_generators = {}

    def _sampling_grid(self, sample_steps: int) -> np.ndarray:
        sampling_steps = np.arange(1, self.num_timesteps + 1)
        if int(sample_steps) != self.num_timesteps:
            idx = np.linspace(0.0, self.num_timesteps - 1, int(sample_steps)).astype(np.int64)
            sampling_steps = sampling_steps[idx]
        return sampling_steps[::-1]

    def _sensitivity(self, s: int, t: int) -> float:
        if not (1 <= int(s) < int(t) <= self.num_timesteps):
            raise ValueError(f"Need 1 <= s < t <= T, got s={s}, t={t}.")
        rho_s = float(self.interpolation_t[int(s)].item())
        rho_t = float(self.interpolation_t[int(t)].item())
        denom = rho_s * (1.0 - rho_t * rho_t)
        if abs(denom) <= 1e-15:
            raise RuntimeError(f"Degenerate sensitivity denominator for s={s}, t={t}.")
        return float((rho_s * rho_s - rho_t * rho_t) / denom)

    def _sensitivity_squared(self, s: int, t: int) -> float:
        S = self._sensitivity(s=s, t=t)
        return float(S * S)

    def _adapter_generator(self, device: torch.device) -> torch.Generator:
        key = str(device)
        if key not in self._adapter_generators:
            rank = torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else 0
            gen = torch.Generator(device=device)
            gen.manual_seed(int(self._adapter_seed) + 7919 * int(rank))
            self._adapter_generators[key] = gen
        return self._adapter_generators[key]

    def _sample_adapter_pairs(self, b: int, device: torch.device):
        # Sample K independently per example.  Equal weights by default, exactly as V4.
        gen = self._adapter_generator(device)
        u = torch.rand(b, device=device, generator=gen)
        k_index = torch.bucketize(u, self.lr_nfe_cdf.to(device), right=False)
        k_index = torch.clamp(k_index, max=len(self.lr_nfes) - 1)

        t = torch.empty(b, dtype=torch.long, device=device)
        s = torch.empty(b, dtype=torch.long, device=device)
        raw_s2 = torch.empty(b, dtype=torch.float32, device=device)
        norm_s2 = torch.empty(b, dtype=torch.float32, device=device)
        nfe_tensor = torch.empty(b, dtype=torch.long, device=device)

        for i, k in enumerate(self.lr_nfes):
            mask = (k_index == i)
            n = int(mask.sum().item())
            if n == 0:
                continue
            intervals = self.lr_intervals_by_nfe[k]
            pick = torch.randint(0, len(intervals), (n,), device=device, generator=gen)
            t_vals = torch.tensor([intervals[int(j)][0] for j in pick.tolist()], device=device, dtype=torch.long)
            s_vals = torch.tensor([intervals[int(j)][1] for j in pick.tolist()], device=device, dtype=torch.long)
            s2_vals = torch.tensor(
                [self._sensitivity_squared(int(ss), int(tt)) for tt, ss in zip(t_vals.tolist(), s_vals.tolist())],
                device=device,
                dtype=torch.float32,
            )
            t[mask] = t_vals
            s[mask] = s_vals
            raw_s2[mask] = s2_vals
            norm_s2[mask] = s2_vals / float(self.lr_mean_s2_by_nfe[k])
            nfe_tensor[mask] = int(k)

        sensitivity_norm = torch.sqrt(torch.clamp(norm_s2, min=0.0))
        return t, s, raw_s2, norm_s2, sensitivity_norm, nfe_tensor

    def _raw_to_clean_logits(self, raw_logits: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        if self.p_flip:
            return x_t * (-raw_logits) + (1.0 - x_t) * raw_logits
        return raw_logits

    def _adapter_branch(self, x_0: torch.Tensor, label=None):
        b, device = x_0.size(0), x_0.device
        t, s, raw_s2, norm_s2, sensitivity_norm, nfe_tensor = self._sample_adapter_pairs(b, device)

        x_t_prob = self.q_sample(x_0.float(), t)
        gen = self._adapter_generator(device)
        x_t = torch.bernoulli(x_t_prob, generator=gen)
        model_t = self._model_time(t)

        # Exact gradient isolation.  Evaluate the base feature extractor in eval
        # mode so this extra branch does not consume dropout RNG and therefore
        # does not perturb the stochastic training stream of the base branch.
        denoiser_was_training = self._denoise_fn.training
        self._denoise_fn.eval()
        try:
            with torch.no_grad():
                if label is not None:
                    base_raw, hidden = self._denoise_fn.forward_base_with_hidden(
                        x_t, label=label, time_steps=model_t
                    )
                else:
                    base_raw, hidden = self._denoise_fn.forward_base_with_hidden(
                        x_t, time_steps=model_t
                    )
        finally:
            if denoiser_was_training:
                self._denoise_fn.train()

        residual_raw = self._denoise_fn.adapter_residual(
            hidden=hidden.detach(),
            t_current=t,
            t_target=s,
            sensitivity_norm=sensitivity_norm,
        )
        corrected_raw = base_raw.detach() + residual_raw

        base_clean_logits = self._raw_to_clean_logits(base_raw.detach(), x_t)
        corrected_clean_logits = self._raw_to_clean_logits(corrected_raw, x_t)
        base_prob = torch.sigmoid(base_clean_logits)
        corrected_prob = torch.sigmoid(corrected_clean_logits)

        per_example_base_brier = (
            (base_prob.float() - x_0.float()).square().flatten(1).mean(dim=1)
        )
        per_example_corr_brier = (
            (corrected_prob.float() - x_0.float()).square().flatten(1).mean(dim=1)
        )

        # V4's per-deployment normalized sensitivity risk, but now it trains ONLY
        # the extra low-rank correction capacity.  It cannot reallocate base BFM
        # parameters away from high-NFE behavior.
        correction_loss = (norm_s2 * per_example_corr_brier).mean()

        return {
            "correction_loss": correction_loss,
            "adapter_plain_brier": per_example_corr_brier.mean(),
            "adapter_base_brier": per_example_base_brier.mean(),
            "adapter_brier_delta": (per_example_corr_brier - per_example_base_brier).mean(),
            "adapter_norm_s2_mean": norm_s2.mean(),
            "adapter_raw_s2_mean": raw_s2.mean(),
            "adapter_residual_abs_mean": residual_raw.detach().abs().mean(),
            "adapter_residual_abs_max": residual_raw.detach().abs().max(),
            "adapter_nfe_mean": nfe_tensor.float().mean(),
        }

    def _train_loss(self, x_0, label=None, x_ct=None):
        if x_ct is not None:
            raise NotImplementedError("x_ct-conditioned training is not implemented in V8.")

        # Base branch: EXACT aligned BFM objective, adapter is not used.
        st = self._prepare_train_state(x_0, label=label, x_ct=None)

        # Adapter branch: separate deployment interval, no gradients to base model.
        corr = self._adapter_branch(st["x_0"], label=label)
        total_loss = st["base_loss"] + self.lr_lambda * corr["correction_loss"]

        return {
            "loss": total_loss,
            "bce_loss": st["base_loss_unweighted"],
            "bfm_base_loss": st["base_loss"],
            "acc": st["acc"],
            "lr_corr_loss": corr["correction_loss"],
            "lr_plain_brier": corr["adapter_plain_brier"],
            "lr_base_brier_same_pairs": corr["adapter_base_brier"],
            "lr_brier_delta": corr["adapter_brier_delta"],
            "lr_norm_s2_mean": corr["adapter_norm_s2_mean"],
            "lr_raw_s2_mean": corr["adapter_raw_s2_mean"],
            "lr_residual_abs_mean": corr["adapter_residual_abs_mean"],
            "lr_residual_abs_max": corr["adapter_residual_abs_max"],
        }

    def base_parameters_for_clip(self):
        adapter_ids = {id(p) for p in self._denoise_fn.adapter_parameters()}
        return [p for p in self.parameters() if p.requires_grad and id(p) not in adapter_ids]

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
        """Same corrected analytic BFM sampler, with a low-rank residual per interval."""
        del full
        device = next(self._denoise_fn.parameters()).device

        if shape is not None:
            x_t = torch.bernoulli(0.5 * torch.ones(shape, device=device, dtype=torch.float32))
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
            raise ValueError(f"sample_steps must be in [1,{self.num_timesteps}], got {sample_steps}")

        sampling_steps = self._sampling_grid(sample_steps)
        if return_all:
            x_all = [x_t]

        if self.dataset == "imagenet":
            if label is None:
                label = (torch.arange(b, device=device) * 100).long()
            else:
                label = torch.full((b,), label, device=device, dtype=torch.long)

        # For trained deployment NFEs use the matching V4 normalization.  For an
        # unseen NFE, use that grid's own analytic mean S^2; no learned lookup is needed.
        inference_intervals = [
            (int(sampling_steps[i]), int(sampling_steps[i + 1]))
            for i in range(len(sampling_steps) - 1)
        ]
        if inference_intervals:
            inference_mean_s2 = float(np.mean([
                self._sensitivity_squared(s=s, t=t) for t, s in inference_intervals
            ]))
        else:
            inference_mean_s2 = 1.0

        for i, step_value in enumerate(sampling_steps):
            t = torch.full((b,), int(step_value), device=device, dtype=torch.long)
            model_t = self._model_time(t)

            # Base prediction / hidden representation are exactly original BFM.
            if self.dataset.startswith("imagenet") or self.dataset.startswith("laion") or self.dataset.startswith("ising"):
                base_raw, hidden = self._denoise_fn.forward_base_with_hidden(
                    x_t, label=label, time_steps=model_t
                )
            else:
                base_raw, hidden = self._denoise_fn.forward_base_with_hidden(
                    x_t, time_steps=model_t
                )

            # Final 1->0 hard projection is outside the probabilistic sensitivity theorem:
            # use the untouched base predictor there.
            if int(step_value) != 1:
                next_step_value = int(sampling_steps[i + 1])
                s = torch.full((b,), next_step_value, device=device, dtype=torch.long)
                raw_s2_scalar = self._sensitivity_squared(s=next_step_value, t=int(step_value))
                sens_norm_scalar = np.sqrt(raw_s2_scalar / max(inference_mean_s2, 1e-12))
                sens_norm = torch.full(
                    (b,), float(sens_norm_scalar), device=device, dtype=torch.float32
                )
                residual_raw = self._denoise_fn.adapter_residual(
                    hidden=hidden,
                    t_current=t,
                    t_target=s,
                    sensitivity_norm=sens_norm,
                )
                raw_logits = base_raw + residual_raw
            else:
                raw_logits = base_raw

            raw_logits = raw_logits / temp
            if guidance is not None:
                raise NotImplementedError(
                    "V8 guidance path is intentionally not implemented in the first LSUN control."
                )

            clean_logits = self._raw_to_clean_logits(raw_logits, x_t)
            clean_prob = torch.sigmoid(clean_logits)

            if int(step_value) != 1:
                t_target = torch.full(
                    (b,), int(sampling_steps[i + 1]), device=device, dtype=torch.long
                )
                x_target_prob = self._reverse_probability(
                    clean_prob=clean_prob,
                    x_t=x_t,
                    t_current=t,
                    t_target=t_target,
                )
                x_next = torch.bernoulli(x_target_prob)
            else:
                if self.hard_final:
                    x_next = (clean_prob > 0.5).float()
                else:
                    x_next = torch.bernoulli(clean_prob)

            x_t = x_next
            if mask is not None:
                x_t = latent * mask_tensor + x_t * (1.0 - mask_tensor)
            if return_all:
                x_all.append(x_t)

        if return_all:
            return torch.cat(x_all, dim=0)
        return x_t
