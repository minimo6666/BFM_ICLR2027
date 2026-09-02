"""V9: boundary-consistent terminal repair on top of the frozen V8 model.

This is deliberately a narrow causal experiment.  It does not retrain the V8
backbone or sensitivity adapter and it does not modify the analytic BFM reverse
posterior.  It trains only a small terminal carry head to repair the empirically
localized ``2->1->0`` / ``3->1->0`` incompatibility.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from models.binarylatent_flow_lowrank_sensitivity_v8 import (
    BinaryDiffusionFlowLowRankSensitivityV8,
)


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


class BinaryDiffusionFlowBoundaryRepairV9(BinaryDiffusionFlowLowRankSensitivityV8):
    def __init__(self, H, denoise_fn, mask_id):
        super().__init__(H, denoise_fn, mask_id)
        if not hasattr(self._denoise_fn, "terminal_carry_residual"):
            raise TypeError("V9 requires TransformerBDBoundaryRepairV9 as denoiser")

        nfe_text = os.environ.get("BFM_V9_BOUNDARY_NFES", "64,32")
        boundary_nfes = tuple(int(item.strip()) for item in nfe_text.split(",") if item.strip())
        if not boundary_nfes or len(set(boundary_nfes)) != len(boundary_nfes):
            raise ValueError("BFM_V9_BOUNDARY_NFES must contain unique NFEs")
        for nfe in boundary_nfes:
            if nfe not in self.lr_grids_by_nfe:
                raise ValueError(
                    f"Boundary NFE {nfe} is absent from BFM_LR_NFES={self.lr_nfes}"
                )
            grid = self.lr_grids_by_nfe[nfe]
            if len(grid) < 2 or int(grid[-1]) != 1:
                raise RuntimeError(f"Unexpected sampling grid for NFE={nfe}: {grid}")

        self.boundary_nfes = boundary_nfes
        self.boundary_branches = _env_int("BFM_V9_BRANCHES", 4)
        if self.boundary_branches < 2:
            raise ValueError("BFM_V9_BRANCHES must be at least 2")
        self.boundary_hard_tau = _env_float("BFM_V9_HARD_TAU", 0.5)
        if self.boundary_hard_tau <= 0.0:
            raise ValueError("BFM_V9_HARD_TAU must be positive")
        self.boundary_lambda_bce = _env_float("BFM_V9_LAMBDA_BCE", 1.0)
        self.boundary_lambda_prob = _env_float("BFM_V9_LAMBDA_PROB", 1.0)
        self.boundary_lambda_hard = _env_float("BFM_V9_LAMBDA_HARD", 1.0)
        # Disabled in the first causal run: a non-zero value requires an extra
        # frozen 171M-parameter terminal forward.  The head is already
        # zero-initialized and the entire V8 model is frozen.
        self.boundary_lambda_anchor = _env_float("BFM_V9_LAMBDA_ANCHOR", 0.0)
        self.boundary_terminal_chunk = _env_int("BFM_V9_TERMINAL_CHUNK", 32)
        if self.boundary_terminal_chunk < 1:
            raise ValueError("BFM_V9_TERMINAL_CHUNK must be positive")

    def boundary_parameters(self):
        return list(self._denoise_fn.boundary_parameters())

    def freeze_v8_for_boundary_training(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self._denoise_fn.boundary_parameters():
            parameter.requires_grad_(True)
        # Frozen modules must not consume dropout RNG or update state.
        self.eval()
        self._denoise_fn.terminal_carry.train()

    def _boundary_metadata(
        self,
        batch: int,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ):
        choices = torch.randint(
            0,
            len(self.boundary_nfes),
            (batch,),
            device=device,
            generator=generator,
        )
        source_t = torch.empty(batch, dtype=torch.long, device=device)
        sensitivity_norm = torch.empty(batch, dtype=torch.float32, device=device)
        nfe_tensor = torch.empty(batch, dtype=torch.long, device=device)
        for index, nfe in enumerate(self.boundary_nfes):
            mask = choices == index
            if not bool(mask.any()):
                continue
            grid = self.lr_grids_by_nfe[nfe]
            preterminal = int(grid[-2])
            raw_s2 = self._sensitivity_squared(s=1, t=preterminal)
            normalized = np.sqrt(raw_s2 / max(self.lr_mean_s2_by_nfe[nfe], 1.0e-12))
            source_t[mask] = preterminal
            sensitivity_norm[mask] = float(normalized)
            nfe_tensor[mask] = int(nfe)
        return source_t, sensitivity_norm, nfe_tensor

    @staticmethod
    def _branch_sample(
        probability: torch.Tensor,
        branches: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Antithetic Bernoulli branches with exact per-branch marginals."""
        shape = (probability.shape[0], branches, *probability.shape[1:])
        half = branches // 2
        if half > 0:
            u = torch.rand(
                (probability.shape[0], half, *probability.shape[1:]),
                device=probability.device,
                generator=generator,
            )
            uniforms = torch.cat([u, 1.0 - u], dim=1)
        else:
            uniforms = probability.new_empty((probability.shape[0], 0, *probability.shape[1:]))
        if uniforms.shape[1] < branches:
            extra = torch.rand(
                (probability.shape[0], 1, *probability.shape[1:]),
                device=probability.device,
                generator=generator,
            )
            uniforms = torch.cat([uniforms, extra], dim=1)
        return (uniforms < probability.unsqueeze(1)).to(probability.dtype).reshape(shape)

    @torch.no_grad()
    def _source_prediction(
        self,
        x_t: torch.Tensor,
        source_t: torch.Tensor,
        sensitivity_norm: torch.Tensor,
        adapter_scale: float = 1.0,
    ):
        base_raw, hidden = self._denoise_fn.forward_base_with_hidden(
            x_t, time_steps=self._model_time(source_t)
        )
        target_t = torch.ones_like(source_t)
        residual = self._denoise_fn.adapter_residual(
            hidden=hidden,
            t_current=source_t,
            t_target=target_t,
            sensitivity_norm=sensitivity_norm,
        )
        raw = base_raw + float(adapter_scale) * residual
        clean_logits = self._raw_to_clean_logits(raw, x_t)
        clean_probability = torch.sigmoid(clean_logits)
        x1_probability = self._reverse_probability(
            clean_prob=clean_probability,
            x_t=x_t,
            t_current=source_t,
            t_target=target_t,
        )
        return clean_probability, x1_probability

    def _terminal_logits(
        self,
        *,
        x1: torch.Tensor,
        source_clean_prob: torch.Tensor,
        source_t: torch.Tensor,
        sensitivity_norm: torch.Tensor,
        adapter_scale: torch.Tensor,
        enable_repair: bool,
    ) -> torch.Tensor:
        outputs = []
        for start in range(0, x1.shape[0], self.boundary_terminal_chunk):
            stop = min(start + self.boundary_terminal_chunk, x1.shape[0])
            x1_chunk = x1[start:stop]
            with torch.no_grad():
                terminal_t = torch.ones(stop - start, dtype=torch.long, device=x1.device)
                base_raw, hidden = self._denoise_fn.forward_base_with_hidden(
                    x1_chunk, time_steps=self._model_time(terminal_t)
                )
                base_clean_logits = self._raw_to_clean_logits(base_raw, x1_chunk)
            if enable_repair:
                residual = self._denoise_fn.terminal_carry_residual(
                    terminal_hidden=hidden.detach(),
                    source_clean_prob=source_clean_prob[start:stop].detach(),
                    source_t=source_t[start:stop],
                    sensitivity_norm=sensitivity_norm[start:stop],
                    adapter_scale=adapter_scale[start:stop],
                )
                outputs.append(base_clean_logits.detach() + residual)
            else:
                outputs.append(base_clean_logits.detach())
        return torch.cat(outputs, dim=0)

    def boundary_objective(
        self,
        x_0: torch.Tensor,
        *,
        branches: Optional[int] = None,
        enable_repair: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        """Train/evaluate the isolated final probabilistic boundary.

        Two branch distributions are intentionally different:

        * dynamic branches use the learned marginalized reverse kernel and are
          used only for tower/dynamic consistency;
        * supervised branches use the exact bridge conditioned on the paired
          ground-truth X0 and are used for terminal BCE.

        Attaching the paired X0 label to marginalized dynamic branches would be
        statistically incorrect, because that draw no longer retains the
        particular paired X0 after conditioning on Xt.
        """
        x_0 = x_0.float()
        batch, device = x_0.shape[0], x_0.device
        branch_count = int(self.boundary_branches if branches is None else branches)
        if branch_count < 2:
            raise ValueError("branches must be at least 2")

        source_t, sensitivity_norm, nfe_tensor = self._boundary_metadata(
            batch, device, generator=generator
        )
        with torch.no_grad():
            x_t_probability = self.q_sample(x_0, source_t)
            x_t = torch.bernoulli(x_t_probability, generator=generator)
            source_clean_prob, learned_x1_probability = self._source_prediction(
                x_t=x_t,
                source_t=source_t,
                sensitivity_norm=sensitivity_norm,
                adapter_scale=1.0,
            )
            dynamic_x1 = self._branch_sample(
                learned_x1_probability, branch_count, generator=generator
            )

            target_t = torch.ones_like(source_t)
            bridge_zero, bridge_one = self._endpoint_bridge_probabilities(
                x_t=x_t,
                t_current=source_t,
                t_target=target_t,
            )
            paired_x1_probability = (
                (1.0 - x_0) * bridge_zero + x_0 * bridge_one
            ).clamp(0.0, 1.0)
            paired_x1 = self._branch_sample(
                paired_x1_probability, branch_count, generator=generator
            )

        def flatten_branches(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(batch * branch_count, *value.shape[2:])

        source_repeat = source_clean_prob.repeat_interleave(branch_count, dim=0)
        source_t_repeat = source_t.repeat_interleave(branch_count)
        sensitivity_repeat = sensitivity_norm.repeat_interleave(branch_count)
        adapter_scale_repeat = torch.ones_like(sensitivity_repeat)

        dynamic_logits = self._terminal_logits(
            x1=flatten_branches(dynamic_x1),
            source_clean_prob=source_repeat,
            source_t=source_t_repeat,
            sensitivity_norm=sensitivity_repeat,
            adapter_scale=adapter_scale_repeat,
            enable_repair=enable_repair,
        )
        paired_logits = self._terminal_logits(
            x1=flatten_branches(paired_x1),
            source_clean_prob=source_repeat,
            source_t=source_t_repeat,
            sensitivity_norm=sensitivity_repeat,
            adapter_scale=adapter_scale_repeat,
            enable_repair=enable_repair,
        )

        dynamic_probability = torch.sigmoid(dynamic_logits).reshape(
            batch, branch_count, *x_0.shape[1:]
        )
        dynamic_probability_mean = dynamic_probability.float().mean(dim=1)
        smooth_hard_mean = torch.sigmoid(
            dynamic_logits.float() / self.boundary_hard_tau
        ).reshape(batch, branch_count, *x_0.shape[1:]).mean(dim=1)
        actual_hard_mean = (dynamic_logits.detach() > 0.0).float().reshape(
            batch, branch_count, *x_0.shape[1:]
        ).mean(dim=1)

        paired_target = x_0.repeat_interleave(branch_count, dim=0)
        supervised_bce = F.binary_cross_entropy_with_logits(
            paired_logits.float(), paired_target.float()
        )
        probability_consistency = (
            dynamic_probability_mean - source_clean_prob.detach().float()
        ).square().mean()
        hard_consistency = (
            smooth_hard_mean - source_clean_prob.detach().float()
        ).square().mean()
        if enable_repair and self.boundary_lambda_anchor > 0.0:
            base_for_anchor = self._terminal_logits(
                x1=flatten_branches(dynamic_x1),
                source_clean_prob=source_repeat,
                source_t=source_t_repeat,
                sensitivity_norm=sensitivity_repeat,
                adapter_scale=adapter_scale_repeat,
                enable_repair=False,
            )
            anchor = (dynamic_logits.float() - base_for_anchor.float()).square().mean()
        else:
            anchor = dynamic_logits.new_zeros(())

        loss = (
            self.boundary_lambda_bce * supervised_bce
            + self.boundary_lambda_prob * probability_consistency
            + self.boundary_lambda_hard * hard_consistency
            + self.boundary_lambda_anchor * anchor
        )

        with torch.no_grad():
            local_brier = (source_clean_prob.float() - x_0).square().mean()
            endpoint_probability_brier = (
                dynamic_probability_mean - x_0
            ).square().mean()
            endpoint_hard_brier = (actual_hard_mean - x_0).square().mean()
            probability_gap = (
                dynamic_probability_mean - source_clean_prob
            ).square().mean()
            hard_gap = (actual_hard_mean - source_clean_prob).square().mean()
            hard_sample_bit_error = (
                actual_hard_mean * (1.0 - x_0)
                + (1.0 - actual_hard_mean) * x_0
            ).mean()

        return {
            "loss": loss,
            "boundary_bce": supervised_bce,
            "boundary_prob_consistency": probability_consistency,
            "boundary_soft_hard_consistency": hard_consistency,
            "boundary_anchor": anchor,
            "teacher_local_brier": local_brier,
            "endpoint_probability_brier": endpoint_probability_brier,
            "endpoint_hard_brier": endpoint_hard_brier,
            "probability_realization_gap": probability_gap,
            "hard_realization_gap": hard_gap,
            "hard_sample_bit_error": hard_sample_bit_error,
            "source_t_mean": source_t.float().mean(),
            "nfe_mean": nfe_tensor.float().mean(),
        }

    def _train_loss(self, x_0, label=None, x_ct=None):
        if label is not None or x_ct is not None:
            raise NotImplementedError("The first V9 boundary test supports unconditional LSUN only")
        return self.boundary_objective(x_0, enable_repair=True)

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
        adapter_scale=None,
        adapter_last_intervals=None,
        boundary_repair=True,
    ):
        """V8 sampler with an optional carry correction at physical t=1."""
        del full
        device = next(self._denoise_fn.parameters()).device
        effective_adapter_scale = self.lr_adapter_scale if adapter_scale is None else float(adapter_scale)
        if not np.isfinite(effective_adapter_scale):
            raise ValueError("adapter_scale must be finite")
        if adapter_last_intervals is not None:
            adapter_last_intervals = int(adapter_last_intervals)
            if adapter_last_intervals < 0:
                raise ValueError("adapter_last_intervals must be non-negative")

        if shape is not None:
            x_t = torch.bernoulli(torch.full(shape, 0.5, device=device))
            b = shape[0]
        else:
            x_t = torch.bernoulli(
                torch.full(
                    (b, np.prod(self.shape), self.codebook_size), 0.5, device=device
                )
            )
        if mask is not None:
            mask_tensor = mask["mask"].unsqueeze(0).to(device)
            latent = mask["latent"].unsqueeze(0).to(device)
            x_t = latent * mask_tensor + x_t * (1.0 - mask_tensor)

        sample_steps = self.num_timesteps if sample_steps is None else int(sample_steps)
        if sample_steps < 1 or sample_steps > self.num_timesteps:
            raise ValueError(
                f"sample_steps must be in [1,{self.num_timesteps}], got {sample_steps}"
            )
        sampling_steps = self._sampling_grid(sample_steps)
        if return_all:
            x_all = [x_t]
        if self.dataset == "imagenet":
            if label is None:
                label = (torch.arange(b, device=device) * 100).long()
            else:
                label = torch.full((b,), label, device=device, dtype=torch.long)
        if guidance is not None:
            raise NotImplementedError("V9 guidance is not implemented")

        intervals = [
            (int(sampling_steps[i]), int(sampling_steps[i + 1]))
            for i in range(len(sampling_steps) - 1)
        ]
        boundary_trained_for_grid = sample_steps in self.boundary_nfes
        mean_s2 = float(np.mean([
            self._sensitivity_squared(s=s, t=t) for t, s in intervals
        ])) if intervals else 1.0
        adapter_start = 0
        if adapter_last_intervals is not None:
            adapter_start = max(0, len(intervals) - adapter_last_intervals)

        carry = None
        for index, step_value_raw in enumerate(sampling_steps):
            step_value = int(step_value_raw)
            t = torch.full((b,), step_value, device=device, dtype=torch.long)
            if self.dataset.startswith(("imagenet", "laion", "ising")):
                base_raw, hidden = self._denoise_fn.forward_base_with_hidden(
                    x_t, label=label, time_steps=self._model_time(t)
                )
            else:
                base_raw, hidden = self._denoise_fn.forward_base_with_hidden(
                    x_t, time_steps=self._model_time(t)
                )

            if step_value != 1:
                next_step = int(sampling_steps[index + 1])
                s = torch.full((b,), next_step, device=device, dtype=torch.long)
                raw_s2 = self._sensitivity_squared(s=next_step, t=step_value)
                sens_scalar = np.sqrt(raw_s2 / max(mean_s2, 1.0e-12))
                sens = torch.full((b,), float(sens_scalar), device=device)
                adapter_enabled = (
                    effective_adapter_scale != 0.0
                    and (adapter_last_intervals is None or index >= adapter_start)
                )
                if adapter_enabled:
                    residual = self._denoise_fn.adapter_residual(
                        hidden=hidden,
                        t_current=t,
                        t_target=s,
                        sensitivity_norm=sens,
                    )
                    raw = base_raw + effective_adapter_scale * residual
                else:
                    raw = base_raw
                clean_logits = self._raw_to_clean_logits(raw / float(temp), x_t)
                clean_probability = torch.sigmoid(clean_logits)
                if next_step == 1 and adapter_enabled:
                    carry = {
                        "source_clean_prob": clean_probability,
                        "source_t": t,
                        "sensitivity_norm": sens,
                        "adapter_scale": torch.full(
                            (b,), float(effective_adapter_scale), device=device
                        ),
                    }
                x_next_probability = self._reverse_probability(
                    clean_prob=clean_probability,
                    x_t=x_t,
                    t_current=t,
                    t_target=s,
                )
                x_next = torch.bernoulli(x_next_probability)
            else:
                base_clean_logits = self._raw_to_clean_logits(base_raw, x_t)
                if (
                    bool(boundary_repair)
                    and boundary_trained_for_grid
                    and carry is not None
                ):
                    terminal_residual = self._denoise_fn.terminal_carry_residual(
                        terminal_hidden=hidden,
                        source_clean_prob=carry["source_clean_prob"],
                        source_t=carry["source_t"],
                        sensitivity_norm=carry["sensitivity_norm"],
                        adapter_scale=carry["adapter_scale"],
                    )
                    clean_logits = (base_clean_logits + terminal_residual) / float(temp)
                else:
                    clean_logits = base_clean_logits / float(temp)
                clean_probability = torch.sigmoid(clean_logits)
                x_next = (
                    (clean_probability > 0.5).float()
                    if self.hard_final
                    else torch.bernoulli(clean_probability)
                )

            x_t = x_next
            if mask is not None:
                x_t = latent * mask_tensor + x_t * (1.0 - mask_tensor)
            if return_all:
                x_all.append(x_t)
        return torch.cat(x_all, dim=0) if return_all else x_t
