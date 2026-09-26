"""Corrected samplers for the project's binary uniform-source DFM.

This module keeps the *trained DFM model and probability path unchanged* and
only replaces the finite-step CTMC sampler following Wan et al.,
"Corrected Samplers for Discrete Flow Models" (ICML 2026).

Compatibility
-------------
The existing BinaryDiscreteFlowModelDecouple uses the linear uniform-source path

    p_t(x_t | x_1) = t delta(x_t=x_1) + (1-t) / 2,

and predicts the clean posterior p(x_1 | x_t).  This is exactly the
MixtureDiscreteProbPath with a linear scheduler used by the official corrected
sampler toy experiment, specialized to K=2 states.

No additional trainable parameters are introduced, so an existing
dfm_binary_ema_*.th checkpoint can be loaded directly.

Implemented samplers
--------------------
- paper_euler: official paper-style frozen-rate Euler step on the optimized grid.
- time_corrected: integrates the singular time coefficient analytically while
  freezing the posterior over each interval.
- location_corrected: samples the first global jump location/time, re-evaluates
  the posterior there, and then performs a time-corrected remainder step.

For every method, ``sample_steps`` denotes the number of integration intervals.
Location correction can evaluate the model twice per interval, so its nominal
NFE is twice ``sample_steps``. In the unlikely event that no sample in a batch
has an exit inside an interval, the second call is skipped. The realized number
is reported in ``last_sampling_stats``.

The corrected solvers stop at t=1-delta because the linear mixture CTMC rate is
singular at t=1.  The official uniform-source simulation does the same.  For
image generation a small delta (default 1e-4) is recommended and should be
reported.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch

from binarylatent_dfm_uniform import BinaryDiscreteFlowModelDecouple


class BinaryCorrectedDFMDecouple(BinaryDiscreteFlowModelDecouple):
    """Inference-only corrected samplers on top of the existing Binary DFM."""

    def __init__(self, H, denoise_fn, mask_id):
        super().__init__(H, denoise_fn, mask_id)
        self.last_sampling_stats = {}

    @staticmethod
    def _optimized_grid(
        n_segments: int,
        delta: float,
        device: torch.device,
    ) -> torch.Tensor:
        if n_segments <= 0:
            raise ValueError(f"n_segments must be positive, got {n_segments}.")
        if not (0.0 < delta < 1.0):
            raise ValueError(f"delta must be in (0,1), got {delta}.")
        # Official uniform-source grid:
        # t_i = 1 - delta^(i/N), i=0,...,N.
        i = torch.arange(n_segments + 1, device=device, dtype=torch.float64)
        grid = 1.0 - torch.pow(
            torch.tensor(delta, device=device, dtype=torch.float64),
            i / float(n_segments),
        )
        grid[0] = 0.0
        grid[-1] = 1.0 - float(delta)
        return grid.to(torch.float32)

    @staticmethod
    def _uniform_grid(
        n_segments: int,
        delta: float,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.linspace(
            0.0,
            1.0 - float(delta),
            steps=n_segments + 1,
            device=device,
            dtype=torch.float32,
        )

    def _sample_setup(
        self,
        b: int,
        shape,
        label,
        mask,
    ):
        try:
            device = next(self._denoise_fn.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if shape is not None:
            sample_shape: Tuple[int, ...] = tuple(shape)
            if len(sample_shape) < 2:
                raise ValueError(
                    f"shape must include batch and data dimensions, got {shape}."
                )
            b = sample_shape[0]
        else:
            sample_shape = (b, int(np.prod(self.shape)), self.codebook_size)

        x_t = torch.randint(
            low=0,
            high=self.state_space_size,
            size=sample_shape,
            device=device,
            dtype=torch.long,
        ).float()

        m = None
        fixed_latent = None
        if mask is not None:
            m = mask["mask"].unsqueeze(0).to(device=device, dtype=x_t.dtype)
            fixed_latent = mask["latent"].unsqueeze(0).to(
                device=device, dtype=x_t.dtype
            )
            x_t = fixed_latent * m + x_t * (1.0 - m)

        if self.dataset == "imagenet":
            if label is None:
                label = (torch.arange(b, device=device) * 100).long()
            elif not torch.is_tensor(label):
                label = torch.full(
                    (b,), label, device=device, dtype=torch.long
                )

        return device, b, x_t, label, m, fixed_latent

    @staticmethod
    def _reapply_fixed(x_t, m, fixed_latent):
        if m is None:
            return x_t
        return fixed_latent * m + x_t * (1.0 - m)

    def _clean_probs_at(
        self,
        x_t: torch.Tensor,
        flow_t: torch.Tensor,
        temp: float,
        label=None,
        guidance=None,
    ) -> torch.Tensor:
        network_time = self._network_time_from_flow_time(flow_t)
        clean_logits = self._predict_clean_logits(
            x_t=x_t,
            network_time=network_time,
            temp=float(temp),
            label=label,
            guidance=guidance,
        )
        return self._binary_class_probs(clean_logits)

    @staticmethod
    def _opposite_prob(
        x_t: torch.Tensor,
        clean_probs: torch.Tensor,
    ) -> torch.Tensor:
        """p(x_1 != current x_t | x_t), one value per binary coordinate."""
        p0 = clean_probs[..., 0]
        p1 = clean_probs[..., 1]
        return torch.where(x_t > 0.5, p0, p1)

    @staticmethod
    def _bernoulli_flip(
        x_t: torch.Tensor,
        flip_prob: torch.Tensor,
    ) -> torch.Tensor:
        flip_prob = flip_prob.clamp(0.0, 1.0)
        do_flip = torch.rand_like(flip_prob) < flip_prob
        return torch.where(do_flip, 1.0 - x_t, x_t)

    @torch.no_grad()
    def sample_corrected(
        self,
        method: str,
        temp: float = 1.0,
        sample_steps: Optional[int] = None,
        b: int = 8,
        shape=None,
        return_all: bool = False,
        label=None,
        mask=None,
        guidance=None,
        delta: float = 0.05,
        opt_grid: bool = True,
    ):
        """Sample with one of the corrected DFM solvers.

        Args:
            method:
                "paper_euler", "time_corrected", or "location_corrected".
            sample_steps:
                Number of integration intervals for every sampler. Location
                correction can use two denoiser evaluations per interval.
            delta:
                Stop at t=1-delta to avoid the linear-path singularity.
            opt_grid:
                Use the paper's geometric remaining-noise grid.
        """
        method = str(method).lower()
        aliases = {
            "euler": "paper_euler",
            "paper-euler": "paper_euler",
            "tc": "time_corrected",
            "timecorrected": "time_corrected",
            "time-corrected": "time_corrected",
            "lc": "location_corrected",
            "locationcorrected": "location_corrected",
            "location-corrected": "location_corrected",
        }
        method = aliases.get(method, method)
        if method not in {
            "paper_euler",
            "time_corrected",
            "location_corrected",
        }:
            raise ValueError(f"Unknown corrected DFM method: {method!r}.")

        if sample_steps is None:
            sample_steps = self.num_timesteps
        sample_steps = int(sample_steps)
        if sample_steps <= 0:
            raise ValueError(f"sample_steps must be positive, got {sample_steps}.")
        device, b, x_t, label, m, fixed_latent = self._sample_setup(
            b=b, shape=shape, label=label, mask=mask
        )

        n_segments = sample_steps
        grid_fn = self._optimized_grid if opt_grid else self._uniform_grid
        time_grid = grid_fn(n_segments, float(delta), device)

        history = [x_t.clone()] if return_all else None
        realized_nfe = 0
        active_second_calls = 0

        for seg_idx in range(n_segments):
            t_left = float(time_grid[seg_idx].item())
            t_right = float(time_grid[seg_idx + 1].item())

            flow_t = torch.full(
                (b,), t_left, device=device, dtype=torch.float32
            )
            probs = self._clean_probs_at(
                x_t, flow_t, temp=temp, label=label, guidance=guidance
            )
            realized_nfe += 1

            p_other = self._opposite_prob(x_t, probs)

            if method == "paper_euler":
                # Official paper-style Euler:
                # intensity = h * [dot{kappa}/(1-kappa)] * p_other.
                h = t_right - t_left
                coeff = h / max(
                    1.0 - t_left, torch.finfo(torch.float32).eps
                )
                intensity = coeff * p_other
                flip_prob = -torch.expm1(-intensity)
                x_t = self._bernoulli_flip(x_t, flip_prob)

            elif method == "time_corrected":
                # Analytically integrate dot{kappa}/(1-kappa) for linear kappa=t:
                # integral = log((1-t_left)/(1-t_right)).
                c = math.log(
                    max(1.0 - t_left, 1e-30)
                    / max(1.0 - t_right, 1e-30)
                )
                intensity = c * p_other
                flip_prob = -torch.expm1(-intensity)
                x_t = self._bernoulli_flip(x_t, flip_prob)

            else:
                # Location-corrected, binary specialization of the official
                # uniform-source solver.
                #
                # First global event rate per sample is sum_d p_other[d].
                flat_p = p_other.reshape(b, -1)
                total_rate = flat_p.sum(dim=1)
                positive = total_rate > 0

                exp_dist = torch.full(
                    (b,), float("inf"), device=device, dtype=torch.float32
                )
                if positive.any():
                    u = torch.rand(
                        int(positive.sum().item()),
                        device=device,
                        dtype=torch.float32,
                    ).clamp_min(torch.finfo(torch.float32).tiny)
                    exp_dist[positive] = -torch.log(u) / total_rate[positive]

                # Linear scheduler inverse:
                # t_exit = 1 - (1-t_left) exp(-E).
                exit_times = 1.0 - (1.0 - t_left) * torch.exp(-exp_dist)
                active = positive & (exit_times < t_right)

                if active.any():
                    active_indices = torch.nonzero(active, as_tuple=False).flatten()

                    # Sample one coordinate according to off-diagonal posterior
                    # mass. In a binary state space, its target value is
                    # deterministically the opposite bit.
                    weights = flat_p[active]
                    weight_sum = weights.sum(dim=1, keepdim=True)
                    weights = weights / weight_sum.clamp_min(
                        torch.finfo(weights.dtype).eps
                    )
                    coord = torch.multinomial(weights, num_samples=1).squeeze(1)

                    x_active = x_t[active].clone()
                    x_active_flat = x_active.reshape(x_active.shape[0], -1)
                    rows = torch.arange(
                        x_active_flat.shape[0], device=device
                    )
                    x_active_flat[rows, coord] = 1.0 - x_active_flat[rows, coord]
                    x_active = x_active_flat.reshape_as(x_active)
                    x_t[active] = x_active

                    # Re-evaluate posterior at the sampled first-event location.
                    # This is the location correction and is the second NFE.
                    active_label = None
                    if torch.is_tensor(label):
                        active_label = label[active]
                    elif label is not None:
                        active_label = label

                    probs2 = self._clean_probs_at(
                        x_t[active],
                        exit_times[active],
                        temp=temp,
                        label=active_label,
                        guidance=guidance,
                    )
                    realized_nfe += 1
                    active_second_calls += 1

                    p_other2 = self._opposite_prob(x_t[active], probs2)
                    c = torch.log(
                        (1.0 - exit_times[active]).clamp_min(1e-30)
                        / max(1.0 - t_right, 1e-30)
                    )
                    c = c.view(
                        c.shape[0], *([1] * (p_other2.ndim - 1))
                    )
                    intensity2 = c * p_other2
                    flip_prob2 = -torch.expm1(-intensity2)
                    x_t[active] = self._bernoulli_flip(
                        x_t[active], flip_prob2
                    )

            x_t = self._reapply_fixed(x_t, m, fixed_latent)
            if return_all:
                history.append(x_t.clone())

        self.last_sampling_stats = {
            "method": method,
            "requested_steps": sample_steps,
            "nominal_nfe": sample_steps * (2 if method == "location_corrected" else 1),
            "realized_batch_model_calls": realized_nfe,
            "segments": n_segments,
            "second_calls_with_active_samples": active_second_calls,
            "delta": float(delta),
            "opt_grid": bool(opt_grid),
            "terminal_flow_t": float(time_grid[-1].item()),
        }

        if return_all:
            return torch.cat(history, dim=0)
        return x_t
