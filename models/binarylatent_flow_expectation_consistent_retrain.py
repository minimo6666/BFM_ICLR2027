"""Bernoulli Flow sampler with expectation-consistent X0 marginalization.

This is a drop-in replacement for ``binarylatent_flow_decouple_correct_t(7).py``.
The denoiser and the main BCE X0/flip-prediction objective are unchanged.

Main correction
---------------
Let

    m_theta = P_theta(X0 = 1 | Xt)

be the clean-bit posterior predicted by the network.  The legacy sampler first
inserted this soft value into the oracle bridge and then applied Bayes
normalization, i.e. F(m_theta).  Because the bridge posterior is nonlinear in
X0, this is generally not the same as marginalizing the two valid clean states.

This implementation instead computes

    P_theta(Xs = 1 | Xt)
      = (1 - m_theta) P(Xs = 1 | Xt, X0 = 0)
        + m_theta     P(Xs = 1 | Xt, X0 = 1).

The optional auxiliary reverse-posterior loss is corrected in the same way.
The primary BCE denoising loss remains unchanged, so checkpoints trained with
aux == 0 can be evaluated with this sampler without retraining.
"""

from __future__ import annotations

import os
import pdb
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def g_function_symbol(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Apply a binary-symmetric channel with flip probability ``y``.

    If the final dimension of ``x`` stores candidate states [1, 0], the output
    stores the corresponding likelihoods of the observed bit after the channel.
    """
    if x.dim() != y.dim():
        y = y.unsqueeze(-1).unsqueeze(-1)
    return x * (1.0 - y) + (1.0 - x) * y


def g_function_binary(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Legacy helper retained for compatibility with external imports."""
    if x.dim() != y.dim():
        y = y.unsqueeze(-1).unsqueeze(-1)
    return x * y + (1.0 - x) * (1.0 - y)


class BinaryDiffusionFlowDecouple(nn.Module):
    def __init__(self, H, denoise_fn, mask_id):
        super().__init__()

        self.num_classes = H.codebook_size
        self.latent_emb_dim = H.emb_dim
        self.shape = tuple(H.latent_shape)
        self.num_timesteps = int(H.total_steps)

        self.mask_id = mask_id
        self._denoise_fn = denoise_fn
        self.n_samples = H.batch_size
        self.loss_type = H.loss_type
        self.mask_schedule = H.mask_schedule

        self.loss_final = H.loss_final
        self.use_softmax = H.use_softmax

        self.p_flip = H.p_flip
        self.focal = H.focal
        self.aux = H.aux
        self.dataset = H.dataset
        self.guidance = H.guidance

        # Index 0 is clean (tau=1); index T is pure Bernoulli noise (tau=0).
        # There are T+1 endpoints and T reverse transitions.
        self.register_buffer(
            "interpolation_t",
            torch.linspace(1.0, 0.0, steps=self.num_timesteps + 1),
            persistent=False,
        )

        self.codebook_size = H.codebook_size
        self.block_size = H.block_size
        self.image_size = H.img_size

        # Default to the corrected expectation-consistent construction.
        # Set H.x0_posterior_mode = "legacy_plugin" only for a controlled
        # ablation against the original F(m_theta) implementation.
        configured_posterior_mode = getattr(H, "x0_posterior_mode", None)
        self.x0_posterior_mode = str(
            configured_posterior_mode or "expectation_consistent"
        ).lower()
        valid_modes = {"expectation_consistent", "legacy_plugin"}
        if self.x0_posterior_mode not in valid_modes:
            raise ValueError(
                f"Unknown x0_posterior_mode={self.x0_posterior_mode!r}; "
                f"expected one of {sorted(valid_modes)}."
            )

        configured_posterior_eps = getattr(H, "posterior_eps", None)
        self.posterior_eps = float(
            1e-12 if configured_posterior_eps is None else configured_posterior_eps
        )

        # Preserve the original deterministic final projection by default so
        # that legacy/new comparisons isolate only the posterior construction.
        configured_hard_final = getattr(H, "hard_final", None)
        self.hard_final = (
            True if configured_hard_final is None else bool(configured_hard_final)
        )

        # Auxiliary posterior training interval.  "mixed" uses one-step targets
        # for half the batch and uniformly sampled arbitrary-interval targets
        # for the other half.  This directly trains the low-NFE regime while
        # preserving local reverse-transition supervision.
        configured_aux_interval = getattr(H, "aux_interval_mode", None) or "mixed"
        self.aux_interval_mode = str(
            os.environ.get(
                "BFM_AUX_INTERVAL_MODE",
                configured_aux_interval,
            )
        ).lower()
        if self.aux_interval_mode not in {"one_step", "random", "mixed"}:
            raise ValueError(
                f"Unknown aux_interval_mode={self.aux_interval_mode!r}; "
                "expected one_step, random, or mixed."
            )

    def sample_time(self, b: int, device: torch.device) -> torch.Tensor:
        return torch.randint(
            1, self.num_timesteps + 1, (b,), device=device
        ).long()

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return q(X_t=1 | X_0) for the global Bernoulli path."""
        interpolation = self.interpolation_t.to(x_0.device)
        tau_t = interpolation[t]
        dim = x_0.ndim - 1
        tau_t = tau_t.view(-1, *([1] * dim))
        return (1.0 + (2.0 * x_0 - 1.0) * tau_t) / 2.0

    def _cross_step_flip_probability(
        self,
        t_current: torch.Tensor,
        t_target: torch.Tensor,
        reference_state_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Return the exact forward flip probability target -> current.

        ``t_target < t_current`` in the reverse sampler.  Since index 0 is the
        clean endpoint, tau_target > tau_current.
        """
        interpolation = self.interpolation_t.to(reference_state_tensor.device)
        tau_current = interpolation[t_current]
        tau_target = interpolation[t_target]

        gamma = 0.5 * (tau_target - tau_current) / (
            tau_target + self.posterior_eps
        )
        gamma = gamma.clamp(min=0.0, max=0.5)

        # reference_state_tensor has shape [B, ..., 2].
        return gamma.view(
            -1, *([1] * (reference_state_tensor.ndim - 1))
        )

    def _endpoint_bridge_probabilities(
        self,
        x_t: torch.Tensor,
        t_current: torch.Tensor,
        t_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the two exact oracle bridge probabilities.

        Returns
        -------
        posterior_x0_zero:
            P(X_target=1 | X_current=x_t, X0=0)
        posterior_x0_one:
            P(X_target=1 | X_current=x_t, X0=1)
        """
        x_t = x_t.float()
        x_t_states = torch.stack((x_t, 1.0 - x_t), dim=-1)

        gamma = self._cross_step_flip_probability(
            t_current=t_current,
            t_target=t_target,
            reference_state_tensor=x_t_states,
        )
        likelihood = g_function_symbol(x_t_states, gamma)

        x0_zero = torch.zeros_like(x_t)
        x0_one = torch.ones_like(x_t)

        alpha_zero = self.q_sample(x0_zero, t_target)
        alpha_one = self.q_sample(x0_one, t_target)

        prior_zero = torch.stack((alpha_zero, 1.0 - alpha_zero), dim=-1)
        prior_one = torch.stack((alpha_one, 1.0 - alpha_one), dim=-1)

        unnormalized_zero = prior_zero * likelihood
        unnormalized_one = prior_one * likelihood

        posterior_zero = unnormalized_zero / unnormalized_zero.sum(
            dim=-1, keepdim=True
        ).clamp_min(self.posterior_eps)
        posterior_one = unnormalized_one / unnormalized_one.sum(
            dim=-1, keepdim=True
        ).clamp_min(self.posterior_eps)

        return posterior_zero[..., 0], posterior_one[..., 0]

    def _expectation_consistent_reverse_probability(
        self,
        clean_prob: torch.Tensor,
        x_t: torch.Tensor,
        t_current: torch.Tensor,
        t_target: torch.Tensor,
    ) -> torch.Tensor:
        """Marginalize exact bridges using P_theta(X0=1 | Xt)."""
        posterior_zero, posterior_one = self._endpoint_bridge_probabilities(
            x_t=x_t,
            t_current=t_current,
            t_target=t_target,
        )
        reverse_prob = (
            (1.0 - clean_prob) * posterior_zero
            + clean_prob * posterior_one
        )
        return reverse_prob.clamp(0.0, 1.0)

    def _legacy_plugin_reverse_probability(
        self,
        clean_prob: torch.Tensor,
        x_t: torch.Tensor,
        t_current: torch.Tensor,
        t_target: torch.Tensor,
    ) -> torch.Tensor:
        """Reproduce the original nonlinear soft-X0 plug-in for ablations."""
        x_t = x_t.float()
        alpha_soft = self.q_sample(clean_prob, t_target)
        prior_soft = torch.stack((alpha_soft, 1.0 - alpha_soft), dim=-1)

        x_t_states = torch.stack((x_t, 1.0 - x_t), dim=-1)
        gamma = self._cross_step_flip_probability(
            t_current=t_current,
            t_target=t_target,
            reference_state_tensor=x_t_states,
        )
        likelihood = g_function_symbol(x_t_states, gamma)

        unnormalized = prior_soft * likelihood
        posterior = unnormalized / unnormalized.sum(
            dim=-1, keepdim=True
        ).clamp_min(self.posterior_eps)
        return posterior[..., 0].clamp(0.0, 1.0)

    def _reverse_probability(
        self,
        clean_prob: torch.Tensor,
        x_t: torch.Tensor,
        t_current: torch.Tensor,
        t_target: torch.Tensor,
    ) -> torch.Tensor:
        if self.x0_posterior_mode == "legacy_plugin":
            return self._legacy_plugin_reverse_probability(
                clean_prob=clean_prob,
                x_t=x_t,
                t_current=t_current,
                t_target=t_target,
            )
        return self._expectation_consistent_reverse_probability(
            clean_prob=clean_prob,
            x_t=x_t,
            t_current=t_current,
            t_target=t_target,
        )


    def _sample_aux_target_time(self, t_current: torch.Tensor) -> torch.Tensor:
        """Sample a valid cleaner endpoint in [0, t_current - 1]."""
        one_step = t_current - 1
        if self.aux_interval_mode == "one_step":
            return one_step

        random_target = torch.floor(
            torch.rand_like(t_current, dtype=torch.float32)
            * t_current.float()
        ).long()
        if self.aux_interval_mode == "random":
            return random_target

        use_random = torch.rand_like(
            t_current, dtype=torch.float32
        ) < 0.5
        return torch.where(use_random, random_target, one_step)

    def _train_loss(self, x_0, label=None, x_ct=None):
        x_0 = x_0.float()
        b, device = x_0.size(0), x_0.device

        t = self.sample_time(b, device)

        if x_ct is None:
            x_t = self.q_sample(x_0, t)
        else:
            raise NotImplementedError(
                "x_ct-conditioned training is not implemented in this sampler."
            )

        x_t_in = torch.bernoulli(x_t)
        if label is not None:
            if self.guidance and np.random.random() < 0.1:
                label = None
            raw_logits = self._denoise_fn(
                idx=x_t_in, label=label, time_steps=t
            )
        else:
            raw_logits = self._denoise_fn(x_t_in, time_steps=t)

        # Convert a flip logit into the corresponding clean-X0 logit.  This is
        # exact because sigmoid(-l) = 1 - sigmoid(l).
        if self.p_flip:
            clean_logits = x_t_in * (-raw_logits) + (1.0 - x_t_in) * raw_logits

            if self.focal >= 0:
                flip_target = torch.logical_xor(
                    x_0.bool(), x_t_in.bool()
                ).float()
                kl_loss = focal_loss(
                    raw_logits, flip_target, gamma=self.focal
                )
            else:
                kl_loss = F.binary_cross_entropy_with_logits(
                    clean_logits, x_0, reduction="none"
                )
        else:
            clean_logits = raw_logits
            if self.focal >= 0:
                kl_loss = focal_loss(
                    clean_logits,
                    x_0,
                    alpha=self.focal,
                    gamma=self.focal,
                )
            else:
                kl_loss = F.binary_cross_entropy_with_logits(
                    clean_logits, x_0, reduction="none"
                )

        if torch.isinf(kl_loss).any() or torch.isnan(kl_loss).any():
            pdb.set_trace()

        if self.loss_final == "weighted":
            weight = (1.0 - (t / self.num_timesteps)).view(-1, 1, 1)
        elif self.loss_final == "mean":
            weight = 1.0
        else:
            raise NotImplementedError(
                f"Unknown loss_final={self.loss_final!r}."
            )

        loss = (weight * kl_loss).mean()
        kl_loss_mean = kl_loss.mean()

        with torch.no_grad():
            if self.use_softmax:
                acc = (
                    (
                        (clean_logits[..., 1] > clean_logits[..., 0]).float()
                        == x_0.view(-1)
                    ).float().sum()
                    / float(x_0.numel())
                )
            else:
                acc = (
                    ((clean_logits > 0.0).float() == x_0).float().sum()
                    / float(x_0.numel())
                )

        if self.aux > 0:
            clean_prob = torch.sigmoid(clean_logits)
            t_target = self._sample_aux_target_time(t)

            # Correct predicted reverse marginal:
            # (1-m_theta) B(X0=0) + m_theta B(X0=1).
            x_tm1_prob = self._expectation_consistent_reverse_probability(
                clean_prob=clean_prob,
                x_t=x_t_in,
                t_current=t,
                t_target=t_target,
            )

            # Exact oracle target conditioned on the sampled binary X0.
            posterior_zero, posterior_one = self._endpoint_bridge_probabilities(
                x_t=x_t_in,
                t_current=t,
                t_target=t_target,
            )
            x_tm1_gt = (
                (1.0 - x_0) * posterior_zero
                + x_0 * posterior_one
            )

            with torch.autocast(
                device_type=x_tm1_prob.device.type, enabled=False
            ):
                aux_loss = F.binary_cross_entropy(
                    x_tm1_prob.float().clamp(1e-6, 1.0 - 1e-6),
                    x_tm1_gt.float().clamp(0.0, 1.0),
                    reduction="none",
                )

            aux_loss = (weight * aux_loss).mean()
            reverse_mae = (x_tm1_prob - x_tm1_gt).abs().mean()
            mean_aux_gap = (t - t_target).float().mean()
            loss = loss + self.aux * aux_loss

        stats = {
            "loss": loss,
            "bce_loss": kl_loss_mean,
            "acc": acc,
        }
        if self.aux > 0:
            stats["aux loss"] = aux_loss
            stats["reverse_mae"] = reverse_mae
            stats["mean_aux_gap"] = mean_aux_gap
        return stats

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
        del full  # retained only for API compatibility

        device = next(self._denoise_fn.parameters()).device
        if shape is not None:
            x_t = torch.bernoulli(
                0.5 * torch.ones(shape, device=device, dtype=torch.float32)
            )
            b = shape[0]
        else:
            x_t = torch.bernoulli(
                0.5
                * torch.ones(
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
                f"sample_steps must be in [1, {self.num_timesteps}], "
                f"got {sample_steps}."
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

            if (
                self.dataset.startswith("imagenet")
                or self.dataset.startswith("laion")
                or self.dataset.startswith("ising")
            ):
                raw_logits = self._denoise_fn(
                    x_t, time_steps=t, label=label
                )
                raw_logits = raw_logits / temp

                if guidance is not None:
                    raw_logits_uncond = self._denoise_fn(
                        x_t, time_steps=t, y=None
                    )
                    raw_logits_uncond = raw_logits_uncond / temp
                    raw_logits = (
                        (1.0 + guidance) * raw_logits
                        - guidance * raw_logits_uncond
                    )
            else:
                raw_logits = self._denoise_fn(x_t, time_steps=t)
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
                    (b,), next_step_value, device=device, dtype=torch.long
                )

                x_target_prob = self._reverse_probability(
                    clean_prob=clean_prob,
                    x_t=x_t,
                    t_current=t,
                    t_target=t_target,
                )
                x_next = torch.bernoulli(x_target_prob)
            else:
                # At target time 0, the expectation-consistent reverse marginal
                # is exactly clean_prob.  Keep the original hard projection by
                # default for a controlled comparison.
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

    def forward(self, x, label=None, x_t=None):
        return self._train_loss(x, label, x_t)


def focal_loss(inputs, targets, alpha=-1, gamma=1):
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    p_t = (1.0 - p_t).clamp(min=1e-6, max=1.0 - 1e-6)
    loss = ce_loss * (p_t**gamma)

    if alpha == -1:
        neg_weight = targets.sum((-1, -2))
        neg_weight = neg_weight / targets[0].numel()
        neg_weight = neg_weight.view(-1, 1, 1)
        alpha_t = (
            (1.0 - neg_weight) * targets
            + neg_weight * (1.0 - targets)
        )
        loss = alpha_t * loss
    elif alpha > 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
    return loss
