"""Binary uniform-source Discrete Flow Model (DFM) baseline.

This module adapts the uniform-prior Discrete Flow Model of
Campbell et al., "Generative Flows on Discrete State-Spaces"
(ICML 2024, arXiv:2402.04997) to the binary latent interface used by
BFM_NIPS26.

The implementation follows the authors' official uniform toy code:
  https://github.com/andrew-cr/discrete_flow_models/
  notebooks/toycode_uniform.ipynb

Important distinction
---------------------
This is Campbell et al.'s CTMC-based DFM, not Gat et al.'s 2024
"Discrete Flow Matching" method. In a paper/rebuttal, name the baseline
accordingly, e.g. "Binary DFM (Campbell et al.)".

Binary specialization
---------------------
Each bit is a categorical variable with S=2 states and uniform source
p_0(x)=1/2. The conditional probability flow is

    p_t(x_t | x_1) = t * delta(x_t=x_1) + (1-t) / 2,

where t=0 is uniform binary noise and t=1 is clean data. The denoiser
predicts the clean bit x_1 with BCE. Sampling uses the finite-step Euler
simulation of the learned CTMC rates from the official implementation.

Drop-in interface
-----------------
The constructor, forward(), _train_loss(), q_sample(), and sample()
signatures match binarylatent_flow_decouple_correct_t.py closely. The
existing denoise_fn still outputs one logit per bit.

Recommended faithful baseline settings
---------------------------------------
  --loss_final mean
  --aux 0
  focal < 0
  dfm_stochasticity = 0.0  (minimum-jump rate R*_t)

A positive dfm_stochasticity is supported through H.dfm_stochasticity
(or H.dfm_noise), but coarse Euler grids can require probability clipping,
as in the authors' official toy implementation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class BinaryDiscreteFlowModelDecouple(nn.Module):
    """Binary uniform-source DFM using the project's existing denoiser."""

    state_space_size: int = 2

    def __init__(self, H, denoise_fn: nn.Module, mask_id):
        super().__init__()

        self.num_classes = H.codebook_size
        self.latent_emb_dim = H.emb_dim
        self.shape = tuple(H.latent_shape)
        self.num_timesteps = int(H.total_steps)

        if self.num_timesteps <= 0:
            raise ValueError(f"H.total_steps must be positive, got {self.num_timesteps}.")

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

        self.codebook_size = H.codebook_size
        self.block_size = H.block_size
        self.image_size = H.img_size

        # Campbell et al.'s extra CTMC stochasticity eta (called noise/N in
        # their minimal code). eta=0 gives the minimum-jump rate R*_t.
        _dfm_stoch = getattr(H, "dfm_stochasticity", None)
        if _dfm_stoch is None:
            _dfm_stoch = getattr(H, "dfm_noise", 0.0)
        if _dfm_stoch is None:
            _dfm_stoch = 0.0
        self.dfm_stochasticity = float(_dfm_stoch)
        if self.dfm_stochasticity < 0:
            raise ValueError(
                "dfm_stochasticity must be non-negative, got "
                f"{self.dfm_stochasticity}."
            )

        # TransformerDecouple uses a sinusoidal time embedding and accepts
        # floating point values. Keeping continuous time is closest to the
        # official DFM objective. Set H.dfm_continuous_time=False only if a
        # replacement denoiser requires integer time indices.
        _continuous = getattr(H, "dfm_continuous_time", None)
        self.dfm_continuous_time = True if _continuous is None else bool(_continuous)

        # IMPORTANT for fixed-checkpoint low-NFE evaluation: TransformerDecouple
        # currently builds its sinusoidal time scale from H.sample_steps. If the
        # model is re-instantiated with sample_steps=16 after being trained with
        # 256, the same numerical time receives a different embedding. Pin the
        # denominator to H.total_steps so inference NFE changes only the Euler
        # grid, not the learned model's time conditioning.
        self._pin_denoiser_time_scale_to_training_horizon()

        # Retained for compatibility with the BFM class. Index tau follows the
        # project's convention: tau=0 is clean and tau=T is noise. DFM flow
        # time is the reverse orientation: t_DFM = 1 - tau/T.
        self.register_buffer(
            "interpolation_t",
            torch.linspace(1.0, 0.0, steps=self.num_timesteps + 1),
            persistent=False,
        )

        if self.aux and self.aux > 0:
            raise ValueError(
                "Campbell DFM does not use BFM's auxiliary posterior loss. "
                "Set H.aux=0 for a clean baseline."
            )

    def _pin_denoiser_time_scale_to_training_horizon(self) -> None:
        """Keep the time embedding invariant when inference NFE changes."""
        time_module = getattr(self._denoise_fn, "time_step_embedding", None)
        base_embedding = getattr(time_module, "emb", None)
        if base_embedding is not None and hasattr(base_embedding, "num_steps"):
            base_embedding.num_steps = float(self.num_timesteps)

    # ------------------------------------------------------------------
    # Time/path helpers
    # ------------------------------------------------------------------
    def sample_time(self, b: int, device: torch.device) -> torch.Tensor:
        """Sample continuous DFM times t in (0,1), as in the official code."""
        eps = torch.finfo(torch.float32).eps
        return torch.rand((b,), device=device, dtype=torch.float32).clamp(eps, 1.0 - eps)

    def _network_time_from_flow_time(self, flow_t: torch.Tensor) -> torch.Tensor:
        """Map DFM time (0=noise, 1=data) to the project's denoiser time.

        The existing BFM code conditions the network with tau in [1,T], where
        tau=T is noise and tau close to 0 is data. Hence tau=(1-t_DFM)T.
        """
        model_time = (1.0 - flow_t) * float(self.num_timesteps)
        model_time = model_time.clamp(min=0.0, max=float(self.num_timesteps))
        if self.dfm_continuous_time:
            return model_time
        return model_time.round().clamp(min=1, max=self.num_timesteps).long()

    @staticmethod
    def _broadcast_batch_scalar(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return value.view(value.shape[0], *([1] * (reference.ndim - 1)))

    def q_sample(self, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return p(x_t=1 | x_1) for compatibility with the BFM interface.

        Here ``t`` uses the project's reverse index convention (0=data,
        T=noise). The equivalent DFM time is t_DFM=1-t/T, so

            p(x_t=1 | x_1) = t_DFM*x_1 + (1-t_DFM)*0.5.
        """
        if not torch.is_tensor(t):
            t = torch.as_tensor(t, device=x_1.device)
        t = t.to(device=x_1.device, dtype=torch.float32)
        if t.ndim == 0:
            t = t.expand(x_1.shape[0])
        flow_t = 1.0 - t / float(self.num_timesteps)
        flow_t = flow_t.clamp(0.0, 1.0)
        flow_t = self._broadcast_batch_scalar(flow_t, x_1)
        return flow_t * x_1 + (1.0 - flow_t) * 0.5

    def _sample_conditional_flow(
        self, x_1: torch.Tensor, flow_t: torch.Tensor
    ) -> torch.Tensor:
        """Sample the official uniform conditional flow exactly.

        With probability t keep the clean state; otherwise replace it with an
        independent uniform binary state. This is identical in distribution to
        Bernoulli(t*x_1 + (1-t)/2), but mirrors the authors' reference code.
        """
        t_view = self._broadcast_batch_scalar(flow_t, x_1)
        uniform_source = torch.randint(
            low=0,
            high=self.state_space_size,
            size=x_1.shape,
            device=x_1.device,
            dtype=torch.long,
        ).to(dtype=x_1.dtype)
        keep_clean = torch.rand_like(x_1, dtype=torch.float32) < t_view
        return torch.where(keep_clean, x_1, uniform_source)

    # ------------------------------------------------------------------
    # Denoiser helpers
    # ------------------------------------------------------------------
    def _call_denoiser(
        self,
        x_t: torch.Tensor,
        network_time: torch.Tensor,
        label=None,
    ) -> torch.Tensor:
        if label is None:
            return self._denoise_fn(x_t, time_steps=network_time)
        return self._denoise_fn(idx=x_t, label=label, time_steps=network_time)

    def _predict_clean_logits(
        self,
        x_t: torch.Tensor,
        network_time: torch.Tensor,
        temp: float = 1.0,
        label=None,
        guidance: Optional[float] = None,
    ) -> torch.Tensor:
        if temp <= 0:
            raise ValueError(f"temp must be positive, got {temp}.")

        conditional_logits = self._call_denoiser(x_t, network_time, label=label) / temp

        if guidance is not None:
            if label is None:
                raise ValueError("Classifier-free guidance requires a conditional label.")
            unconditional_logits = self._call_denoiser(
                x_t, network_time, label=None
            ) / temp
            conditional_logits = (
                (1.0 + guidance) * conditional_logits
                - guidance * unconditional_logits
            )

        # p_flip is only a prediction reparameterization. Convert its raw
        # flip/no-flip logit back to a clean x_1 logit before CTMC sampling.
        if self.p_flip:
            conditional_logits = (
                x_t * (-conditional_logits) + (1.0 - x_t) * conditional_logits
            )

        return conditional_logits

    @staticmethod
    def _binary_class_probs(clean_logits: torch.Tensor) -> torch.Tensor:
        p_one = torch.sigmoid(clean_logits)
        return torch.stack((1.0 - p_one, p_one), dim=-1)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _train_loss(self, x_0, label=None, x_ct=None):
        """Campbell DFM denoising objective specialized to binary states."""
        if x_ct is not None:
            raise NotImplementedError(
                "x_ct/super-resolution conditioning is not implemented for this DFM baseline."
            )

        x_1 = x_0.float()
        b, device = x_1.shape[0], x_1.device

        # Official uniform DFM: t~U(0,1), x_t is clean with probability t and
        # otherwise independently redrawn from the uniform categorical source.
        flow_t = self.sample_time(b, device)
        x_t = self._sample_conditional_flow(x_1, flow_t)
        network_time = self._network_time_from_flow_time(flow_t)

        train_label = label
        if train_label is not None and self.guidance and np.random.random() < 0.1:
            train_label = None

        raw_logits = self._call_denoiser(x_t, network_time, label=train_label)

        if self.p_flip:
            target = torch.logical_xor(x_1.bool(), x_t.bool()).to(x_1.dtype)
        else:
            target = x_1

        if self.focal >= 0:
            per_element_loss = focal_loss(
                raw_logits, target, alpha=-1, gamma=self.focal
            )
        else:
            per_element_loss = F.binary_cross_entropy_with_logits(
                raw_logits, target, reduction="none"
            )

        if not torch.isfinite(per_element_loss).all():
            raise FloatingPointError("Non-finite DFM training loss encountered.")

        # The official DFM objective is an unweighted cross-entropy. The
        # 'weighted' branch is retained only for compatibility with existing
        # experiment scripts and applies the project's previous flow-time weight.
        if self.loss_final == "mean":
            weight = 1.0
        elif self.loss_final == "weighted":
            weight = self._broadcast_batch_scalar(flow_t, per_element_loss)
        else:
            raise NotImplementedError(
                f"Unsupported loss_final={self.loss_final!r}; use 'mean' for DFM."
            )

        loss = (weight * per_element_loss).mean()
        bce_loss = per_element_loss.mean()

        with torch.no_grad():
            clean_logits = raw_logits
            if self.p_flip:
                clean_logits = x_t * (-raw_logits) + (1.0 - x_t) * raw_logits
            acc = ((clean_logits > 0.0).to(x_1.dtype) == x_1).float().mean()

        return {
            "loss": loss,
            "bce_loss": bce_loss,
            "acc": acc,
            "flow_t": flow_t.mean().detach(),
        }

    # ------------------------------------------------------------------
    # CTMC Euler sampling
    # ------------------------------------------------------------------
    def _euler_ctmc_step(
        self,
        x_t: torch.Tensor,
        x_1_probs: torch.Tensor,
        flow_t: float,
        dt: float,
        final_step: bool,
    ) -> torch.Tensor:
        """One Euler step from the authors' uniform DFM implementation."""
        if not (0.0 <= flow_t < 1.0):
            raise ValueError(f"flow_t must be in [0,1), got {flow_t}.")
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}.")

        x_index = x_t.long()
        x_1_probs_at_x_t = torch.gather(
            x_1_probs, dim=-1, index=x_index.unsqueeze(-1)
        )

        # The official implementation disables extra stochasticity on the
        # final step. eta=0 is the minimum-jump rate from Proposition 3.4.
        eta = 0.0 if final_step else self.dfm_stochasticity
        denominator = max(1.0 - flow_t, torch.finfo(x_1_probs.dtype).eps)
        coefficient = (
            1.0
            + eta
            + eta * (self.state_space_size - 1) * flow_t
        ) / denominator

        step_probs = (
            dt * coefficient * x_1_probs
            + dt * eta * x_1_probs_at_x_t
        ).clamp(min=0.0, max=1.0)

        # Off-diagonal probabilities first; set the diagonal so rows sum to 1.
        step_probs.scatter_(dim=-1, index=x_index.unsqueeze(-1), value=0.0)
        diagonal = (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
        diagonal = diagonal.to(dtype=step_probs.dtype)
        step_probs.scatter_(dim=-1, index=x_index.unsqueeze(-1), src=diagonal)

        # Numerical safeguard. For binary S=2, the official clamp already
        # yields a valid row; normalization only corrects floating-point drift.
        row_sum = step_probs.sum(dim=-1, keepdim=True)
        if torch.any(row_sum <= 0):
            current_one_hot = F.one_hot(
                x_index, num_classes=self.state_space_size
            ).to(step_probs.dtype)
            step_probs = torch.where(row_sum > 0, step_probs, current_one_hot)
            row_sum = step_probs.sum(dim=-1, keepdim=True)
        step_probs = step_probs / row_sum.clamp_min(
            torch.finfo(step_probs.dtype).eps
        )

        flat_probs = step_probs.reshape(-1, self.state_space_size)
        next_state = torch.multinomial(flat_probs, num_samples=1).reshape(x_t.shape)
        return next_state.to(dtype=x_t.dtype)

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
        """Sample with exactly ``sample_steps`` denoiser evaluations (NFEs)."""
        del full  # Interface compatibility; DFM always uses the requested grid.

        try:
            device = next(self._denoise_fn.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if sample_steps is None:
            sample_steps = self.num_timesteps
        sample_steps = int(sample_steps)
        if sample_steps <= 0:
            raise ValueError(f"sample_steps must be positive, got {sample_steps}.")

        if shape is not None:
            sample_shape: Tuple[int, ...] = tuple(shape)
            if len(sample_shape) < 2:
                raise ValueError(f"shape must include batch and data dimensions, got {shape}.")
            b = sample_shape[0]
        else:
            sample_shape = (b, int(np.prod(self.shape)), self.codebook_size)

        # Uniform binary source p_0(x)=Bernoulli(0.5).
        x_t = torch.randint(
            low=0,
            high=self.state_space_size,
            size=sample_shape,
            device=device,
            dtype=torch.long,
        ).float()

        if mask is not None:
            m = mask["mask"].unsqueeze(0).to(device=device, dtype=x_t.dtype)
            fixed_latent = mask["latent"].unsqueeze(0).to(device=device, dtype=x_t.dtype)
            x_t = fixed_latent * m + x_t * (1.0 - m)

        if self.dataset == "imagenet":
            if label is None:
                label = (torch.arange(b, device=device) * 100).long()
            elif not torch.is_tensor(label):
                label = torch.full((b,), label, device=device, dtype=torch.long)

        if return_all:
            x_all = [x_t.clone()]

        # DFM time goes from 0 (uniform source) to 1 (data). There is one
        # denoiser call per interval, so K intervals are exactly K NFEs.
        time_grid = torch.linspace(
            0.0, 1.0, steps=sample_steps + 1, device=device, dtype=torch.float32
        )

        for step_index in range(sample_steps):
            flow_t_scalar = float(time_grid[step_index].item())
            next_t_scalar = float(time_grid[step_index + 1].item())
            dt = next_t_scalar - flow_t_scalar
            final_step = step_index == sample_steps - 1

            flow_t_batch = torch.full(
                (b,), flow_t_scalar, device=device, dtype=torch.float32
            )
            network_time = self._network_time_from_flow_time(flow_t_batch)

            clean_logits = self._predict_clean_logits(
                x_t=x_t,
                network_time=network_time,
                temp=float(temp),
                label=label,
                guidance=guidance,
            )
            x_1_probs = self._binary_class_probs(clean_logits)

            x_t = self._euler_ctmc_step(
                x_t=x_t,
                x_1_probs=x_1_probs,
                flow_t=flow_t_scalar,
                dt=dt,
                final_step=final_step,
            )

            if mask is not None:
                x_t = fixed_latent * m + x_t * (1.0 - m)

            if return_all:
                x_all.append(x_t.clone())

        if return_all:
            return torch.cat(x_all, dim=0)
        return x_t

    def forward(self, x, label=None, x_t=None):
        return self._train_loss(x, label=label, x_ct=x_t)


# Optional aliases to simplify integration with existing naming conventions.
BinaryDFMDecouple = BinaryDiscreteFlowModelDecouple
BinaryDiffusionFlowDecouple = BinaryDiscreteFlowModelDecouple


def focal_loss(inputs, targets, alpha=-1, gamma=1):
    """Binary focal loss with batch-adaptive class balancing."""
    probabilities = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    modulation = (1.0 - p_t).clamp(min=1e-6, max=1.0 - 1e-6)
    loss = ce_loss * modulation.pow(gamma)

    if alpha == -1:
        reduce_dims = tuple(range(1, targets.ndim))
        positive_fraction = targets.mean(dim=reduce_dims, keepdim=True)
        alpha_t = (1.0 - positive_fraction) * targets + positive_fraction * (1.0 - targets)
        loss = alpha_t * loss
    elif alpha > 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss

    return loss
