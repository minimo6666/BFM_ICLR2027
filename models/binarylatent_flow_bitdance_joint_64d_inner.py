"""BFM with a BitDance-style joint binary endpoint predictor.

Outer process:
    X0 --BFM Bernoulli corruption--> Xt --global Transformer--> Ht

Endpoint predictor:
    (Gaussian 64-D state, inner time, Ht[cell]) --BitDance DiffHead-->
    a *jointly sampled* 64-bit clean endpoint for that spatial cell.

Reverse transition:
    X0_joint ~ p_phi(X0 | Xt)
    Xs       ~ q_BFM(Xs | Xt, X0_joint)

The analytical BFM bridge is therefore retained.  What changes relative to the
baseline is the learned endpoint family: product Bernoulli logits are replaced
by a conditional joint binary diffusion model.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from torch import nn

from models.binarylatent_flow_expectation_consistent_retrain_tminus1 import (
    BinaryDiffusionFlowDecouple as ExpectationConsistentBFM,
)
from models.bitdance_binary_diffusion_head_64d_inner import BitDanceDiffusionHead


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else int(default)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    return float(value) if value else float(default)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return bool(default)
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean {name}={value!r}")


class BinaryDiffusionFlowBitDanceJoint(ExpectationConsistentBFM):
    """Drop-in BFM sampler using the BitDance p=1 joint prediction head."""

    def __init__(self, H, denoise_fn: nn.Module, mask_id):
        super().__init__(H, denoise_fn, mask_id)

        if not hasattr(denoise_fn, "forward_features"):
            raise TypeError(
                "BitDance joint BFM requires TransformerBD.forward_features(). "
                "Use the patched models/transformer.py included with this experiment."
            )

        # The legacy Linear(768 -> 64) classifier is intentionally not trained
        # in this variant.  Freezing it avoids DDP unused-parameter issues while
        # keeping the baseline Transformer module otherwise byte-for-byte close.
        if hasattr(denoise_fn, "head"):
            for parameter in denoise_fn.head.parameters():
                parameter.requires_grad_(False)

        cond_dim = int(H.bert_n_emb)
        target_dim = int(H.codebook_size)
        diff_dim = _env_int("BFM_BITDANCE_DIFF_DIM", cond_dim)
        diff_layers = _env_int("BFM_BITDANCE_DIFF_LAYERS", 6)
        adaln_layers = _env_int("BFM_BITDANCE_ADALN_LAYERS", 2)
        grad_checkpointing = _env_bool(
            "BFM_BITDANCE_GRAD_CHECKPOINT", True
        )
        time_schedule = os.environ.get(
            "BFM_BITDANCE_TIME_SCHEDULE", "logit_normal"
        ).strip().lower()
        time_shift = _env_float("BFM_BITDANCE_TIME_SHIFT", 1.0)
        p_std = _env_float("BFM_BITDANCE_P_STD", 0.8)
        p_mean = _env_float("BFM_BITDANCE_P_MEAN", -0.8)
        last_step_size = _env_float("BFM_BITDANCE_LAST_STEP_SIZE", 0.05)

        self.diff_batch_mul = _env_int("BFM_BITDANCE_DIFF_BATCH_MUL", 1)
        self.inner_sample_steps = _env_int("BFM_BITDANCE_SAMPLE_STEPS", 20)
        self.head_sample_chunk = _env_int(
            "BFM_BITDANCE_HEAD_SAMPLE_CHUNK", 4096
        )

        if self.diff_batch_mul < 1:
            raise ValueError("BFM_BITDANCE_DIFF_BATCH_MUL must be >= 1")
        if self.inner_sample_steps < 1:
            raise ValueError("BFM_BITDANCE_SAMPLE_STEPS must be >= 1")

        self.diff_head = BitDanceDiffusionHead(
            target_dim=target_dim,
            cond_dim=cond_dim,
            hidden_dim=diff_dim,
            depth=diff_layers,
            adaln_depth=adaln_layers,
            grad_checkpointing=grad_checkpointing,
            time_shift=time_shift,
            time_schedule=time_schedule,
            p_std=p_std,
            p_mean=p_mean,
            last_step_size=last_step_size,
        )

        # BitDance adds a learned position embedding immediately before its
        # diffusion head.  Keep that prediction-head detail while preserving the
        # baseline Transformer's existing 2-D sinusoidal input positions.
        self.diff_pos_emb = nn.Parameter(
            torch.zeros(1, int(H.block_size), cond_dim)
        )
        nn.init.normal_(self.diff_pos_emb, mean=0.0, std=0.02)

        self.bitdance_config = {
            "prediction_target": "flip" if self.p_flip else "clean",
            "target_dim": target_dim,
            "cond_dim": cond_dim,
            "diff_dim": diff_dim,
            "diff_layers": diff_layers,
            "adaln_layers": adaln_layers,
            "grad_checkpointing": grad_checkpointing,
            "time_schedule": time_schedule,
            "time_shift": time_shift,
            "p_std": p_std,
            "p_mean": p_mean,
            "diff_batch_mul": self.diff_batch_mul,
            "inner_sample_steps": self.inner_sample_steps,
            "last_step_size": last_step_size,
            "head_sample_chunk": self.head_sample_chunk,
        }

    def _global_features(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        label=None,
    ) -> torch.Tensor:
        if label is not None:
            features = self._denoise_fn.forward_features(
                idx=x_t, label=label, time_steps=t - 1
            )
        else:
            features = self._denoise_fn.forward_features(
                x_t, time_steps=t - 1
            )
        return features + self.diff_pos_emb[:, : features.shape[1], :]

    def _train_loss(self, x_0, label=None, x_ct=None):
        if x_ct is not None:
            raise NotImplementedError(
                "x_ct-conditioned training is not implemented in BitDance-joint BFM."
            )

        x_0 = x_0.float()
        b, device = x_0.shape[0], x_0.device
        t = self.sample_time(b, device)
        x_t_prob = self.q_sample(x_0, t)
        x_t_in = torch.bernoulli(x_t_prob)

        if label is not None and self.guidance and np.random.random() < 0.1:
            label = None

        features = self._global_features(x_t_in, t, label=label)
        target_signed = self._signed_head_target(x_0, x_t_in)

        target_flat = target_signed.reshape(-1, target_signed.shape[-1])
        cond_flat = features.reshape(-1, features.shape[-1])

        # Upstream BitDance uses multiple independent inner corruptions per
        # visual token (diff_batch_mul=4 in its ImageNet CLI).  We expose the
        # same mechanism, but default to 1 here because the baseline BFM batch
        # size is much larger and 4x can exceed a 24-GB GPU.  Setting the env var
        # to 4 recovers the upstream multiplicity exactly.
        if self.diff_batch_mul > 1:
            target_flat = target_flat.repeat(self.diff_batch_mul, 1)
            cond_flat = cond_flat.repeat(self.diff_batch_mul, 1)

        stats = self.diff_head.training_loss(target_flat, cond_flat)
        stats["outer_t_mean"] = t.float().mean().detach()
        stats["diff_batch_mul"] = torch.tensor(
            float(self.diff_batch_mul), device=device
        )
        return stats

    def _signed_head_target(self, x_0, x_t):
        """Convert the configured clean/flip endpoint target to signed bits."""
        target = (
            torch.logical_xor(x_0.bool(), x_t.bool()).float()
            if self.p_flip
            else x_0
        )
        return target.mul(2.0).sub(1.0)

    def _head_bits_to_clean(self, bits, x_t):
        """Map a sampled flip vector back to the clean X0 endpoint."""
        if self.p_flip:
            return torch.logical_xor(bits.bool(), x_t.bool()).float()
        return bits

    @torch.no_grad()
    def _sample_joint_endpoint(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        label=None,
    ) -> torch.Tensor:
        features = self._global_features(x_t, t, label=label)
        b, n, _ = features.shape
        cond_flat = features.reshape(-1, features.shape[-1])
        endpoint_cont = self.diff_head.sample_continuous_chunked(
            cond_flat,
            num_sampling_steps=self.inner_sample_steps,
            chunk_size=self.head_sample_chunk,
        )
        # BitDance applies sign at the binary-token boundary.  >=0 only differs
        # from torch.sign at the measure-zero exact-zero case and guarantees 0/1.
        endpoint_bits = (endpoint_cont >= 0.0).float()
        endpoint_bits = endpoint_bits.view(b, n, self.codebook_size)
        return self._head_bits_to_clean(endpoint_bits, x_t)

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
        # ``temp`` and ``guidance`` belong to the old Bernoulli-logit sampler.
        # The faithful BitDance head instead samples from a Gaussian inner flow.
        del temp, guidance, full

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
                    (b, int(np.prod(self.shape)), self.codebook_size),
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
                f"sample_steps must be in [1, {self.num_timesteps}], got {sample_steps}."
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

            x0_joint = self._sample_joint_endpoint(x_t, t, label=label)

            if int(step_value) != 1:
                next_step_value = int(sampling_steps[i + 1])
                t_target = torch.full(
                    (b,), next_step_value, device=device, dtype=torch.long
                )
                posterior_zero, posterior_one = self._endpoint_bridge_probabilities(
                    x_t=x_t,
                    t_current=t,
                    t_target=t_target,
                )
                x_target_prob = (
                    (1.0 - x0_joint) * posterior_zero
                    + x0_joint * posterior_one
                ).clamp(0.0, 1.0)
                x_next = torch.bernoulli(x_target_prob)
            else:
                # The learned head already supplies one *joint* clean endpoint
                # sample.  At t=1 -> 0 the exact BFM bridge is that endpoint.
                x_next = x0_joint

            x_t = x_next

            if mask is not None:
                x_t = latent * mask_tensor + x_t * (1.0 - mask_tensor)

            if return_all:
                x_all.append(x_t)

        if return_all:
            return torch.cat(x_all, dim=0)
        return x_t

    def forward(self, x, label=None, x_t=None):
        return self._train_loss(x, label=label, x_ct=x_t)
