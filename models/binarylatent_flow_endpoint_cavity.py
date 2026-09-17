"""Full-horizon endpoint predictor with a context-only cavity corrector.

The endpoint model learns the existing quantity

    P(X0_i = 1 | X_t),

while an independent cavity Transformer replaces queried input bits by the
explicit neutral value 0.5 and learns

    P(X0_i = 1 | X_t outside the queried mask).

At sampling time the latter is converted into a same-noise-level conditional
using the exact Bernoulli forward channel:

    P(X_t_i = 1 | context)
      = (1 - tau_t) / 2 + tau_t P(X0_i = 1 | context).

Consequently the revision decision contains no likelihood term from the bit it
is deciding.  This removes the structural self-locking term at every noisy
state, but it can favor the correct direction only when the remaining context
supports it.  The practical role is therefore prevention of basin lock-in, not
a guarantee of recovering a paired ground truth from an established basin.  No
physical-time cutoff is used.
"""

from __future__ import annotations

import copy
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from experiments.ICLR27.diagnostics.masked_bit_pretrain_core import (
    MASK_VALUE,
    make_masked_input,
    masked_bce_with_logits,
    sample_bit_mask,
)
from models.binarylatent_flow_expectation_consistent_retrain_tminus1 import (
    BinaryDiffusionFlowDecouple,
)


def _configured_float(H, attribute: str, environment: str, default: float) -> float:
    value = os.environ.get(environment)
    if value is None:
        value = getattr(H, attribute, default)
    return float(value)


def _configured_int(H, attribute: str, environment: str, default: int) -> int:
    value = os.environ.get(environment)
    if value is None:
        value = getattr(H, attribute, default)
    return int(value)


class BinaryDiffusionFlowEndpointCavity(BinaryDiffusionFlowDecouple):
    """Comparison BFM plus an independent context-only revision model.

    The endpoint and cavity Transformers do not share parameters.  Both can be
    initialized strictly from the same MASK-pretrained ``TransformerBD`` state
    dict, but cavity gradients cannot degrade the endpoint predictor.
    """

    def __init__(self, H, denoise_fn, mask_id, cavity_denoise_fn=None):
        super().__init__(H, denoise_fn, mask_id)

        if cavity_denoise_fn is None:
            cavity_denoise_fn = copy.deepcopy(denoise_fn)
        self._cavity_denoise_fn = cavity_denoise_fn

        self.cavity_weight = _configured_float(
            H, "cavity_weight", "BFM_CAVITY_WEIGHT", 1.0
        )
        self.cavity_train_mask_ratio = _configured_float(
            H,
            "cavity_train_mask_ratio",
            "BFM_CAVITY_TRAIN_MASK_RATIO",
            0.10,
        )
        self.cavity_corrector_mask_ratio = _configured_float(
            H,
            "cavity_corrector_mask_ratio",
            "BFM_CAVITY_CORRECTOR_MASK_RATIO",
            0.10,
        )
        self.cavity_corrector_sweeps = _configured_int(
            H,
            "cavity_corrector_sweeps",
            "BFM_CAVITY_CORRECTOR_SWEEPS",
            1,
        )
        self.cavity_refresh_rate = _configured_float(
            H,
            "cavity_refresh_rate",
            "BFM_CAVITY_REFRESH_RATE",
            0.25,
        )

        if self.cavity_weight < 0.0:
            raise ValueError("cavity_weight must be non-negative")
        for name, ratio in (
            ("cavity_train_mask_ratio", self.cavity_train_mask_ratio),
            ("cavity_corrector_mask_ratio", self.cavity_corrector_mask_ratio),
        ):
            if not (0.0 < ratio < 1.0):
                raise ValueError(f"{name} must be in (0,1), got {ratio}")
        if self.cavity_corrector_sweeps < 0:
            raise ValueError("cavity_corrector_sweeps must be non-negative")
        if not (0.0 <= self.cavity_refresh_rate <= 1.0):
            raise ValueError(
                "cavity_refresh_rate must be in [0,1], got "
                f"{self.cavity_refresh_rate}"
            )

        # One random, persistent partition is rotated through the whole reverse
        # chain.  With rho=0.10 this creates ten groups, so one-sweep sampling
        # revisits every bit once per ten reached noise levels instead of
        # repeatedly drawing overlapping Bernoulli masks.
        self.cavity_corrector_groups = max(
            2, int(round(1.0 / self.cavity_corrector_mask_ratio))
        )
        self._cavity_group_assignment = None
        self._cavity_group_cursor = 0

        # The clean-logit interpretation is essential for the cavity target.
        if self.p_flip:
            raise ValueError("Endpoint+cavity training requires p_flip=False")
        if float(self.focal) >= 0.0:
            raise ValueError("Endpoint+cavity training requires focal=-1")
        if float(self.aux) != 0.0:
            raise ValueError("Endpoint+cavity training requires aux=0")
        if self.loss_final != "mean":
            raise ValueError("Endpoint+cavity training requires loss_final='mean'")

        self.cavity_config = {
            "weight": self.cavity_weight,
            "train_mask_ratio": self.cavity_train_mask_ratio,
            "corrector_mask_ratio": self.cavity_corrector_mask_ratio,
            "corrector_sweeps": self.cavity_corrector_sweeps,
            "corrector_groups": self.cavity_corrector_groups,
            "corrector_group_fraction": 1.0 / self.cavity_corrector_groups,
            "refresh_rate": self.cavity_refresh_rate,
            "mask_value": MASK_VALUE,
            "physical_times": f"1..{self.num_timesteps}",
            "corrector_position": "before_predictor",
            "kernel_claim": "lazy_parallel_pseudo_gibbs",
            "parameter_sharing": False,
            "endpoint_frozen": not any(
                parameter.requires_grad
                for parameter in self._denoise_fn.parameters()
            ),
        }
        self.endpoint_frozen = bool(self.cavity_config["endpoint_frozen"])

    def train(self, mode: bool = True):
        super().train(mode)
        if self.endpoint_frozen:
            self._denoise_fn.eval()
        return self

    @staticmethod
    def _ensure_one_masked_bit_per_sample(bit_mask: torch.Tensor) -> torch.Tensor:
        """Remove the negligible empty-mask corner case for small smoke tests."""
        flat = bit_mask.reshape(bit_mask.shape[0], -1)
        empty_rows = ~flat.any(dim=1)
        if empty_rows.any():
            rows = torch.nonzero(empty_rows, as_tuple=False).squeeze(1)
            columns = torch.randint(
                flat.shape[1], (rows.numel(),), device=bit_mask.device
            )
            flat[rows, columns] = True
        return bit_mask

    def _sample_cavity_mask(
        self,
        reference: torch.Tensor,
        ratio: float,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        bit_mask = sample_bit_mask(reference, ratio, generator=generator)
        return self._ensure_one_masked_bit_per_sample(bit_mask)

    def _denoise_pair(
        self,
        endpoint_input: torch.Tensor,
        cavity_input: torch.Tensor,
        network_time: torch.Tensor,
        label=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the parameter-independent endpoint and cavity models."""
        effective_label = label
        if label is not None:
            if self.training and self.guidance and np.random.random() < 0.1:
                effective_label = None

        if effective_label is None:
            endpoint_logits = self._denoise_fn(
                endpoint_input, time_steps=network_time
            )
            cavity_logits = self._cavity_denoise_fn(
                cavity_input, time_steps=network_time
            )
        else:
            endpoint_logits = self._denoise_fn(
                endpoint_input,
                label=effective_label,
                time_steps=network_time,
            )
            cavity_logits = self._cavity_denoise_fn(
                cavity_input,
                label=effective_label,
                time_steps=network_time,
            )
        return endpoint_logits, cavity_logits

    def _train_loss(self, x_0, label=None, x_ct=None):
        if x_ct is not None:
            raise NotImplementedError(
                "x_ct-conditioned training is not implemented for endpoint+cavity."
            )

        x_0 = x_0.float()
        batch_size, device = x_0.shape[0], x_0.device
        physical_time = self.sample_time(batch_size, device)
        x_t = torch.bernoulli(self.q_sample(x_0, physical_time))

        cavity_mask = self._sample_cavity_mask(
            x_t, self.cavity_train_mask_ratio
        )
        cavity_input = make_masked_input(
            x_t, cavity_mask, mask_value=MASK_VALUE
        )

        endpoint_logits = None
        if self.endpoint_frozen:
            # The endpoint checkpoint is already the controlled FID baseline.
            # Do not spend a forward pass—or inject dropout noise—merely to
            # recompute a loss that cannot update it.
            effective_label = label
            if (
                label is not None
                and self.training
                and self.guidance
                and np.random.random() < 0.1
            ):
                effective_label = None
            if effective_label is None:
                cavity_logits = self._cavity_denoise_fn(
                    cavity_input, time_steps=physical_time - 1
                )
            else:
                cavity_logits = self._cavity_denoise_fn(
                    cavity_input,
                    label=effective_label,
                    time_steps=physical_time - 1,
                )
            endpoint_loss = torch.zeros((), device=device)
        else:
            endpoint_logits, cavity_logits = self._denoise_pair(
                endpoint_input=x_t,
                cavity_input=cavity_input,
                network_time=physical_time - 1,
                label=label,
            )
            endpoint_loss = F.binary_cross_entropy_with_logits(
                endpoint_logits, x_0, reduction="mean"
            )

        cavity_loss = masked_bce_with_logits(
            cavity_logits, x_0, cavity_mask
        )
        loss = endpoint_loss + self.cavity_weight * cavity_loss

        with torch.no_grad():
            masked_count = cavity_mask.sum().clamp_min(1)
            cavity_accuracy = (
                ((cavity_logits >= 0.0) == x_0.bool())
                & cavity_mask
            ).float().sum() / masked_count

            cavity_clean_prob = torch.sigmoid(cavity_logits)
            cavity_clean_probability_of_truth = torch.where(
                x_0.bool(), cavity_clean_prob, 1.0 - cavity_clean_prob
            )
            tau = self.interpolation_t[physical_time].view(
                -1, *([1] * (x_0.ndim - 1))
            )
            context_conditional = (
                0.5 * (1.0 - tau) + tau * cavity_clean_prob
            )
            correct_direction_probability = torch.where(
                x_0.bool(), context_conditional, 1.0 - context_conditional
            )
            wrong_observation = (x_t != x_0) & cavity_mask
            wrong_weights = wrong_observation.float()
            wrong_count = wrong_weights.sum().clamp_min(1.0)
            clean_gt_probability_on_wrong_input = (
                cavity_clean_probability_of_truth * wrong_weights
            ).sum() / wrong_count
            same_level_gt_probability_on_wrong_input = (
                correct_direction_probability * wrong_weights
            ).sum() / wrong_count

        stats = {
            "loss": loss,
            "bce_loss": cavity_loss if self.endpoint_frozen else endpoint_loss,
            "cavity_bce": cavity_loss,
            "acc": (
                cavity_accuracy
                if self.endpoint_frozen
                else torch.zeros((), device=device)
            ),
            "cavity_acc": cavity_accuracy,
            "cavity_mask_fraction": cavity_mask.float().mean(),
            "cavity_wrong_input_fraction": wrong_observation.float().mean(),
            "cavity_clean_gt_probability_on_wrong_input": (
                clean_gt_probability_on_wrong_input
            ),
            "cavity_same_level_gt_probability_on_wrong_input": (
                same_level_gt_probability_on_wrong_input
            ),
        }
        if endpoint_logits is not None:
            with torch.no_grad():
                endpoint_accuracy = (
                    (endpoint_logits >= 0.0) == x_0.bool()
                ).float().mean()
            stats["endpoint_bce"] = endpoint_loss
            stats["endpoint_acc"] = endpoint_accuracy
            stats["acc"] = endpoint_accuracy
        return stats

    def _sampling_clean_logits(
        self,
        x: torch.Tensor,
        network_time: torch.Tensor,
        *,
        temp: float,
        label=None,
        guidance=None,
    ) -> torch.Tensor:
        if temp <= 0.0:
            raise ValueError(f"temp must be positive, got {temp}")

        conditional_dataset = (
            self.dataset.startswith("imagenet")
            or self.dataset.startswith("laion")
            or self.dataset.startswith("ising")
        )
        if conditional_dataset:
            logits = self._cavity_denoise_fn(
                x, time_steps=network_time, label=label
            ) / temp
            if guidance is not None:
                unconditional = self._cavity_denoise_fn(
                    x, time_steps=network_time, label=None
                ) / temp
                logits = (1.0 + guidance) * logits - guidance * unconditional
            return logits

        if guidance is not None:
            raise ValueError(
                "classifier-free guidance was requested for an unconditional dataset"
            )
        return self._cavity_denoise_fn(x, time_steps=network_time) / temp

    @torch.no_grad()
    def sample(self, *args, **kwargs):
        """Reset the random rotating partition for each generated batch."""
        self._cavity_group_assignment = None
        self._cavity_group_cursor = 0
        try:
            return super().sample(*args, **kwargs)
        finally:
            # Never retain a full latent-shaped tensor between sampling calls
            # or inside a checkpoint/deepcopy.
            self._cavity_group_assignment = None
            self._cavity_group_cursor = 0

    def _corrector_partition_mask(self, reference: torch.Tensor) -> torch.Tensor:
        assignment = self._cavity_group_assignment
        if (
            assignment is None
            or assignment.shape != reference.shape
            or assignment.device != reference.device
        ):
            assignment = torch.randint(
                self.cavity_corrector_groups,
                reference.shape,
                device=reference.device,
            )
            self._cavity_group_assignment = assignment

        group = self._cavity_group_cursor % self.cavity_corrector_groups
        bit_mask = assignment == group

        # Empty groups are practically impossible for the real 16K-bit latent,
        # but keep tiny tests and unusual shapes well-defined.
        flat_mask = bit_mask.reshape(bit_mask.shape[0], -1)
        empty_rows = ~flat_mask.any(dim=1)
        if empty_rows.any():
            flat_assignment = assignment.reshape(assignment.shape[0], -1)
            rows = torch.nonzero(empty_rows, as_tuple=False).squeeze(1)
            columns = torch.randint(
                flat_mask.shape[1], (rows.numel(),), device=reference.device
            )
            flat_assignment[rows, columns] = group
            flat_mask[rows, columns] = True

        self._cavity_group_cursor += 1
        return bit_mask

    @torch.no_grad()
    def _pre_reverse_corrector(
        self,
        x_t: torch.Tensor,
        physical_time: torch.Tensor,
        *,
        temp: float,
        label=None,
        guidance=None,
    ) -> torch.Tensor:
        """Run a lazy parallel pseudo-Gibbs update before the predictor.

        With a singleton mask and an exact cavity predictor this is the exact
        lazy Gibbs conditional and leaves ``p_t`` invariant.  The practical
        sparse block update factorizes correlated masked bits, so it is
        explicitly an approximation; the constant refresh probability limits
        its perturbation without introducing a timestep-specific gate.
        """
        if (
            self.cavity_corrector_sweeps == 0
            or self.cavity_refresh_rate == 0.0
        ):
            # Exact baseline ablation: consume neither denoiser compute nor RNG.
            return x_t

        corrected = x_t
        for _ in range(self.cavity_corrector_sweeps):
            bit_mask = self._corrector_partition_mask(corrected)
            cavity_input = make_masked_input(
                corrected, bit_mask, mask_value=MASK_VALUE
            )
            cavity_logits = self._sampling_clean_logits(
                cavity_input,
                physical_time - 1,
                temp=temp,
                label=label,
                guidance=guidance,
            )
            cavity_clean_prob = torch.sigmoid(cavity_logits)

            tau = self.interpolation_t[physical_time].view(
                -1, *([1] * (corrected.ndim - 1))
            )
            same_level_conditional = (
                0.5 * (1.0 - tau) + tau * cavity_clean_prob
            ).clamp(0.0, 1.0)
            proposed = torch.bernoulli(same_level_conditional)
            # A state-independent lazy mixture is analytically clean: for an
            # exact kernel C, (1-eta)I + eta C has the same invariant law.  For
            # our approximate block kernel, eta also linearly attenuates its
            # one-step distributional perturbation.
            refresh_shape = (corrected.shape[0],) + (1,) * (
                corrected.ndim - 1
            )
            refresh = torch.rand(
                refresh_shape,
                device=corrected.device,
                dtype=torch.float32,
            ) < self.cavity_refresh_rate
            corrected = torch.where(bit_mask & refresh, proposed, corrected)

        return corrected
