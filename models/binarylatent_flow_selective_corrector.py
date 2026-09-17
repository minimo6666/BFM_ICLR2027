"""GT-supervised selective context-only corrector for direct-X0 BFM.

Ground truth is used only to construct training labels. Sampling observes only
X_t, endpoint outputs, and context-only cavity evidence. There is no random
refresh: a correction is made only when a conservative learned gate and the
cavity proposal both pass deterministic thresholds.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.ICLR27.diagnostics.masked_bit_pretrain_core import (
    MASK_VALUE,
    make_masked_input,
    masked_bce_with_logits,
)
from models.binarylatent_flow_expectation_consistent_retrain_tminus1 import (
    BinaryDiffusionFlowDecouple,
)


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


class BinaryDiffusionFlowSelectiveCorrector(BinaryDiffusionFlowDecouple):
    """Direct-X0 endpoint plus a conservative GT-supervised correction gate."""

    def __init__(self, H, denoise_fn, mask_id, corrector_denoise_fn):
        super().__init__(H, denoise_fn, mask_id)
        self._corrector_denoise_fn = corrector_denoise_fn

        # Gate context comes from detached endpoint features. Analytic per-bit
        # inputs expose the self-evidence decomposition without giving it GT.
        self.gate_context_head = nn.Linear(
            self._denoise_fn.n_embd, self.codebook_size, bias=True
        )
        self.gate_analytic_head = nn.Sequential(
            nn.Linear(5, 32), nn.SiLU(), nn.Linear(32, 1, bias=False)
        )

        # Initial sampler is exactly baseline: no score can pass the gate.
        nn.init.zeros_(self.gate_context_head.weight)
        nn.init.constant_(self.gate_context_head.bias, -12.0)
        # Keep a non-degenerate hidden feature map, but zero its final
        # projection so the analytic branch initially contributes exactly 0.
        nn.init.zeros_(self.gate_analytic_head[-1].weight)

        self.endpoint_weight = _env_float("BFM_SELECTIVE_ENDPOINT_WEIGHT", 1.0)
        self.cavity_weight = _env_float("BFM_SELECTIVE_CAVITY_WEIGHT", 1.0)
        self.gate_weight = _env_float("BFM_SELECTIVE_GATE_WEIGHT", 0.25)
        self.gate_threshold = _env_float("BFM_SELECTIVE_GATE_THRESHOLD", 0.95)
        self.proposal_threshold = _env_float(
            "BFM_SELECTIVE_PROPOSAL_THRESHOLD", 0.75
        )
        self.advantage_delta = _env_float(
            "BFM_SELECTIVE_ADVANTAGE_DELTA", 0.02
        )
        self.max_action_fraction = _env_float(
            "BFM_SELECTIVE_MAX_ACTION_FRACTION", 0.005
        )
        self.max_train_candidates = _env_int(
            "BFM_SELECTIVE_MAX_TRAIN_CANDIDATES", 256
        )
        self.min_train_negatives = _env_int(
            "BFM_SELECTIVE_MIN_TRAIN_NEGATIVES", 64
        )
        self.high_confidence_pgt = _env_float(
            "BFM_SELECTIVE_HIGH_CONFIDENCE_PGT", 0.10
        )
        self.corrector_enabled = os.environ.get(
            "BFM_SELECTIVE_CORRECTOR_ENABLED", "1"
        ).lower() in {"1", "true", "yes"}

        if self.p_flip:
            raise ValueError("Selective corrector requires p_flip=False")
        if float(self.focal) >= 0.0:
            raise ValueError("Selective corrector requires focal=-1")
        if float(self.aux) != 0.0:
            raise ValueError("Selective corrector requires aux=0")
        if self.loss_final != "mean":
            raise ValueError("Selective corrector requires loss_final='mean'")
        if not 0.5 < self.gate_threshold < 1.0:
            raise ValueError("gate threshold must be in (0.5,1)")
        if not 0.5 < self.proposal_threshold < 1.0:
            raise ValueError("proposal threshold must be in (0.5,1)")
        if not 0.0 < self.max_action_fraction < 1.0:
            raise ValueError("max action fraction must be in (0,1)")

        self.selective_config = {
            "endpoint_weight": self.endpoint_weight,
            "cavity_weight": self.cavity_weight,
            "gate_weight": self.gate_weight,
            "gate_threshold": self.gate_threshold,
            "proposal_threshold": self.proposal_threshold,
            "advantage_delta": self.advantage_delta,
            "max_action_fraction": self.max_action_fraction,
            "max_train_candidates": self.max_train_candidates,
            "min_train_negatives": self.min_train_negatives,
            "high_confidence_pgt": self.high_confidence_pgt,
            "corrector_enabled": self.corrector_enabled,
            "gate_teacher": "counterfactual one-step reverse advantage",
            "cavity_training_mix": "50% ordinary random, 25% repair anchors, 25% hard negatives",
            "sampling_selection": "gate threshold plus per-sample top-k cap",
            "sampling_refresh": "none; deterministic evidence-qualified correction",
            "gt_at_sampling": False,
        }

    def set_corrector_enabled(self, enabled: bool) -> None:
        self.corrector_enabled = bool(enabled)

    def _self_evidence(self, physical_time, reference):
        tau = self.interpolation_t[physical_time].to(reference.device)
        lam = torch.log((1.0 + tau) / (1.0 - tau).clamp_min(1e-6))
        return lam.view(-1, *([1] * (reference.ndim - 1)))

    def _endpoint_logits_features(self, x_t, physical_time, label=None):
        features = self._denoise_fn.forward_features(
            x_t, label=label, time_steps=physical_time - 1
        )
        return self._denoise_fn.head(features), features

    def _gate_logits(self, x_t, endpoint_logits, endpoint_features, physical_time):
        detached_logits = endpoint_logits.detach()
        state_sign = 2.0 * x_t.float() - 1.0
        lam = self._self_evidence(physical_time, detached_logits)
        context_proxy = detached_logits - state_sign * lam
        analytic = torch.stack(
            (
                detached_logits,
                detached_logits.abs(),
                state_sign,
                context_proxy,
                lam.expand_as(detached_logits),
            ),
            dim=-1,
        )
        return self.gate_context_head(
            endpoint_features.detach()
        ) + self.gate_analytic_head(analytic).squeeze(-1)

    @staticmethod
    def _balanced_gate_bce(gate_logits, positive_mask, supervision_mask=None):
        supervised = torch.ones_like(positive_mask, dtype=torch.bool) if supervision_mask is None else supervision_mask.bool()
        positive = positive_mask.bool() & supervised
        negative = (~positive_mask.bool()) & supervised
        zero = gate_logits.sum() * 0.0
        positive_loss = F.softplus(-gate_logits[positive]).mean() if positive.any() else zero
        negative_loss = F.softplus(gate_logits[negative]).mean() if negative.any() else zero
        if positive.any() and negative.any():
            return 0.5 * positive_loss + 0.5 * negative_loss
        return positive_loss + negative_loss

    def _training_candidate_mask(
        self,
        high_conf_wrong,
        state_wrong,
        context_conflict,
        endpoint_pgt,
        gate_logits,
    ):
        """Deterministic oracle anchors plus structurally difficult negatives."""
        batch, bits = high_conf_wrong.shape[0], high_conf_wrong[0].numel()
        result = torch.zeros_like(high_conf_wrong, dtype=torch.bool)
        flat_anchor = high_conf_wrong.reshape(batch, bits)
        flat_state_wrong = state_wrong.reshape(batch, bits)
        flat_conflict = context_conflict.reshape(batch, bits)
        flat_pgt = endpoint_pgt.reshape(batch, bits)
        flat_gate = gate_logits.detach().reshape(batch, bits)
        flat_result = result.reshape(batch, bits)
        for row in range(batch):
            random_quota = self.max_train_candidates // 2
            anchor_quota = (self.max_train_candidates - random_quota) // 2
            negative_quota = self.max_train_candidates - random_quota - anchor_quota
            anchor_idx = torch.nonzero(flat_anchor[row], as_tuple=False).flatten()
            if anchor_idx.numel() > anchor_quota:
                order = torch.argsort(flat_pgt[row, anchor_idx])
                anchor_idx = anchor_idx[order[:anchor_quota]]
            flat_result[row, anchor_idx] = True

            correct_idx = torch.nonzero(
                (~flat_state_wrong[row]) & (~flat_result[row]),
                as_tuple=False,
            ).flatten()
            negative_wanted = min(negative_quota, int(correct_idx.numel()))
            if negative_wanted:
                negative_score = (
                    2.0 * flat_conflict[row, correct_idx].float()
                    + torch.sigmoid(flat_gate[row, correct_idx])
                    + (1.0 - flat_pgt[row, correct_idx])
                )
                chosen_negative = correct_idx[
                    torch.topk(negative_score, negative_wanted).indices
                ]
                flat_result[row, chosen_negative] = True

            ordinary_idx = torch.nonzero(
                ~flat_result[row], as_tuple=False
            ).flatten()
            ordinary_wanted = min(random_quota, int(ordinary_idx.numel()))
            if ordinary_wanted:
                chosen_ordinary = ordinary_idx[
                    torch.randperm(
                        ordinary_idx.numel(), device=ordinary_idx.device
                    )[:ordinary_wanted]
                ]
                flat_result[row, chosen_ordinary] = True

            remaining = self.max_train_candidates - int(flat_result[row].sum())
            other_idx = torch.nonzero(~flat_result[row], as_tuple=False).flatten()
            if remaining and other_idx.numel():
                score = (
                    3.0 * flat_state_wrong[row, other_idx].float()
                    + 2.0 * flat_conflict[row, other_idx].float()
                    + torch.sigmoid(flat_gate[row, other_idx])
                    + (1.0 - flat_pgt[row, other_idx])
                )
                chosen = other_idx[
                    torch.topk(score, min(remaining, score.numel())).indices
                ]
                flat_result[row, chosen] = True

        return result

    def _select_sampling_candidates(self, gate_logits):
        score = torch.sigmoid(gate_logits)
        eligible = score >= self.gate_threshold
        batch, bits = score.shape[0], score[0].numel()
        max_actions = max(1, int(round(bits * self.max_action_fraction)))
        flat_score = score.reshape(batch, bits)
        flat_eligible = eligible.reshape(batch, bits)
        flat_selected = torch.zeros_like(flat_eligible)
        for row in range(batch):
            indices = torch.nonzero(flat_eligible[row], as_tuple=False).flatten()
            if indices.numel() > max_actions:
                indices = indices[
                    torch.topk(flat_score[row, indices], max_actions).indices
                ]
            flat_selected[row, indices] = True
        return flat_selected.reshape_as(eligible), score

    def _proposal_from_candidates(self, x_t, physical_time, candidates, label=None):
        cavity_input = make_masked_input(x_t, candidates, MASK_VALUE)
        cavity_logits = self._corrector_denoise_fn(
            cavity_input, label=label, time_steps=physical_time - 1
        )
        cavity_clean = torch.sigmoid(cavity_logits)
        tau = self.interpolation_t[physical_time].view(-1, 1, 1)
        same_level = 0.5 * (1.0 - tau) + tau * cavity_clean
        proposal = same_level >= 0.5
        confidence = torch.maximum(cavity_clean, 1.0 - cavity_clean)
        action = (
            candidates
            & (proposal != x_t.bool())
            & (confidence >= self.proposal_threshold)
        )
        return cavity_logits, same_level, proposal, action

    def _train_loss(self, x_0, label=None, x_ct=None):
        if x_ct is not None:
            raise NotImplementedError("x_ct training is not used")
        x_0 = x_0.float()
        physical_time = self.sample_time(x_0.shape[0], x_0.device)
        x_t = torch.bernoulli(self.q_sample(x_0, physical_time))

        endpoint_logits, endpoint_features = self._endpoint_logits_features(
            x_t, physical_time, label=label
        )
        endpoint_loss = F.binary_cross_entropy_with_logits(
            endpoint_logits, x_0, reduction="mean"
        )
        truth = x_0.bool()
        endpoint_probability = torch.sigmoid(endpoint_logits.detach())
        endpoint_pgt = torch.where(truth, endpoint_probability, 1.0 - endpoint_probability)
        state_wrong = x_t.bool() != truth
        endpoint_wrong = (endpoint_logits.detach() >= 0.0) != truth
        high_conf_wrong = state_wrong & endpoint_wrong & (
            endpoint_pgt < self.high_confidence_pgt
        )
        state_sign = 2.0 * x_t.float() - 1.0
        context_proxy = endpoint_logits.detach() - state_sign * self._self_evidence(
            physical_time, endpoint_logits
        )
        context_conflict = state_sign * context_proxy < 0.0

        gate_logits = self._gate_logits(
            x_t, endpoint_logits, endpoint_features, physical_time
        )
        candidate_mask = self._training_candidate_mask(
            high_conf_wrong, state_wrong, context_conflict,
            endpoint_pgt, gate_logits
        )
        cavity_logits, _, proposal, action = self._proposal_from_candidates(
            x_t, physical_time, candidate_mask, label=label
        )
        with torch.no_grad():
            proposed_state = torch.where(action, proposal.float(), x_t)
            if action.any():
                proposed_logits, _ = self._endpoint_logits_features(
                    proposed_state, physical_time, label=label
                )
            else:
                proposed_logits = endpoint_logits.detach()
            keep_clean = torch.sigmoid(endpoint_logits.detach())
            proposed_clean = torch.sigmoid(proposed_logits)
            target_time = physical_time - 1
            keep_next = self._reverse_probability(
                keep_clean, x_t, physical_time, target_time
            )
            proposed_next = self._reverse_probability(
                proposed_clean, proposed_state, physical_time, target_time
            )
            keep_gt = torch.where(truth, keep_next, 1.0 - keep_next)
            proposed_gt = torch.where(
                truth, proposed_next, 1.0 - proposed_next
            )
            final_mask = (physical_time == 1).view(-1, 1, 1)
            keep_clean_gt = torch.where(
                truth, keep_clean, 1.0 - keep_clean
            )
            proposed_clean_gt = torch.where(
                truth, proposed_clean, 1.0 - proposed_clean
            )
            keep_gt = torch.where(final_mask, keep_clean_gt, keep_gt)
            proposed_gt = torch.where(
                final_mask, proposed_clean_gt, proposed_gt
            )
            reverse_advantage = proposed_gt - keep_gt
            positive_advantage = action & (
                reverse_advantage > self.advantage_delta
            )
        gate_loss = self._balanced_gate_bce(
            gate_logits, positive_advantage, candidate_mask
        )
        cavity_loss = masked_bce_with_logits(cavity_logits, x_0, candidate_mask)
        loss = (
            self.endpoint_weight * endpoint_loss
            + self.cavity_weight * cavity_loss
            + self.gate_weight * gate_loss
        )

        with torch.no_grad():
            gate_selected, gate_probability = self._select_sampling_candidates(gate_logits)
            # Training flux is reported only where sampling and train candidate
            # sets overlap; the paired on-policy evaluator remains decisive.
            diagnostic_action = action & gate_selected
            repaired = diagnostic_action & state_wrong & (proposal == truth)
            damaged = diagnostic_action & ~state_wrong & (proposal != truth)
            selected_count = gate_selected.sum().clamp_min(1)
            positive_count = positive_advantage.sum().clamp_min(1)
            selected_true_positive = (
                gate_selected & positive_advantage
            ).sum()
            supervised_count = candidate_mask.sum().clamp_min(1)
            stats = {
                "loss": loss,
                "bce_loss": endpoint_loss,
                "endpoint_bce": endpoint_loss,
                "cavity_bce": cavity_loss,
                "gate_bce": gate_loss,
                "acc": ((endpoint_logits >= 0.0) == truth).float().mean(),
                "high_conf_wrong_fraction": high_conf_wrong.float().mean(),
                "candidate_fraction": candidate_mask.float().mean(),
                "positive_advantage_fraction": (
                    positive_advantage.sum().float()
                    / supervised_count
                ),
                "mean_counterfactual_advantage": (
                    (reverse_advantage * candidate_mask.float()).sum()
                    / supervised_count
                ),
                "gate_selected_fraction": gate_selected.float().mean(),
                "gate_precision": selected_true_positive.float() / selected_count,
                "gate_recall": selected_true_positive.float() / positive_count,
                "repair_count": repaired.sum().float(),
                "damage_count": damaged.sum().float(),
                "net_correction_flux": repaired.sum().float() - damaged.sum().float(),
                "mean_gate_probability": gate_probability.mean(),
            }
        return stats

    @torch.no_grad()
    def _pre_reverse_corrector(
        self,
        x_t,
        physical_time,
        *,
        temp: float,
        label=None,
        guidance=None,
    ):
        if not self.corrector_enabled:
            return x_t
        if guidance is not None:
            raise ValueError("Selective-corrector guidance is not implemented")
        endpoint_logits, endpoint_features = self._endpoint_logits_features(
            x_t, physical_time, label=label
        )
        gate_logits = self._gate_logits(
            x_t, endpoint_logits / temp, endpoint_features, physical_time
        )
        candidates, _ = self._select_sampling_candidates(gate_logits)
        if not candidates.any():
            return x_t
        _, _, proposal, action = self._proposal_from_candidates(
            x_t, physical_time, candidates, label=label
        )
        return torch.where(action, proposal.float(), x_t)
