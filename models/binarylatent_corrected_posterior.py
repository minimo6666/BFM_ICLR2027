import numpy as np
import torch

from models.binarylatent import BinaryDiffusion


class BinaryDiffusionCorrectedPosterior(BinaryDiffusion):
    """
    Official BLD with ONLY the intermediate reverse-posterior construction changed.

    Network semantics are unchanged:
        m_theta = P_theta(X0=1 | Xt)
    and official BLD timestep conditioning remains:
        time_steps = t - 1.

    For a reverse jump t -> s (s < t), define the exact one-bit endpoint bridges:
        B0 = P(Xs=1 | Xt, X0=0)
        B1 = P(Xs=1 | Xt, X0=1)

    Then the corrected one-bit reverse marginal is:
        P(Xs=1 | Xt) = (1-m_theta) B0 + m_theta B1.

    Notes
    -----
    1. This is exact for the ONE-BIT marginal implied by m_theta and the BLD
       bitwise forward process. It does not claim to recover a full joint
       posterior over all latent bits.
    2. By default sample() preserves official BLD's final hard projection at
       t=1 -> 0 so that a sampler-only ablation changes only the bridge formula.
       Set final_mode="bernoulli" only for a separate fully probabilistic
       final-step experiment.
    """

    @torch.no_grad()
    def _cross_step_likelihood(self, x_t, t_current, t_target):
        """
        Same repeated scheduler.one_step composition as official BLD.

        Output ordering:
            [..., 0] = q(X_t=x_t | X_s=1)
            [..., 1] = q(X_t=x_t | X_s=0)
        """
        if t_current.ndim != 1 or t_target.ndim != 1:
            raise ValueError("t_current and t_target must be batch vectors.")
        if not torch.all(t_current == t_current[0]):
            raise ValueError("This sampler loop expects one shared current t per batch.")
        if not torch.all(t_target == t_target[0]):
            raise ValueError("This sampler loop expects one shared target s per batch.")

        gap = int(t_current[0].item() - t_target[0].item())
        if gap <= 0:
            raise ValueError(
                f"Reverse target must be cleaner: current={t_current[0].item()}, "
                f"target={t_target[0].item()}."
            )

        # Official BLD state ordering: first slot is bit=1, second is bit=0.
        likelihood = torch.stack((x_t, 1.0 - x_t), dim=-1)

        for mns in range(gap):
            likelihood = self.scheduler.one_step(
                likelihood,
                t_current - mns,
            )

        return likelihood

    @torch.no_grad()
    def _endpoint_bridge_probabilities(
        self,
        x_t,
        t_current,
        t_target,
        eps=1e-12,
    ):
        """
        Exact one-bit endpoint-conditioned bridges:
            B0 = P(X_s=1 | X_t=x_t, X0=0)
            B1 = P(X_s=1 | X_t=x_t, X0=1)
        """
        likelihood = self._cross_step_likelihood(
            x_t=x_t,
            t_current=t_current,
            t_target=t_target,
        )

        x0_zero = torch.zeros_like(x_t)
        x0_one = torch.ones_like(x_t)

        # Distribution ordering [P(bit=1), P(bit=0)].
        x0_zero_dist = torch.stack((x0_zero, 1.0 - x0_zero), dim=-1)
        x0_one_dist = torch.stack((x0_one, 1.0 - x0_one), dim=-1)

        # q(X_s | X0=z)
        prior_zero = self.scheduler(x0_zero_dist, t_target)
        prior_one = self.scheduler(x0_one_dist, t_target)

        # Bayes bridge:
        # q(X_s | X_t, X0=z) ∝ q(X_s | X0=z) q(X_t | X_s)
        post_zero = prior_zero * likelihood
        post_one = prior_one * likelihood

        post_zero = post_zero / post_zero.sum(
            dim=-1, keepdim=True
        ).clamp_min(eps)
        post_one = post_one / post_one.sum(
            dim=-1, keepdim=True
        ).clamp_min(eps)

        return post_zero[..., 0], post_one[..., 0]

    @torch.no_grad()
    def _corrected_reverse_probability(
        self,
        clean_prob,
        x_t,
        t_current,
        t_target,
    ):
        """
        Correct one-bit marginalization over X0 in {0,1}.
        """
        B0, B1 = self._endpoint_bridge_probabilities(
            x_t=x_t,
            t_current=t_current,
            t_target=t_target,
        )
        return ((1.0 - clean_prob) * B0 + clean_prob * B1).clamp(0.0, 1.0)

    @torch.no_grad()
    def sample(
        self,
        temp=1.0,
        sample_steps=None,
        b=5,
        return_all=False,
        label=None,
        mask=None,
        guidance=None,
        full=False,
        final_mode="hard",
    ):
        """
        Same official BLD sampling loop except the reverse-posterior formula.

        final_mode:
          "hard"      -> preserve official BLD final MAP threshold (strict ablation)
          "bernoulli" -> sample X0 ~ Bernoulli(m_theta) at the final step
        """
        if final_mode not in {"hard", "bernoulli"}:
            raise ValueError(f"Unknown final_mode={final_mode!r}")

        device = next(self._denoise_fn.parameters()).device

        x_t = torch.bernoulli(
            torch.full(
                (b, np.prod(self.shape), self.codebook_size),
                0.5,
                device=device,
                dtype=torch.float32,
            )
        )

        if mask is not None:
            m = mask["mask"].unsqueeze(0).to(device)
            latent = mask["latent"].unsqueeze(0).to(device)
            x_t = latent * m + x_t * (1.0 - m)

        if sample_steps is None:
            sample_steps = self.num_timesteps
        sample_steps = int(sample_steps)
        if sample_steps < 1 or sample_steps > self.num_timesteps:
            raise ValueError(
                f"sample_steps must be in [1,{self.num_timesteps}], got {sample_steps}"
            )

        sampling_steps = np.array(range(1, self.num_timesteps + 1))
        if sample_steps != self.num_timesteps:
            idx = np.linspace(0.0, 1.0, sample_steps)
            idx = np.array(idx * (self.num_timesteps - 1), int)
            sampling_steps = sampling_steps[idx]

        if return_all:
            x_all = [x_t]

        if self.dataset == "imagenet":
            if label is None:
                label = torch.arange(b, device=device) * 100
                label = label.long()
            else:
                label = torch.full(
                    (b,), label, device=device, dtype=torch.long
                )

        sampling_steps = sampling_steps[::-1]

        for i, step_value in enumerate(sampling_steps):
            t = torch.full(
                (b,),
                int(step_value),
                device=device,
                dtype=torch.long,
            )

            # EXACT official BLD time-conditioning convention.
            if (
                self.dataset.startswith("imagenet")
                or self.dataset.startswith("laion")
            ):
                raw_logits = self._denoise_fn(
                    x_t, label, time_steps=t - 1
                )
                raw_logits = raw_logits / temp

                if guidance is not None:
                    raw_logits_uncond = self._denoise_fn(
                        x_t, None, time_steps=t - 1
                    )
                    raw_logits_uncond = raw_logits_uncond / temp
                    raw_logits = (
                        (1.0 + guidance) * raw_logits
                        - guidance * raw_logits_uncond
                    )
            else:
                raw_logits = self._denoise_fn(
                    x_t, time_steps=t - 1
                )
                raw_logits = raw_logits / temp

            clean_prob = torch.sigmoid(raw_logits)

            # EXACT official BLD p_flip -> clean-X0 posterior conversion.
            if self.p_flip:
                clean_prob = (
                    x_t * (1.0 - clean_prob)
                    + (1.0 - x_t) * clean_prob
                )

            if int(step_value) != 1:
                t_target = torch.full(
                    (b,),
                    int(sampling_steps[i + 1]),
                    device=device,
                    dtype=torch.long,
                )

                x_target_prob = self._corrected_reverse_probability(
                    clean_prob=clean_prob,
                    x_t=x_t,
                    t_current=t,
                    t_target=t_target,
                )
                x_next = torch.bernoulli(x_target_prob)
            else:
                if final_mode == "hard":
                    # Strict single-variable ablation: retain official BLD behavior.
                    x_next = (clean_prob > 0.5).float()
                else:
                    # Fully probabilistic one-bit final marginal.
                    x_next = torch.bernoulli(clean_prob)

            x_t = x_next

            if mask is not None:
                x_t = latent * m + x_t * (1.0 - m)

            if return_all:
                x_all.append(x_t)

        if return_all:
            return torch.cat(x_all, 0)
        return x_t
