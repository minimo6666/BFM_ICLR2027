"""BitDance-style binary diffusion prediction head for BFM.

This is a clean-room adaptation of the public BitDance ImageNet ``DiffHead``
architecture/objective (Apache-2.0 project) to the tensor conventions used by
this BFM repository.  The mathematical ingredients are intentionally kept
faithful to BitDance p=1:

* a 6-block conditional gated-MLP head (configurable),
* logit-normal inner flow time by default,
* Gaussian-to-binary linear interpolation,
* x-prediction converted to velocity matching loss,
* Euler-Maruyama inner sampling with a final Euler step.

The implementation deliberately does not use ``torch.compile`` so it remains
compatible with the existing BFM training environment and easier to debug.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def timestep_embedding(
    t: torch.Tensor,
    dim: int,
    max_period: int = 10000,
    time_factor: float = 1000.0,
) -> torch.Tensor:
    """Sinusoidal scalar-time embedding used by the BitDance head."""
    half = dim // 2
    t_scaled = time_factor * t.float()
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / max(half, 1)
    )
    args = t_scaled[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def time_shift_sana(
    t: torch.Tensor,
    flow_shift: float = 1.0,
    sigma: float = 1.0,
) -> torch.Tensor:
    if flow_shift == 1.0:
        return t
    # Algebraically equivalent to the upstream expression but stable at t=0/1.
    eps = torch.finfo(t.dtype).eps
    tc = t.clamp(eps, 1.0 - eps)
    shifted = (1.0 / flow_shift) / (
        (1.0 / flow_shift) + (1.0 / tc - 1.0).pow(sigma)
    )
    return torch.where(t <= 0, torch.zeros_like(shifted), torch.where(t >= 1, torch.ones_like(shifted), shifted))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(t, self.frequency_embedding_size))


class ResBlock(nn.Module):
    """BitDance gated SwiGLU residual block with AdaLN modulation."""

    def __init__(self, channels: int):
        super().__init__()
        hidden_dim = int(channels * 1.5)
        self.norm = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=True)
        self.w1 = nn.Linear(channels, hidden_dim * 2, bias=True)
        self.w2 = nn.Linear(hidden_dim, channels, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm(x) * (1.0 + scale) + shift
        h1, h2 = self.w1(h).chunk(2, dim=-1)
        h = self.w2(F.silu(h1) * h2)
        return x + h * gate


class FinalLayer(nn.Module):
    def __init__(self, channels: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            channels, eps=1e-6, elementwise_affine=False
        )
        self.ada_ln_modulation = nn.Linear(channels, channels * 2, bias=True)
        self.linear = nn.Linear(channels, out_channels, bias=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        scale, shift = self.ada_ln_modulation(y).chunk(2, dim=-1)
        x = self.norm_final(x) * (1.0 + scale) + shift
        return self.linear(x)


class MlpEncoder(nn.Module):
    """Conditional MLP used as BitDance's p=1 binary diffusion predictor."""

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        num_res_blocks: int,
        num_ada_ln_blocks: int = 2,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        if num_res_blocks < 1:
            raise ValueError("num_res_blocks must be >= 1")
        if num_ada_ln_blocks < 1 or num_ada_ln_blocks > num_res_blocks:
            raise ValueError("num_ada_ln_blocks must be in [1, num_res_blocks]")
        if num_res_blocks % num_ada_ln_blocks != 0:
            raise ValueError(
                "For the faithful shared-AdaLN layout, num_res_blocks must be "
                "divisible by num_ada_ln_blocks."
            )

        self.model_channels = int(model_channels)
        self.grad_checkpointing = bool(grad_checkpointing)
        self.time_embed = TimestepEmbedder(self.model_channels)
        self.cond_embed = nn.Linear(cond_channels, self.model_channels)
        self.input_proj = nn.Linear(in_channels, self.model_channels)
        self.res_blocks = nn.ModuleList(
            [ResBlock(self.model_channels) for _ in range(num_res_blocks)]
        )
        self.ada_ln_blocks = nn.ModuleList(
            [
                nn.Linear(self.model_channels, self.model_channels * 3, bias=True)
                for _ in range(num_ada_ln_blocks)
            ]
        )
        self.ada_ln_switch_freq = num_res_blocks // num_ada_ln_blocks
        self.final_layer = FinalLayer(self.model_channels, in_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        self.apply(basic_init)
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Faithful zero-init: every AdaLN controller and the final output start
        # at zero, so the diffusion head begins as a stable residual predictor.
        for block in self.ada_ln_blocks:
            nn.init.constant_(block.weight, 0.0)
            nn.init.constant_(block.bias, 0.0)
        nn.init.constant_(self.final_layer.ada_ln_modulation.weight, 0.0)
        nn.init.constant_(self.final_layer.ada_ln_modulation.bias, 0.0)
        nn.init.constant_(self.final_layer.linear.weight, 0.0)
        nn.init.constant_(self.final_layer.linear.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        x = self.input_proj(x)
        t_emb = self.time_embed(t)
        c_emb = self.cond_embed(cond)
        y = F.silu(t_emb + c_emb)

        scale = shift = gate = None
        for i, block in enumerate(self.res_blocks):
            if i % self.ada_ln_switch_freq == 0:
                controller = self.ada_ln_blocks[i // self.ada_ln_switch_freq]
                scale, shift, gate = controller(y).chunk(3, dim=-1)
            assert scale is not None and shift is not None and gate is not None
            if self.grad_checkpointing and self.training:
                x = checkpoint(
                    block,
                    x,
                    scale,
                    shift,
                    gate,
                    use_reentrant=False,
                )
            else:
                x = block(x, scale, shift, gate)

        return self.final_layer(x, y)


class BitDanceDiffusionHead(nn.Module):
    """Joint binary-vector predictor and its native rectified-flow objective."""

    def __init__(
        self,
        target_dim: int,
        cond_dim: int,
        hidden_dim: int = 768,
        depth: int = 6,
        adaln_depth: int = 2,
        grad_checkpointing: bool = True,
        time_shift: float = 1.0,
        time_schedule: str = "logit_normal",
        p_std: float = 0.8,
        p_mean: float = -0.8,
        last_step_size: float = 0.05,
    ):
        super().__init__()
        self.target_dim = int(target_dim)
        self.time_shift = float(time_shift)
        self.time_schedule = str(time_schedule)
        self.p_std = float(p_std)
        self.p_mean = float(p_mean)
        self.last_step_size = float(last_step_size)

        self.net = MlpEncoder(
            in_channels=self.target_dim,
            model_channels=int(hidden_dim),
            cond_channels=int(cond_dim),
            num_res_blocks=int(depth),
            num_ada_ln_blocks=int(adaln_depth),
            grad_checkpointing=bool(grad_checkpointing),
        )

    def _sample_inner_time(self, n: int, device: torch.device) -> torch.Tensor:
        if self.time_schedule == "logit_normal":
            t = (
                torch.randn(n, device=device, dtype=torch.float32) * self.p_std
                + self.p_mean
            ).sigmoid()
        elif self.time_schedule == "uniform":
            t = torch.rand(n, device=device, dtype=torch.float32)
        else:
            raise NotImplementedError(
                f"unknown BitDance time_schedule={self.time_schedule!r}"
            )
        return time_shift_sana(t, self.time_shift)

    def training_loss(
        self,
        target_signed: torch.Tensor,
        cond: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """BitDance velocity-MSE loss.

        ``target_signed`` is the clean binary vector in {-1,+1}^C.  The head is
        intentionally evaluated in float32, matching the public BitDance loss.
        """
        # The public BitDance head explicitly disables CUDA autocast for the
        # inner flow objective.  Preserve that behavior even though the outer
        # BFM Transformer is trained under AMP.
        device_type = target_signed.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            x = target_signed.float()
            c = cond.float()
            t = self._sample_inner_time(x.shape[0], x.device)
            ti = t[:, None]
            eps = torch.randn_like(x)
            z = (1.0 - ti) * eps + ti * x
            denom = (1.0 - ti).clamp_min(0.05)
            v_target = (x - z) / denom
            x_pred = self.net(z, t, c)
            v_pred = (x_pred - z) / denom
            loss = (v_target - v_pred).float().square().mean()

        with torch.no_grad():
            endpoint_mse = (x_pred.float() - x).square().mean()
            sign_acc = ((x_pred >= 0) == (x >= 0)).float().mean()
            mean_t = t.mean()

        return {
            "loss": loss,
            "rf_loss": loss.detach(),
            "inner_x_mse": endpoint_mse,
            "inner_sign_acc": sign_acc,
            "inner_t_mean": mean_t,
        }

    @staticmethod
    def _score_from_velocity(
        velocity: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        # Exact algebra used by the public BitDance Euler-Maruyama sampler for
        # alpha(t)=t and sigma(t)=1-t.
        alpha_t = t
        sigma_t = 1.0 - t
        reverse_alpha_ratio = alpha_t
        var = sigma_t.square() + alpha_t * sigma_t
        return (reverse_alpha_ratio * velocity - x) / var.clamp_min(1e-8)

    @torch.no_grad()
    def sample_continuous(
        self,
        cond: torch.Tensor,
        num_sampling_steps: int = 20,
    ) -> torch.Tensor:
        """Sample one joint continuous binary vector per conditioning row."""
        if num_sampling_steps < 1:
            raise ValueError("num_sampling_steps must be >= 1")

        c = cond.float()
        x = torch.randn(
            (c.shape[0], self.target_dim),
            device=c.device,
            dtype=torch.float32,
        )

        t_all = torch.linspace(
            0.0,
            1.0 - self.last_step_size,
            num_sampling_steps + 1,
            device=c.device,
            dtype=torch.float32,
        )
        t_all = time_shift_sana(t_all, self.time_shift)
        dts = t_all[1:] - t_all[:-1]

        for i in range(num_sampling_steps):
            t_scalar = t_all[i]
            t_batch = torch.full(
                (c.shape[0],), t_scalar, device=c.device, dtype=torch.float32
            )
            x_pred = self.net(x, t_batch, c)
            denom = (1.0 - t_batch[:, None]).clamp_min(0.05)
            velocity = (x_pred - x) / denom
            score = self._score_from_velocity(velocity, x, t_scalar)
            drift = velocity + (1.0 - t_scalar) * score
            dt = dts[i]
            noise_scale = torch.sqrt(
                torch.clamp(2.0 * (1.0 - t_scalar) * dt, min=0.0)
            )
            x = x + drift * dt + noise_scale * torch.randn_like(x)

        # Faithful final deterministic Euler step from 0.95 -> 1.0 by default.
        t_final = torch.full(
            (c.shape[0],),
            1.0 - self.last_step_size,
            device=c.device,
            dtype=torch.float32,
        )
        x_pred = self.net(x, t_final, c)
        velocity = (x_pred - x) / (1.0 - t_final[:, None]).clamp_min(0.05)
        x = x + velocity * self.last_step_size
        return x

    @torch.no_grad()
    def sample_continuous_chunked(
        self,
        cond: torch.Tensor,
        num_sampling_steps: int = 20,
        chunk_size: int = 0,
    ) -> torch.Tensor:
        """Memory-only batching; does not change the per-cell model family."""
        if chunk_size is None or int(chunk_size) <= 0 or cond.shape[0] <= int(chunk_size):
            return self.sample_continuous(cond, num_sampling_steps)
        outputs = []
        chunk_size = int(chunk_size)
        for start in range(0, cond.shape[0], chunk_size):
            outputs.append(
                self.sample_continuous(
                    cond[start : start + chunk_size], num_sampling_steps
                )
            )
        return torch.cat(outputs, dim=0)
