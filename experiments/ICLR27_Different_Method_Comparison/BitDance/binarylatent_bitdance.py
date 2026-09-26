"""Controlled-backbone BitDance adaptation for 16x16x64 binary BAE latents.

This module ports the *generation dynamics* of BitDance (Ai et al., 2026,
"Scaling Autoregressive Generative Models with Binary Tokens") to the binary
latent interface used by this project.

The controlled comparison intentionally keeps the existing TransformerBD
modules (tok_emb, 2-D position embedding, 24 Transformer blocks, final norm)
and the same frozen BAE / packed [256,64] bits.  BitDance-specific changes are:

  * autoregressive next-patch factorization over the 16x16 spatial grid;
  * block-causal execution of the existing TransformerBD blocks;
  * the official conditional continuous binary flow/diffusion head;
  * hard sign projection back to {-1,+1}, then {0,1}, after inner sampling.

Default ``parallel_num=4`` is the BitDance-B-4x analogue: each outer step
predicts one 2x2 group (4 spatial tokens) jointly, giving exactly 64 expensive
global-Transformer steps for a 16x16 latent.  The inner head uses 100 sampling
steps by default, matching the released ImageNet evaluation code.

Important terminology: this is *not* a Bernoulli-state discrete diffusion.
The clean token is binary, but the BitDance head follows a continuous Gaussian
path z_t=(1-t)e+t*x in R^D and is discretized by sign only after sampling.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Official BitDance inner continuous-flow utilities
# -----------------------------------------------------------------------------

def timestep_embedding(
    t: torch.Tensor,
    dim: int,
    max_period: int = 10000,
    time_factor: float = 1000.0,
) -> torch.Tensor:
    half = dim // 2
    t = time_factor * t.float()
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / max(half, 1)
    )
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb.to(dtype=t.dtype)


def time_shift_sana(t: torch.Tensor, flow_shift: float = 1.0, sigma: float = 1.0):
    if flow_shift == 1.0:
        return t
    eps = torch.finfo(t.dtype).eps
    t_safe = t.clamp(min=eps, max=1.0 - eps)
    return (1.0 / flow_shift) / (
        (1.0 / flow_shift) + (1.0 / t_safe - 1.0) ** sigma
    )


def _get_score_from_velocity(velocity: torch.Tensor, x: torch.Tensor, t: torch.Tensor):
    """Same velocity->score conversion used by the released BitDance sampler."""
    alpha_t = t
    d_alpha_t = torch.ones_like(t)
    sigma_t = 1.0 - t
    d_sigma_t = -torch.ones_like(t)
    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    return (reverse_alpha_ratio * velocity - x) / var.clamp_min(1e-8)


@torch.no_grad()
def _euler_maruyama(
    input_dim: int,
    forward_fn,
    cond: torch.Tensor,
    num_sampling_steps: int = 100,
    last_step_size: float = 0.05,
    time_shift: float = 1.0,
) -> torch.Tensor:
    """Unconditional-CFG specialization of BitDance's released EM sampler."""
    if num_sampling_steps <= 0:
        raise ValueError(f"num_sampling_steps must be positive, got {num_sampling_steps}")
    if not (0.0 < last_step_size < 1.0):
        raise ValueError(f"last_step_size must be in (0,1), got {last_step_size}")

    x_shape = list(cond.shape)
    x_shape[-1] = int(input_dim)
    x = torch.randn(x_shape, device=cond.device, dtype=torch.float32)

    t_all = torch.linspace(
        0.0,
        1.0 - last_step_size,
        num_sampling_steps + 1,
        device=cond.device,
        dtype=torch.float32,
    )
    t_all = time_shift_sana(t_all, time_shift)
    dts = t_all[1:] - t_all[:-1]

    batch = cond.shape[0]
    t_batch = torch.zeros(batch, device=cond.device, dtype=torch.float32)

    for i in range(num_sampling_steps):
        t_scalar = t_all[i]
        t_batch.fill_(float(t_scalar))
        output = forward_fn(x.to(dtype=cond.dtype), t_batch, cond).float()

        view_shape = (batch,) + (1,) * (x.ndim - 1)
        t_view = t_batch.view(view_shape)
        velocity = (output - x) / (1.0 - t_view).clamp_min(0.05)
        score = _get_score_from_velocity(velocity, x, t_view)
        drift = velocity + (1.0 - t_view) * score

        dt = dts[i]
        noise_scale = (2.0 * (1.0 - t_view) * dt).clamp_min(0.0).sqrt()
        x = x + drift * dt + noise_scale * torch.randn_like(x)

    # Released BitDance code finishes with one deterministic Euler step.
    t_batch.fill_(1.0 - last_step_size)
    output = forward_fn(x.to(dtype=cond.dtype), t_batch, cond).float()
    view_shape = (batch,) + (1,) * (x.ndim - 1)
    t_view = t_batch.view(view_shape)
    velocity = (output - x) / (1.0 - t_view).clamp_min(0.05)
    x = x + velocity * last_step_size
    return x


class _TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(t, self.frequency_embedding_size))


class _FinalLayer(nn.Module):
    def __init__(self, channels: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=False)
        self.ada_ln_modulation = nn.Linear(channels, channels * 2, bias=True)
        self.linear = nn.Linear(channels, out_channels, bias=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        scale, shift = self.ada_ln_modulation(y).chunk(2, dim=-1)
        x = self.norm_final(x) * (1.0 + scale) + shift
        return self.linear(x)


class _MlpResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=True)
        hidden_dim = int(channels * 1.5)
        self.w1 = nn.Linear(channels, hidden_dim * 2, bias=True)
        self.w2 = nn.Linear(hidden_dim, channels, bias=True)

    def forward(self, x, scale, shift, gate):
        h = self.norm(x) * (1.0 + scale) + shift
        h1, h2 = self.w1(h).chunk(2, dim=-1)
        h = self.w2(F.silu(h1) * h2)
        return x + h * gate


class _SingleTokenDiffNet(nn.Module):
    """Released BitDance 1x MLP diffusion head."""

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        num_res_blocks: int,
        num_ada_ln_blocks: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.time_embed = _TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(cond_channels, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)
        self.res_blocks = nn.ModuleList(
            [_MlpResBlock(model_channels) for _ in range(num_res_blocks)]
        )
        self.ada_ln_blocks = nn.ModuleList(
            [nn.Linear(model_channels, model_channels * 3, bias=True)
             for _ in range(num_ada_ln_blocks)]
        )
        self.ada_ln_switch_freq = max(1, num_res_blocks // num_ada_ln_blocks)
        if num_res_blocks % self.ada_ln_switch_freq != 0:
            raise ValueError("num_res_blocks must be divisible by AdaLN switch frequency")
        self.final_layer = _FinalLayer(model_channels, in_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(basic_init)
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        for block in self.ada_ln_blocks:
            nn.init.constant_(block.weight, 0)
            nn.init.constant_(block.bias, 0)
        nn.init.constant_(self.final_layer.ada_ln_modulation.weight, 0)
        nn.init.constant_(self.final_layer.ada_ln_modulation.bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
        # Accept [N,1,D]/[N,1,C] for wrapper convenience.
        squeeze = x.ndim == 3
        if squeeze:
            if x.shape[1] != 1 or c.shape[1] != 1:
                raise ValueError("Single-token head expects group size 1")
            x = x[:, 0]
            c = c[:, 0]

        x = self.input_proj(x)
        t_emb = self.time_embed(t)
        c_emb = self.cond_embed(c)
        y = F.silu(t_emb + c_emb)

        scale, shift, gate = self.ada_ln_blocks[0](y).chunk(3, dim=-1)
        for i, block in enumerate(self.res_blocks):
            if i > 0 and i % self.ada_ln_switch_freq == 0:
                scale, shift, gate = self.ada_ln_blocks[
                    i // self.ada_ln_switch_freq
                ](y).chunk(3, dim=-1)
            x = block(x, scale, shift, gate)

        out = self.final_layer(x, y)
        return out[:, None] if squeeze else out


class _TinyAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim % 64 != 0:
            raise ValueError(f"BitDance parallel head expects diff_dim divisible by 64, got {dim}")
        self.dim = dim
        self.n_head = dim // 64
        self.head_dim = dim // self.n_head
        self.scale = self.head_dim ** -0.5
        self.wqkv = nn.Linear(dim, dim * 3, bias=True)
        self.wo = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor):
        b, n, _ = x.shape
        q, k, v = self.wqkv(x).chunk(3, dim=-1)
        q = q.view(b, n, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.n_head, self.head_dim).transpose(1, 2)
        q = q * self.scale
        attn = torch.softmax(q @ k.transpose(-1, -2), dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(b, n, self.dim)
        return self.wo(out)


class _ParallelResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=True)
        self.attn = _TinyAttention(channels)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=True)
        hidden_dim = int(channels * 1.5)
        self.w1 = nn.Linear(channels, hidden_dim * 2, bias=True)
        self.w2 = nn.Linear(hidden_dim, channels, bias=True)

    def forward(self, x, scale1, shift1, gate1, scale2, shift2, gate2):
        h = self.norm1(x) * (1.0 + scale1) + shift1
        x = x + self.attn(h) * gate1
        h = self.norm2(x) * (1.0 + scale2) + shift2
        h1, h2 = self.w1(h).chunk(2, dim=-1)
        x = x + self.w2(F.silu(h1) * h2) * gate2
        return x


class _ParallelDiffNet(nn.Module):
    """Released BitDance parallel (4x/16x) diffusion head."""

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        num_res_blocks: int,
        num_ada_ln_blocks: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.time_embed = _TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(cond_channels, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)
        self.res_blocks = nn.ModuleList(
            [_ParallelResBlock(model_channels) for _ in range(num_res_blocks)]
        )
        self.ada_ln_blocks = nn.ModuleList(
            [nn.Linear(model_channels, model_channels * 6, bias=True)
             for _ in range(num_ada_ln_blocks)]
        )
        self.ada_ln_switch_freq = max(1, num_res_blocks // num_ada_ln_blocks)
        if num_res_blocks % self.ada_ln_switch_freq != 0:
            raise ValueError("num_res_blocks must be divisible by AdaLN switch frequency")
        self.final_layer = _FinalLayer(model_channels, in_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(basic_init)
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        for block in self.ada_ln_blocks:
            nn.init.constant_(block.weight, 0)
            nn.init.constant_(block.bias, 0)
        nn.init.constant_(self.final_layer.ada_ln_modulation.weight, 0)
        nn.init.constant_(self.final_layer.ada_ln_modulation.bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
        x = self.input_proj(x)
        t_emb = self.time_embed(t).unsqueeze(1)
        c_emb = self.cond_embed(c)
        y = F.silu(t_emb + c_emb)

        mods = self.ada_ln_blocks[0](y).chunk(6, dim=-1)
        for i, block in enumerate(self.res_blocks):
            if i > 0 and i % self.ada_ln_switch_freq == 0:
                mods = self.ada_ln_blocks[i // self.ada_ln_switch_freq](y).chunk(6, dim=-1)
            x = block(x, *mods)
        return self.final_layer(x, y)


class BitDanceDiffHead(nn.Module):
    """Conditional continuous binary flow head used by BitDance."""

    def __init__(
        self,
        bit_dim: int,
        cond_dim: int,
        hidden_dim: int,
        depth: int,
        adaln_blocks: int,
        parallel_num: int,
        time_shift: float = 1.0,
        time_schedule: str = "logit_normal",
        p_std: float = 0.8,
        p_mean: float = -0.8,
    ):
        super().__init__()
        self.bit_dim = int(bit_dim)
        self.parallel_num = int(parallel_num)
        self.time_shift = float(time_shift)
        self.time_schedule = str(time_schedule)
        self.p_std = float(p_std)
        self.p_mean = float(p_mean)

        net_cls = _SingleTokenDiffNet if self.parallel_num == 1 else _ParallelDiffNet
        self.net = net_cls(
            in_channels=self.bit_dim,
            model_channels=int(hidden_dim),
            cond_channels=int(cond_dim),
            num_res_blocks=int(depth),
            num_ada_ln_blocks=int(adaln_blocks),
        )

    def _sample_t(self, n: int, device: torch.device) -> torch.Tensor:
        if self.time_schedule == "logit_normal":
            t = (torch.randn(n, device=device) * self.p_std + self.p_mean).sigmoid()
        elif self.time_schedule == "uniform":
            t = torch.rand(n, device=device)
        else:
            raise NotImplementedError(f"unknown BitDance time_schedule={self.time_schedule!r}")
        return time_shift_sana(t, self.time_shift)

    def forward(self, target_pm1: torch.Tensor, cond: torch.Tensor) -> Dict[str, torch.Tensor]:
        if target_pm1.shape[:-1] != cond.shape[:-1]:
            raise ValueError(
                f"target/context prefix mismatch: {target_pm1.shape} vs {cond.shape}"
            )
        n = target_pm1.shape[0]
        t = self._sample_t(n, target_pm1.device)
        noise = torch.randn_like(target_pm1)
        t_view = t.view(n, *([1] * (target_pm1.ndim - 1)))
        z_t = (1.0 - t_view) * noise + t_view * target_pm1
        velocity = (target_pm1 - z_t) / (1.0 - t_view).clamp_min(0.05)

        x_pred = self.net(z_t, t, cond)
        velocity_pred = (x_pred - z_t) / (1.0 - t_view).clamp_min(0.05)
        loss = F.mse_loss(velocity_pred.float(), velocity.float(), reduction="mean")

        with torch.no_grad():
            sign_acc = ((x_pred >= 0) == (target_pm1 >= 0)).float().mean()
            x_mse = F.mse_loss(x_pred.float(), target_pm1.float(), reduction="mean")

        return {
            "loss": loss,
            "flow_mse": loss.detach(),
            "x_mse": x_mse.detach(),
            "sign_acc": sign_acc.detach(),
            "inner_t": t.mean().detach(),
        }

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, num_sampling_steps: int = 100) -> torch.Tensor:
        return _euler_maruyama(
            self.bit_dim,
            self.net.forward,
            cond,
            num_sampling_steps=int(num_sampling_steps),
            time_shift=self.time_shift,
        )


# -----------------------------------------------------------------------------
# Controlled global TransformerBD adapter
# -----------------------------------------------------------------------------

class BinaryBitDanceDecouple(nn.Module):
    """BitDance-B style generator on the project's [B,256,64] bit latents.

    ``denoise_fn`` must be the existing ``TransformerBD`` instance.  Its
    original time embedding and direct bit head are frozen/unused; all global
    Transformer blocks, bit projection weights, 2-D position embeddings and
    final normalization are reused.
    """

    def __init__(self, H, denoise_fn: nn.Module, mask_id=None):
        super().__init__()
        del mask_id  # kept only for drop-in constructor compatibility

        self._denoise_fn = denoise_fn
        self.bit_dim = int(H.codebook_size)
        self.block_size = int(H.block_size)
        self.height = int(H.latent_shape[-2])
        self.width = int(H.latent_shape[-1])
        if self.height * self.width != self.block_size:
            raise ValueError(
                f"Expected block_size == H*W, got {self.block_size} vs "
                f"{self.height}x{self.width}"
            )

        self.parallel_num = int(getattr(H, "bitdance_parallel_num", 4))
        if self.parallel_num not in (1, 4, 16):
            raise ValueError(
                "Controlled BitDance patch mode supports parallel_num in {1,4,16}; "
                f"got {self.parallel_num}."
            )
        patch_side = int(math.isqrt(self.parallel_num))
        if patch_side * patch_side != self.parallel_num:
            raise ValueError("parallel_num must be a square for patch-mode grouping")
        if self.height % patch_side or self.width % patch_side:
            raise ValueError(
                f"Spatial shape {self.height}x{self.width} is not divisible by patch side {patch_side}."
            )
        if self.block_size % self.parallel_num:
            raise ValueError("block_size must be divisible by parallel_num")

        self.patch_side = patch_side
        self.outer_steps = self.block_size // self.parallel_num
        self.diff_batch_mul = int(getattr(H, "bitdance_diff_batch_mul", 4))
        if self.diff_batch_mul <= 0:
            raise ValueError("bitdance_diff_batch_mul must be positive")
        self.inner_infer_steps = int(getattr(H, "bitdance_inner_infer_steps", 100))
        self.perturb_rate = float(getattr(H, "bitdance_perturb_rate", 0.1))
        if not 0.0 <= self.perturb_rate <= 1.0:
            raise ValueError("bitdance_perturb_rate must lie in [0,1]")

        global_dim = int(H.bert_n_emb)
        diff_hidden = int(getattr(H, "bitdance_diff_dim", global_dim))
        diff_layers = int(getattr(H, "bitdance_diff_layers", 6))
        diff_adaln = int(getattr(H, "bitdance_diff_adaln_layers", 2))

        self.start_token = nn.Parameter(torch.zeros(1, 1, global_dim))
        self.query_tokens = nn.Parameter(
            torch.zeros(1, max(self.parallel_num - 1, 0), global_dim)
        )
        # Official BitDance adds a learned target-position embedding immediately
        # before the inner diffusion head.
        self.diff_pos = nn.Parameter(torch.zeros(1, self.block_size, global_dim))
        nn.init.normal_(self.start_token, std=0.02)
        if self.query_tokens.numel() > 0:
            nn.init.normal_(self.query_tokens, std=0.02)
        nn.init.normal_(self.diff_pos, std=0.02)

        self.head = BitDanceDiffHead(
            bit_dim=self.bit_dim,
            cond_dim=global_dim,
            hidden_dim=diff_hidden,
            depth=diff_layers,
            adaln_blocks=diff_adaln,
            parallel_num=self.parallel_num,
            time_shift=float(getattr(H, "bitdance_time_shift", 1.0)),
            time_schedule=str(getattr(H, "bitdance_time_schedule", "logit_normal")),
            p_std=float(getattr(H, "bitdance_p_std", 0.8)),
            p_mean=float(getattr(H, "bitdance_p_mean", -0.8)),
        )

        # TransformerBD's legacy direct Bernoulli head and global diffusion-time
        # token are not part of BitDance's global AR model.  Freeze rather than
        # delete them so the shared backbone class remains untouched.
        legacy_head = getattr(self._denoise_fn, "head", None)
        if legacy_head is not None:
            for p in legacy_head.parameters():
                p.requires_grad_(False)
        legacy_time = getattr(self._denoise_fn, "time_step_embedding", None)
        if legacy_time is not None:
            for p in legacy_time.parameters():
                p.requires_grad_(False)

        self.last_sampling_stats: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Spatial ordering helpers (same 2x2 patch-raster idea as BitDance 4x)
    # ------------------------------------------------------------------
    def _patch_reorder(self, x: torch.Tensor) -> torch.Tensor:
        """Raster [B,H*W,D] -> patch-major [B,H*W,D]."""
        b, n, d = x.shape
        if n != self.block_size:
            raise ValueError(f"Expected {self.block_size} tokens, got {n}")
        p = self.patch_side
        x = x.view(b, self.height, self.width, d)
        x = x.view(b, self.height // p, p, self.width // p, p, d)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(b, self.block_size, d)

    def _patch_unreorder(self, x: torch.Tensor) -> torch.Tensor:
        """Patch-major [B,H*W,D] -> raster [B,H*W,D]."""
        b, n, d = x.shape
        if n != self.block_size:
            raise ValueError(f"Expected {self.block_size} tokens, got {n}")
        p = self.patch_side
        x = x.view(b, self.height // p, self.width // p, p, p, d)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(b, self.height, self.width, d).view(b, self.block_size, d)

    def _reordered_spatial_pos(self) -> torch.Tensor:
        pos = self._denoise_fn.pos_emb[:, : self.block_size, :]
        return self._patch_reorder(pos)

    def _project_clean_tokens(self, x_pm1: torch.Tensor) -> torch.Tensor:
        # TransformerBD's original projection is ((x-.5)*2) @ tok_emb.weight.
        # x_pm1 is already in {-1,+1}, so this is exactly the same map.
        return x_pm1.to(self._denoise_fn.tok_emb.weight.dtype) @ self._denoise_fn.tok_emb.weight

    def _special_group(self, batch: int, dtype: torch.dtype, device: torch.device):
        start = self.start_token.expand(batch, -1, -1)
        if self.parallel_num == 1:
            return start.to(device=device, dtype=dtype)
        query = self.query_tokens.expand(batch, -1, -1)
        return torch.cat([start, query], dim=1).to(device=device, dtype=dtype)

    @staticmethod
    def _block_causal_mask(length: int, group: int, device, dtype):
        if length % group:
            raise ValueError(f"length={length} must be divisible by group={group}")
        ids = torch.arange(length, device=device) // group
        allowed = ids[None, :] <= ids[:, None]  # key-group <= query-group
        mask = torch.zeros((length, length), device=device, dtype=dtype)
        mask.masked_fill_(~allowed, float("-inf"))
        return mask

    def _existing_attention(self, attn: nn.Module, x: torch.Tensor, mask: torch.Tensor):
        """Execute the existing TransformerBD attention weights with block causality."""
        b, length, channels = x.shape
        n_heads = int(attn.num_heads)
        head_dim = channels // n_heads
        qkv = attn.qkv(x).view(b, length, 3, n_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask.view(1, 1, length, length),
            dropout_p=0.0,
        )
        y = y.transpose(1, 2).contiguous().view(b, length, channels)
        y = attn.proj(y)
        y = attn.proj_drop(y)
        return y

    def _global_forward_embedded(self, seq: torch.Tensor) -> torch.Tensor:
        """Run the *existing* TransformerBD blocks under BitDance block causality."""
        length = seq.shape[1]
        mask = self._block_causal_mask(
            length,
            self.parallel_num,
            seq.device,
            seq.dtype,
        )
        x = self._denoise_fn.drop(seq)
        for block in self._denoise_fn.blocks:
            h = block.ln1(x)
            h = self._existing_attention(block.attn, h, mask)
            x = x + block.drop_path(block.gamma_1 * h)
            x = x + block.drop_path(block.gamma_2 * block.mlp(block.ln2(x)))
        return self._denoise_fn.ln_f(x)

    @staticmethod
    def _existing_attention_incremental(
        attn: nn.Module,
        x: torch.Tensor,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """One block-causal group with a per-layer K/V cache.

        Within the newly generated group all tokens may attend to one another,
        and every token may attend to all previous groups.  Therefore no mask
        is needed once only the current group is used as the query.
        """
        b, length, channels = x.shape
        n_heads = int(attn.num_heads)
        head_dim = channels // n_heads
        qkv = attn.qkv(x).view(b, length, 3, n_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k_new, v_new = qkv[0], qkv[1], qkv[2]
        if cache is None:
            k_all, v_all = k_new, v_new
        else:
            k_all = torch.cat([cache[0], k_new], dim=2)
            v_all = torch.cat([cache[1], v_new], dim=2)
        y = F.scaled_dot_product_attention(
            q, k_all, v_all, dropout_p=0.0
        )
        y = y.transpose(1, 2).contiguous().view(b, length, channels)
        y = attn.proj(y)
        y = attn.proj_drop(y)
        return y, (k_all, v_all)

    def _global_forward_group_cached(
        self,
        group_seq: torch.Tensor,
        kv_caches: Optional[list],
    ) -> Tuple[torch.Tensor, list]:
        """Incremental global Transformer step used for practical 50K sampling."""
        if group_seq.shape[1] != self.parallel_num:
            raise ValueError(
                f"Expected one group of {self.parallel_num} tokens, got {group_seq.shape[1]}"
            )
        if kv_caches is None:
            kv_caches = [None] * len(self._denoise_fn.blocks)
        if len(kv_caches) != len(self._denoise_fn.blocks):
            raise ValueError("K/V cache depth mismatch")

        x = self._denoise_fn.drop(group_seq)
        new_caches = []
        for layer_index, block in enumerate(self._denoise_fn.blocks):
            h = block.ln1(x)
            h, layer_cache = self._existing_attention_incremental(
                block.attn, h, kv_caches[layer_index]
            )
            x = x + block.drop_path(block.gamma_1 * h)
            x = x + block.drop_path(block.gamma_2 * block.mlp(block.ln2(x)))
            new_caches.append(layer_cache)
        return self._denoise_fn.ln_f(x), new_caches

    def _flip_context(self, x_pm1: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.perturb_rate <= 0:
            return x_pm1
        # Released BitDance: flip if r1 < p_max * r2; expected flip rate p_max/2.
        r1 = torch.rand_like(x_pm1)
        r2 = torch.rand_like(x_pm1)
        flip = r1 < self.perturb_rate * r2
        return x_pm1 * torch.where(flip, -torch.ones_like(x_pm1), torch.ones_like(x_pm1))

    def _training_context(self, target_pm1_grouped: torch.Tensor) -> torch.Tensor:
        b = target_pm1_grouped.shape[0]
        x_context = self._flip_context(target_pm1_grouped.detach())
        previous = x_context[:, : self.block_size - self.parallel_num, :]
        previous_emb = self._project_clean_tokens(previous)
        previous_pos = self._reordered_spatial_pos()[:, : previous.shape[1], :]
        previous_emb = previous_emb + previous_pos

        special = self._special_group(
            b,
            dtype=previous_emb.dtype if previous_emb.numel() else self.start_token.dtype,
            device=target_pm1_grouped.device,
        )
        seq = torch.cat([special, previous_emb], dim=1)
        if seq.shape[1] != self.block_size:
            raise RuntimeError(
                f"Internal BitDance shift produced length {seq.shape[1]}, expected {self.block_size}"
            )
        return self._global_forward_embedded(seq)

    # ------------------------------------------------------------------
    # Training API expected by the project's sampler loop
    # ------------------------------------------------------------------
    def _train_loss(self, x_0: torch.Tensor, label=None, x_ct=None):
        del label
        if x_ct is not None:
            raise NotImplementedError("BitDance controlled baseline does not use x_ct conditioning")
        if x_0.ndim != 3 or x_0.shape[1:] != (self.block_size, self.bit_dim):
            raise ValueError(
                f"Expected x_0 [B,{self.block_size},{self.bit_dim}], got {tuple(x_0.shape)}"
            )

        target_pm1 = x_0.float() * 2.0 - 1.0
        target_pm1 = self._patch_reorder(target_pm1)
        global_context = self._training_context(target_pm1)
        global_context = global_context + self.diff_pos

        b = x_0.shape[0]
        groups = self.outer_steps
        target_groups = target_pm1.view(b, groups, self.parallel_num, self.bit_dim)
        context_groups = global_context.view(
            b, groups, self.parallel_num, global_context.shape[-1]
        )

        target_flat = target_groups.reshape(
            b * groups, self.parallel_num, self.bit_dim
        )
        context_flat = context_groups.reshape(
            b * groups, self.parallel_num, global_context.shape[-1]
        )

        if self.diff_batch_mul > 1:
            target_flat = target_flat.repeat(self.diff_batch_mul, 1, 1)
            context_flat = context_flat.repeat(self.diff_batch_mul, 1, 1)

        stats = self.head(target_flat, context_flat)
        stats["outer_steps"] = torch.tensor(
            float(self.outer_steps), device=x_0.device
        )
        return stats

    def forward(self, x_0: torch.Tensor, label=None, x_ct=None):
        return self._train_loss(x_0, label=label, x_ct=x_ct)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def _sampling_context_for_group(
        self,
        generated_grouped_pm1: Optional[torch.Tensor],
        group_index: int,
        batch: int,
    ) -> torch.Tensor:
        if generated_grouped_pm1 is None:
            previous_emb = None
            dtype = self.start_token.dtype
            device = self.start_token.device
        else:
            expected = group_index * self.parallel_num
            if generated_grouped_pm1.shape[1] != expected:
                raise ValueError(
                    f"Expected {expected} generated tokens at group {group_index}, "
                    f"got {generated_grouped_pm1.shape[1]}"
                )
            previous_emb = self._project_clean_tokens(generated_grouped_pm1)
            previous_pos = self._reordered_spatial_pos()[:, :expected, :]
            previous_emb = previous_emb + previous_pos
            dtype = previous_emb.dtype
            device = previous_emb.device

        special = self._special_group(batch, dtype=dtype, device=device)
        seq = special if previous_emb is None else torch.cat([special, previous_emb], dim=1)
        expected_len = (group_index + 1) * self.parallel_num
        if seq.shape[1] != expected_len:
            raise RuntimeError(
                f"Sampling prefix length {seq.shape[1]} != expected {expected_len}"
            )
        features = self._global_forward_embedded(seq)
        return features[:, -self.parallel_num :, :]

    @torch.no_grad()
    def sample(
        self,
        temp: float = 1.0,
        sample_steps: Optional[int] = None,
        inner_steps: Optional[int] = None,
        b: int = 8,
        shape=None,
        return_all: bool = False,
        label=None,
        mask=None,
        guidance=None,
        full: bool = False,
    ) -> torch.Tensor:
        """Generate strict binary [B,256,64] latents.

        ``sample_steps`` is retained for compatibility and, when supplied,
        denotes the *inner* BitDance flow-head steps.  The expensive outer
        Transformer count is fixed by the trained factorization:
        ``256 / parallel_num`` (64 for the default 4x model).
        """
        del label, mask, guidance, full
        if temp != 1.0:
            raise ValueError(
                "Released BitDance continuous sampling has no Bernoulli temperature; use temp=1.0."
            )
        if return_all:
            raise NotImplementedError("return_all is not implemented for BitDance")
        if shape is not None:
            if tuple(shape[1:]) != (self.block_size, self.bit_dim):
                raise ValueError(
                    f"shape must be [B,{self.block_size},{self.bit_dim}], got {shape}"
                )
            b = int(shape[0])

        if inner_steps is None:
            inner_steps = sample_steps if sample_steps is not None else self.inner_infer_steps
        inner_steps = int(inner_steps)
        if inner_steps <= 0:
            raise ValueError("inner_steps must be positive")

        generated_groups = []
        kv_caches = None
        reordered_pos = self._reordered_spatial_pos()
        current_input = self._special_group(
            b, dtype=self.start_token.dtype, device=self.start_token.device
        )

        for group_index in range(self.outer_steps):
            context, kv_caches = self._global_forward_group_cached(
                current_input, kv_caches
            )
            start = group_index * self.parallel_num
            end = start + self.parallel_num
            context = context + self.diff_pos[:, start:end, :]
            continuous = self.head.sample(context, num_sampling_steps=inner_steps)
            hard_pm1 = torch.where(
                continuous >= 0,
                torch.ones_like(continuous),
                -torch.ones_like(continuous),
            )
            generated_groups.append(hard_pm1)

            if group_index + 1 < self.outer_steps:
                # The next global group consumes exactly the clean tokens just
                # generated, shifted by one patch group, as in BitDance.
                source_pos = reordered_pos[:, start:end, :]
                current_input = self._project_clean_tokens(hard_pm1) + source_pos

        generated = torch.cat(generated_groups, dim=1)
        raster_pm1 = self._patch_unreorder(generated)
        binary = (raster_pm1 > 0).to(dtype=torch.float32)
        self.last_sampling_stats = {
            "global_transformer_nfe": float(self.outer_steps),
            "inner_flow_steps_per_outer": float(inner_steps),
            "inner_flow_steps_path_total": float(self.outer_steps * inner_steps),
            "parallel_num": float(self.parallel_num),
        }
        return binary

    def parameter_report(self) -> Dict[str, int]:
        global_names = (
            "_denoise_fn.tok_emb",
            "_denoise_fn.pos_emb",
            "_denoise_fn.blocks",
            "_denoise_fn.ln_f",
        )
        counts = {"global": 0, "inner": 0, "special": 0, "trainable_total": 0}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            counts["trainable_total"] += p.numel()
            if name.startswith("head."):
                counts["inner"] += p.numel()
            elif name.startswith("start_token") or name.startswith("query_tokens") or name.startswith("diff_pos"):
                counts["special"] += p.numel()
            elif name.startswith(global_names):
                counts["global"] += p.numel()
            else:
                # Future backbone parameters should be counted with global rather
                # than silently dropped from reporting.
                counts["global"] += p.numel()
        return counts
