# Modified from:
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py  
#   LlamaGen: https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/models/gpt.py

from dataclasses import dataclass
from typing import Optional, List


import torch
import torch.nn as nn
from torch.nn import functional as F
from .drop_path import DropPath
import torchvision
from llama.bindiff.binarylatent import BinaryDiffusion
import math
from einops import rearrange, reduce, pack, unpack
import scipy.stats as stats
import numpy as np
import random

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)

def indexify(x):
    row = torch.arange(x.size(0)).repeat_interleave(x.size(1)).view(-1)
    col = x.view(-1)
    
    indices = (row.tolist(), col.tolist())
    return indices
    
@dataclass
class ModelArgs:
    ### mlm
    mask_ratio_min: float = 0.5
    mask_ratio_max: float = 1.0
    mask_ratio_mu: float = 0.55
    mask_ratio_std: float = 0.25
    smoothing: float = 0.1
    gen_iter: int = 10
    
    ### model
    dim: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    
    ### train setting
    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    drop_path_rate: float = 0.0

    num_classes: int = 1000
    caption_dim: int = 2048
    class_dropout_prob: float = 0.1

    quant_size: int = 256
    binary_size: int = 16
    cls_token_num: int = 1
    block_size: int = 256
    max_batch_size: int = 32
    max_seq_len: int = 2048
    
    denoise_hidden_dim: int = 1024
    p_flip: bool=False
    n_repeat: int=1
    aux: float=0.0
    focal: float=0.0
    alpha: float=-1
    
    sample_temperature: float=1.0
    n_sample_steps: int=256
    infer_steps: int=100
    
    num_block_mlp: int=3
    seq_len: int=256
    
    use_adaLN: bool=False

#################################################################################
#                      Embedding Layers for Class Labels                        #
#################################################################################
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

#################################################################################
#                             Diffusion Func                                    #
#################################################################################
    
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_dim, out_dim, cond_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_dim, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    
class DenoiseBlock(nn.Module):
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * hidden_dim, bias=True)
        )

    def forward(self, x, c):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(3, dim=1)
        x = x + gate_mlp * self.mlp(modulate(self.norm(x), shift_mlp, scale_mlp))
        return x
    
class DenoiseModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, cond_dim, num_blocks, rescale=True):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.rescale = rescale
        self.t_embedder = TimestepEmbedder(hidden_dim)
        
        self.x_embedder = nn.Linear(in_dim, hidden_dim)
        self.c_embedder = nn.Linear(cond_dim, hidden_dim)
        
        self.blocks = nn.ModuleList(
            [DenoiseBlock(hidden_dim, hidden_dim) for _ in range(num_blocks)]
        )
        self.final_layer = FinalLayer(hidden_dim, out_dim, hidden_dim)

    def forward(self, x, time_steps, c):
        if self.rescale:
            x = (x*1.0 - 0.5)*2.0

        x = self.x_embedder(x)
        c = self.c_embedder(c)
        t = self.t_embedder(time_steps)
        
        c = t + c
        
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        return x

    def forward_with_cfg(self, x, time_steps, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, time_steps, c)
        eps, rest = model_out[:, :self.in_dim], model_out[:, self.in_dim:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        
        return torch.cat([eps, rest], dim=1)

#################################################################################
#                                  GPT Model                                    #
#################################################################################
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        # regularization
        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor = None, 
        input_pos: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None
    ):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        output = F.scaled_dot_product_attention(
            xq, keys, values, 
            attn_mask=mask, 
            is_causal=True if mask is None else False, # is_causal=False is for KV cache
            dropout_p=self.attn_dropout_p if self.training else 0)            
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, drop_path: float, use_adaLN=False):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_adaLN = use_adaLN
        
        if self.use_adaLN:
            self.ada_gss = nn.Parameter(torch.randn(1,1,6,config.dim) / config.dim**0.5)
            
    def forward(
        self, x: torch.Tensor, cond_adaln: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None):
        
        if self.use_adaLN:
            # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=2)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (cond_adaln + self.ada_gss).unbind(2) # 116C + B16C
            h = x + gate_msa * self.drop_path(self.attention(modulate(self.attention_norm(x), shift_msa, scale_msa), freqs_cis, start_pos, mask))
            out = h + gate_mlp * self.drop_path(self.feed_forward(modulate(self.ffn_norm(h), shift_mlp, scale_mlp)))
        else:
            h = x + self.drop_path(self.attention(self.attention_norm(x), freqs_cis, start_pos, mask))
            out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
            
        return out
    
class Transformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.num_classes = config.num_classes
        self.cls_token_num = config.cls_token_num
        self.sample_temperature = config.sample_temperature
        self.n_sample_steps = config.n_sample_steps
        self.infer_steps = config.infer_steps
        self.smoothing = config.smoothing
        self.gen_iter = config.gen_iter
        
        self.cls_embedding = LabelEmbedder(config.num_classes, config.dim, config.class_dropout_prob)
        
        self.mask_tok = nn.Parameter(torch.randn(1, config.dim))
        self.tok_embeddings = nn.Linear(config.binary_size, config.dim)
        self.tok_dropout = nn.Dropout(config.token_dropout_p)
        
        # transformer blocks
        if self.config.use_adaLN:
            self.adaLN = nn.Sequential(
                        nn.SiLU(),
                        nn.Linear(config.dim, 6 * config.dim, bias=True)
                    )
            print("We are using adaLN!")
        else:
            self.adaLN = None
            
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(TransformerBlock(config, dpr[layer_id], use_adaLN=self.config.use_adaLN))

        # output layer
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        
        # binary diffusion
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.config.seq_len, config.dim))
        self.denoise_mlp = DenoiseModel(in_dim=config.binary_size, hidden_dim=config.denoise_hidden_dim,
                                    out_dim=config.binary_size, cond_dim=config.dim, num_blocks=config.num_block_mlp)
        self.diffusion_processor = BinaryDiffusion(denoise_fn=self.denoise_mlp, p_flip=config.p_flip,
                                                n_repeat=config.n_repeat, aux=config.aux, focal=config.focal, alpha=config.alpha, 
                                                n_sample_steps=config.n_sample_steps)
            
        # 2d rotary pos embedding
        grid_size = int(self.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num)
        
        # KVCache
        self.max_batch_size = -1
        self.max_seq_length = -1
        
        self.initialize_weights()

    def initialize_weights(self):        
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        
        
        # Zero-out adaLN modulation layers in GPT blocks:
        if self.config.use_adaLN:
            # for layer in self.layers:
            #     nn.init.constant_(layer.adaLN_modulation[-1].weight, 0)
            #     nn.init.constant_(layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.adaLN[-1].weight, 0)
            nn.init.constant_(self.adaLN[-1].bias, 0)
        
        # Initialize pos emb
        nn.init.normal_(self.diffusion_pos_embed_learned, std=0.02)
    
        # Zero-out adaLN modulation layers in MLP blocks:
        for block in self.denoise_mlp.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.denoise_mlp.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.denoise_mlp.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.denoise_mlp.final_layer.linear.weight, 0)
        nn.init.constant_(self.denoise_mlp.final_layer.linear.bias, 0)
        
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.denoise_mlp.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.denoise_mlp.t_embedder.mlp[2].weight, std=0.02)
        

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        # if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
        #     return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype)

        causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        grid_size = int(self.config.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num)

    def forward(
        self, 
        inp: torch.Tensor, 
        cond_idx: torch.Tensor,  # cond_idx_or_embed
        token_all_mask: Optional[torch.Tensor] = None,
        input_pos:  Optional[torch.Tensor] = None, 
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        cfg_scale: Optional[float] = None,
        interpolate: Optional[float] = None,
    ):
        
        device = inp.device if not inp is None else cond_idx.device
        bsz, seq_len, t_n = inp.size() if not inp is None else (cond_idx.size()[0], 0, self.config.binary_size)
        
        ################### Masking #####################
        if token_all_mask is None: ### training process
            step = torch.randint(0, seq_len, (1,))
            ratio = 1. * (step) / seq_len
            mask_rate = torch.cos(math.pi / 2. * ratio)
            num_masked_tokens = torch.floor(seq_len * mask_rate)
            num_masked_tokens = int(torch.maximum(torch.Tensor([1]), num_masked_tokens).item())
            while True:
                noise = torch.rand(bsz, seq_len, device=device)  # noise in [0, 1]
                sorted_noise, _ = torch.sort(noise, dim=1)  # ascend: small is remove, large is keep
                cutoff_mask = sorted_noise[:, num_masked_tokens-1:num_masked_tokens]
                token_all_mask = (noise <= cutoff_mask).float()
                if token_all_mask.sum() == bsz*num_masked_tokens:
                    break
                else:
                    print("Rerandom the noise!")
        ################################################
        
        if inp is not None and cond_idx is not None: # training or naive inference
            if isinstance(cond_idx, tuple):
                embeddings_nll = []
                for idx_c, cond_idx_i in enumerate(cond_idx):
                    cond_embeddings = self.cls_embedding(cond_idx_i, train=self.training).repeat_interleave(self.cls_token_num, dim=1)
                    embeddings_nll.append(cond_embeddings * interpolate[idx_c])
                cond_embeddings = torch.stack(embeddings_nll, dim=0).sum(0)
            else: 
                cond_embeddings = self.cls_embedding(cond_idx, train=self.training).repeat_interleave(self.cls_token_num, dim=1)
            token_embeddings = self.tok_embeddings(inp)
            indices = token_all_mask.nonzero(as_tuple=True)
                
            token_embeddings[indices[0], indices[1], :] = self.mask_tok.to(token_embeddings.dtype)
            token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)
            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis.to(h.device)
            cond_inp = cond_embeddings # cache cond in the forward process
        else:
            if cond_idx is not None: # prefill in inference
                if isinstance(cond_idx, tuple):
                    embeddings_nll = []
                    for idx_c, cond_idx_i in enumerate(cond_idx):
                        cond_embeddings = self.cls_embedding(cond_idx_i, train=self.training).repeat_interleave(self.cls_token_num, dim=1)
                        embeddings_nll.append(cond_embeddings * interpolate[idx_c])
                    token_embeddings = torch.stack(embeddings_nll, dim=0).sum(0)
                else: 
                    token_embeddings = self.cls_embedding(cond_idx, train=self.training).repeat_interleave(self.cls_token_num, dim=1)
                cond_inp = token_embeddings # cache cond in the forward process
            else: # decode_n_tokens(kv cache) in inference
                token_embeddings = self.tok_embeddings(inp)
                indices = token_all_mask.nonzero(as_tuple=True)
                token_embeddings[indices[0], indices[1], :] = self.mask_tok.to(token_embeddings.dtype)

            bs = token_embeddings.shape[0]
            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis.to(h.device)
        
        if self.config.use_adaLN:
            d1, d2, d3 = cond_inp.shape
            cond_adaln = self.adaLN(cond_inp).reshape(d1, d2, 6, d3) # shared_adaLN
        else:
            cond_adaln = None
            
        freqs_cis = self.freqs_cis[:token_embeddings.shape[1]]
        
        mask = torch.ones((token_embeddings.shape[-2], token_embeddings.shape[-2]), dtype=torch.bool).to(device)
        
        # transformer blocks
        for layer in self.layers:
            h = layer(h, cond_adaln, freqs_cis, input_pos, mask)
        
        # output layers
        logits = self.norm(h).float()
        # logits = self.output(h).float()
        # logits = self.final_layer(h).float()
        
        logits = logits[:, self.cls_token_num:].contiguous() # discard all cls tokens
        logits = logits + self.diffusion_pos_embed_learned
        
        if self.training:
            stats = self.diffusion_processor(
                targets.reshape(-1, targets.size(-1)), cond=logits.reshape(-1, logits.size(-1))
            ) # targets: (B*N, in_dim), logits: (B*N, hidden_dim)
            
            loss_all = stats['loss_all'].reshape(bsz, seq_len, -1)
            bce_loss_all = stats['bce_loss_all'].reshape(bsz, seq_len, -1)
            token_all_mask = token_all_mask.unsqueeze(-1).repeat_interleave(t_n, dim=-1)
            loss = (loss_all * token_all_mask).sum() / token_all_mask.sum()  # mean loss on removed patches
            bce_loss = (bce_loss_all * token_all_mask).sum() / token_all_mask.sum()  # mean loss on removed patches
            
            stats['bce_loss'] = bce_loss
            stats['loss'] = loss
            
            return logits, stats
        
        else:
            n_bs, n_seq, n_dim = logits.shape
            logits = self.diffusion_processor.sample(
                temp=self.sample_temperature, sample_steps=self.infer_steps,
                cond=logits.reshape(-1, logits.size(-1)), cfg_scale=cfg_scale, return_logits=True
            )
            return logits.reshape(-1, n_seq, self.config.binary_size), None
            
    
    def decode_one_step(self, x, cond_idx, token_all_mask, input_pos, cfg_scale, interpolate=None, **sampling_kwargs):
        x_combined = torch.cat([x, x])
        token_all_mask_combined = torch.cat([token_all_mask, token_all_mask])
        logits, _ = self(x_combined, cond_idx=cond_idx, token_all_mask=token_all_mask_combined, input_pos=input_pos, cfg_scale=cfg_scale, interpolate=interpolate)
        return logits
    
    @torch.no_grad()
    def generate_with_cfg(self, cond, max_new_tokens, interpolate=None, num_iter=10, cond_padding=1, out_dim=64, emb_masks=None, cfg_scale=1.0, cfg_schedule='constant', gumbel_temp=1., gumbel_schedule='constant', **sampling_kwargs):
        # print("testing confidence+undeterm in inference!")
        if cfg_scale > 1.0:
            if isinstance(cond, tuple):
                cond_tuple = []
                for cond_i in cond:
                    cond_null = torch.ones_like(cond_i) * self.num_classes
                    cond_combined = torch.cat([cond_i, cond_null])
                    cond_tuple.append(cond_combined)
                cond_combined = tuple(cond_tuple)
            else:
                cond_null = torch.ones_like(cond) * self.num_classes
                cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = cond_padding

        T_new = T + max_new_tokens
        max_seq_length = T_new
        if isinstance(cond, tuple):
            max_batch_size = cond[0].shape[0]
            device = cond[0].device
        else:
            max_batch_size = cond.shape[0]
            device = cond.device
        
        # create an empty tensor of the expected final shape and fill in the current tokens
        seq = torch.empty((max_batch_size, max_new_tokens, out_dim), device=device)
        token_all_mask = torch.ones((max_batch_size, max_new_tokens), dtype=torch.bool, device=device)
        
        old_mask_len = max_new_tokens
        for step in range(num_iter): # (256) 252 --> 0
            ratio = 1. * (step + 1) / num_iter
            mask_rate = np.cos(math.pi / 2. * ratio)
            new_mask_len = math.floor(mask_rate*max_new_tokens)
            predict_len = old_mask_len - new_mask_len
            old_mask_len = new_mask_len
            
            # cfg schedule
            if cfg_schedule == 'linear':
                # cfg = 1. + (cfg_scale - 1.) * (1. - mask_rate)
                cfg = 1. + (cfg_scale - 1.) * ratio
            elif cfg_schedule == 'constant':
                cfg = cfg_scale
                
            input_pos = None
            pred = self.decode_one_step(seq, cond_combined, token_all_mask, input_pos, cfg, interpolate=interpolate, **sampling_kwargs)
            
            confidence = torch.abs(pred-0.5)*2.0
            confidence = confidence.mean(-1)
            confidence[(~token_all_mask).nonzero(as_tuple=True)] = float('-inf')
            
            # gumbel schedule
            if gumbel_schedule == 'down':
                gumbel_cur = gumbel_temp * (1. - ratio)
            elif gumbel_schedule == 'up':
                gumbel_cur = gumbel_temp * ratio
            elif gumbel_schedule == 'constant':
                gumbel_cur = gumbel_temp
            
            #### gumbel sampling #####
            confidence = confidence + torch.Tensor(
                gumbel_cur * np.random.gumbel(size=confidence.shape)
                ).to(confidence.device)
            ##########################
            
            indices = torch.topk(confidence, k=predict_len, dim=1)[1]
            indices = indexify(indices)

            # seq[indices] = (pred[indices] > 0.5) * 1.0
            seq[indices] = torch.bernoulli(pred[indices]) * 1.0
            token_all_mask[indices] = False
            
        if not torch.all(~token_all_mask): # if not fully unmasked
            print("Warning: Not unmask all patches! {} left.".format(token_all_mask[0].sum()))
            pred = self.decode_one_step(seq, cond_combined, token_all_mask, input_pos, cfg_scale, interpolate=interpolate, **sampling_kwargs)
            indices = token_all_mask.nonzero(as_tuple=True)
            # seq[indices] = (pred[indices] > 0.5) * 1.0
            seq[indices] = torch.bernoulli(pred[indices]) * 1.0
            token_all_mask[indices] = False
        
        return seq
    
    @torch.no_grad()
    def inpaint_with_cfg(self, cond, max_new_tokens, input_seq, inpaint_mask, mask_schedule='linear', num_iter=10, cond_padding=1, out_dim=64, emb_masks=None, cfg_scale=1.0, cfg_schedule='constant', gumbel_temp=10., gumbel_schedule='constant', **sampling_kwargs):

        cond_null = torch.ones_like(cond) * self.num_classes
        cond_combined = torch.cat([cond, cond_null])

        T = cond_padding

        T_new = T + max_new_tokens
        max_seq_length = T_new
        
        max_batch_size = cond.shape[0]
        device = cond.device
        
        # create an empty tensor of the expected final shape and fill in the current tokens
        seq = torch.empty((max_batch_size, max_new_tokens, out_dim), device=device)
        token_all_mask = torch.ones((max_batch_size, max_new_tokens), dtype=torch.bool, device=device)
        
        # load unmasked info
        ### some steps ###
        inpaint_indices = (~inpaint_mask).nonzero(as_tuple=True)
        seq[inpaint_indices] = input_seq[inpaint_indices]
        token_all_mask[inpaint_indices] = False

        assert len(inpaint_indices[0]) % max_batch_size == 0
        
        max_new_tokens = len(inpaint_indices[0]) // max_batch_size # important!
        
        old_mask_len = max_new_tokens
        for step in range(num_iter): # (256) 252 --> 0
            ratio = 1. * (step + 1) / num_iter
            
            if mask_schedule == "cos":
                mask_rate = np.cos(math.pi / 2. * ratio)
            elif mask_schedule == "linear":
                mask_rate = 1. - ratio
                
            new_mask_len = math.floor(mask_rate*max_new_tokens)
            predict_len = old_mask_len - new_mask_len
            old_mask_len = new_mask_len
            
            # cfg schedule
            if cfg_schedule == 'linear':
                # cfg = 1. + (cfg_scale - 1.) * (1. - mask_rate)
                cfg = 1. + (cfg_scale - 1.) * ratio
            elif cfg_schedule == 'constant':
                cfg = cfg_scale
                
            input_pos = None
            pred = self.decode_one_step(seq, cond_combined, token_all_mask, input_pos, cfg, **sampling_kwargs)
            
            confidence = torch.abs(pred-0.5)*2.0
            confidence = confidence.mean(-1)
            confidence[(~token_all_mask).nonzero(as_tuple=True)] = float('-inf')
            
            # gumbel schedule
            if gumbel_schedule == 'down':
                gumbel_cur = gumbel_temp * (1. - ratio)
            elif gumbel_schedule == 'up':
                gumbel_cur = gumbel_temp * ratio
            elif gumbel_schedule == 'constant':
                gumbel_cur = gumbel_temp
            
            #### gumbel sampling #####
            confidence = confidence + torch.Tensor(
                gumbel_cur * np.random.gumbel(size=confidence.shape)
                ).to(confidence.device)
            ##########################
            
            indices = torch.topk(confidence, k=predict_len, dim=1)[1]
            indices = indexify(indices)

            # seq[indices] = (pred[indices] > 0.5) * 1.0
            seq[indices] = torch.bernoulli(pred[indices]) * 1.0
            token_all_mask[indices] = False
            
        if not torch.all(~token_all_mask): # if not fully unmasked
            print("Warning: Not unmask all patches!")
            pred = self.decode_one_step(seq, cond_combined, token_all_mask, input_pos, cfg_scale, **sampling_kwargs)
            indices = token_all_mask.nonzero(as_tuple=True)
            # seq[indices] =  (pred[indices] > 0.5) * 1.0
            seq[indices] = torch.bernoulli(pred[indices]) * 1.0
            token_all_mask[indices] = False
        
        return seq

    @torch.no_grad()
    def supres_box(self, cond, max_new_tokens, input_seq, high_res, ds_rate, num_iter=10, cond_padding=1, out_dim=64, emb_masks=None, cfg_scale=1.0, cfg_schedule='constant', gumbel_temp=10., gumbel_schedule='constant', **sampling_kwargs):

        cond_null = torch.ones_like(cond) * self.num_classes
        cond_combined = torch.cat([cond, cond_null])

        T = cond_padding

        T_new = T + max_new_tokens
        max_seq_length = T_new
        
        max_batch_size = cond.shape[0]
        device = cond.device
        
        # create an empty tensor of the expected final shape and fill in the current tokens
        seq = input_seq
        low_res = high_res // ds_rate
        input_msk = torch.zeros((1, low_res, low_res, ds_rate**2))
        
        pos_list = list(range(ds_rate**2))
        random.shuffle(pos_list)
        N=4
        list_mask = [pos_list[i::N] for i in range(N)]
        for lm in list_mask:
            for pos in lm:
                input_msk[:,:,:,pos] = 1
                
            mask = input_msk.reshape(
                1, low_res, low_res, ds_rate, ds_rate
                ).permute(0, 1, 3, 2, 4).reshape(
                    1, low_res*ds_rate, low_res*ds_rate
                    ).reshape(1, high_res**2).to(torch.bool)

            seq = self.inpaint_with_cfg(
                cond, max_new_tokens, seq, inpaint_mask=mask, mask_schedule='linear',
                num_iter=num_iter, cond_padding=cond_padding, out_dim=out_dim, emb_masks=emb_masks, 
                cfg_scale=cfg_scale, cfg_schedule=cfg_schedule, gumbel_temp=gumbel_temp, 
                gumbel_schedule=gumbel_schedule, **sampling_kwargs)
            
            for pos in lm:
                input_msk[:,:,:,pos] = 0
            
        return seq
    
    @torch.no_grad()
    def forward_full_image(self, inp: torch.Tensor, cond_idx: torch.Tensor, layer_index: int):
        device = inp.device if not inp is None else cond_idx.device
        bsz, seq_len, t_n = inp.size() if not inp is None else (cond_idx.size()[0], 0, self.config.binary_size)
        
        if inp is not None and cond_idx is not None: # training or naive inference
            if isinstance(cond_idx, tuple):
                embeddings_nll = []
                for cond_idx_i in cond_idx:
                    cond_embeddings = self.cls_embedding(cond_idx_i, train=self.training).repeat_interleave(self.cls_token_num, dim=1)
                    embeddings_nll.append(cond_embeddings)
                cond_embeddings = torch.stack(embeddings_nll, dim=0).mean(0)
            else: 
                cond_embeddings = self.cls_embedding(cond_idx, train=self.training).repeat_interleave(self.cls_token_num, dim=1)
            token_embeddings = self.tok_embeddings(inp)
            
            token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)
            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis.to(h.device)
            cond_inp = cond_embeddings # cache cond in the forward process
        
        if self.config.use_adaLN:
            d1, d2, d3 = cond_inp.shape
            cond_adaln = self.adaLN(cond_inp).reshape(d1, d2, 6, d3) # shared_adaLN
        else:
            cond_adaln = None
            
        freqs_cis = self.freqs_cis[:token_embeddings.shape[1]]
        
        mask = torch.ones((token_embeddings.shape[-2], token_embeddings.shape[-2]), dtype=torch.bool).to(device)
        
        # transformer blocks
        for i_l, layer in enumerate(self.layers):
            h = layer(h, cond_adaln, freqs_cis, None, mask)
            if i_l == layer_index: # output at [layer_index]th layer
                break
        
        # output layers
        # logits = self.norm(h).float()
        logits = h.float()
        
        logits = logits[:, self.cls_token_num:].contiguous() # discard all cls tokens

        return logits
        
    def clear_cond(self):
        return # clear cond in the forward process
        
    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)

#################################################################################
#                      Rotary Positional Embedding Functions                    #
#################################################################################
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py 
def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, cls_token_num=120):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs) # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1) # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+seq_len, head_dim // 2, 2)
    return cond_cache 


def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs) # (grid_size, head_dim // 2)
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, grid_size, -1),
        freqs[None, :, :].expand(grid_size, -1, -1),
    ], dim=-1)  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1) # (grid_size, grid_size, head_dim // 2, 2)
    cache = cache_grid.flatten(0, 1)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+grid_size**2, head_dim // 2, 2)
    return cond_cache 


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2) # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2) # (1, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack([
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)



#################################################################################
#                                GPT Configs                                    #
#################################################################################
# resolution: 256x256 
def BIGR_XXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=24, dim=1536, num_block_mlp=8, denoise_hidden_dim=1536, **kwargs)) # 1.5B

def BIGR_XL(**kwargs):
    return Transformer(ModelArgs(n_layer=36, n_head=20, dim=1280, num_block_mlp=6, denoise_hidden_dim=1280, **kwargs)) # 799M

def BIGR_L(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=16, dim=1024, num_block_mlp=3, denoise_hidden_dim=1024, **kwargs)) # 336M

# resolution: 512x512       
def BIGR_L_512(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=16, dim=1024, num_block_mlp=6, denoise_hidden_dim=1280, **kwargs)) # 373M

BIGR_models = {
    'BiGR-L': BIGR_L, 'BiGR-XL': BIGR_XL, 'BiGR-XXL': BIGR_XXL, 'BiGR-L-512': BIGR_L_512
}
