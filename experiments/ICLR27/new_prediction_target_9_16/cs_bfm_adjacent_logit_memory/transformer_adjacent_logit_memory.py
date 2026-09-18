"""TransformerBD conditioned on the immediately preceding full clean-X0 logit."""
from __future__ import annotations
import torch
from torch import nn
from models.transformer import TransformerBD

class TransformerBDAdjacentLogitMemory(TransformerBD):
    def __init__(self,H,avg_pooling:bool=False):
        super().__init__(H,avg_pooling=avg_pooling)
        if int(H.total_steps)!=64: raise ValueError('adjacent logit memory requires T=64')
        self.prev_logit_scale=4.0
        self.prev_logit_proj=nn.Linear(self.codebook_size,self.n_embd,bias=True)
        nn.init.normal_(self.prev_logit_proj.weight,mean=0.0,std=0.02)
        nn.init.zeros_(self.prev_logit_proj.bias)
        self.adjacent_input_config={'feature':'asinh(prev_clean_logit/4) -> Linear(64,n_embd)','hard_clip':False,'state_semantics':'condition only; never added to current z'}

    def forward(self,x_t,prev_clean_logit=None,label=None,time_steps=None):
        if time_steps is None: raise ValueError('time_steps is required')
        if prev_clean_logit is None: prev_clean_logit=torch.zeros_like(x_t,dtype=torch.float32)
        if prev_clean_logit.shape!=x_t.shape: raise ValueError('prev_clean_logit shape mismatch')
        token=((x_t.float()-0.5)*2.0) @ self.tok_emb.weight
        prev_feat=torch.asinh(prev_clean_logit.float()/self.prev_logit_scale)
        token=token+self.prev_logit_proj(prev_feat.to(token.dtype))
        n=token.shape[1]; hidden=token+self.pos_emb[:,:n,:]
        hidden=torch.cat([hidden,self.time_step_embedding(time_steps)],dim=1)
        if self.exp_type.endswith('tkn') and label is not None:
            hidden=torch.cat([hidden,self.cls_embedding(label).unsqueeze(1)],dim=1)
        hidden=self.drop(hidden)
        for block in self.blocks:
            hidden=block(hidden,label) if self.exp_type=='t2i_cross' else block(hidden)
        return self.head(self.ln_f(hidden[:,:self.block_size,:]))
