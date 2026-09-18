"""CS-BFM with one-step detached clean-logit memory and the unchanged BFM bridge."""
from __future__ import annotations
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from models.binarylatent_flow_expectation_consistent_retrain_tminus1 import BinaryDiffusionFlowDecouple
from experiments.ICLR27.new_prediction_target_9_16.cs_bfm_fromscratch.binarylatent_flow_cs_decomposed import (
    analytic_self_evidence,
)

class BinaryDiffusionFlowAdjacentLogitMemory(BinaryDiffusionFlowDecouple):
    adjacent_logit_memory=True
    def __init__(self,H,denoise_fn,mask_id):
        super().__init__(H,denoise_fn,mask_id)
        if bool(H.p_flip) or float(H.focal)>=0 or float(H.aux)!=0: raise ValueError('requires p_flip=False, focal=-1, aux=0')
        if str(H.loss_final)!='mean' or int(H.total_steps)!=64: raise ValueError('requires loss_final=mean,T=64')
        if float(getattr(self,'barrier_retention_scale',1.0))!=1.0: raise ValueError('requires original linear path')
        self.previous_loss_weight=0.5
        self.recurrent_config={'memory':'previous full z is a detached condition only','train':'coherent adjacent pair; at most two Transformer forwards','loss':'BCE(z_t,X0)+0.5*BCE(z_tplus1,X0)','bridge':'unchanged expectation-consistent B0/B1'}
        self.last_sampling_diagnostics=[]

    def context_logits(self,x_t,prev_z,physical_time,label=None):
        return self._denoise_fn(x_t,prev_clean_logit=prev_z,label=label,time_steps=physical_time.long()-1)

    def full_logits(self,x_t,prev_z,physical_time,label=None)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        c=self.context_logits(x_t,prev_z,physical_time,label=label).float()
        d=analytic_self_evidence(physical_time,total_steps=self.num_timesteps,reference=c)
        z=c+(2.0*x_t.float()-1.0)*d
        return c,d,z

    def _one_step_forward_probability(self,x_t,t,t_next):
        tau=self.interpolation_t.to(x_t.device); tau_t=tau[t]; tau_next=tau[t_next]
        gamma=0.5*(tau_t-tau_next)/(tau_t+self.posterior_eps)
        gamma=gamma.view(-1,*([1]*(x_t.ndim-1))).clamp(0,0.5)
        return x_t.float()*(1-gamma)+(1-x_t.float())*gamma

    def coherent_adjacent_pair(self,x0,t):
        x_t=torch.bernoulli(self.q_sample(x0.float(),t))
        has_prev=t<self.num_timesteps
        x_next=torch.zeros_like(x_t)
        if bool(has_prev.any()):
            tn=t[has_prev]+1
            p=self._one_step_forward_probability(x_t[has_prev],t[has_prev],tn)
            x_next[has_prev]=torch.bernoulli(p)
        return x_t,x_next,has_prev

    def _train_loss(self,x_0,label=None,x_ct=None):
        if x_ct is not None: raise NotImplementedError
        x_0=x_0.float(); b=x_0.shape[0]; device=x_0.device
        t=self.sample_time(b,device); x_t,x_next,has_prev=self.coherent_adjacent_pair(x_0,t)
        prev_z=torch.zeros_like(x_0)
        if bool(has_prev.any()):
            subset_label=label[has_prev] if label is not None else None
            zeros=torch.zeros_like(x_next[has_prev])
            _,_,z_next=self.full_logits(x_next[has_prev],zeros,t[has_prev]+1,label=subset_label)
            previous_loss=F.binary_cross_entropy_with_logits(z_next,x_0[has_prev],reduction='mean')
            # The current loss must not turn z_(t+1) into a hidden communication code.
            prev_z[has_prev]=z_next.detach()
            prev_hard_ber=((z_next.detach()>0)!=x_0[has_prev].bool()).float().mean()
        else:
            previous_loss=torch.zeros((),device=device)
            prev_hard_ber=torch.zeros((),device=device)
        c,d,z=self.full_logits(x_t,prev_z.detach(),t,label=label)
        current_loss=F.binary_cross_entropy_with_logits(z,x_0,reduction='mean')
        total=current_loss+self.previous_loss_weight*previous_loss
        hard=z.detach()>0
        stats={'loss':total,'current_bce':current_loss.detach(),'previous_bce':previous_loss.detach(),'current_hard_ber':(hard!=x_0.bool()).float().mean(),'previous_hard_ber':prev_hard_ber,'hard_x0_acc':(hard==x_0.bool()).float().mean(),'acc':(hard==x_0.bool()).float().mean(),'copy_rate':(hard==x_t.bool()).float().mean(),'mean_abs_c':c.detach().abs().mean(),'mean_abs_z':z.detach().abs().mean(),'mean_abs_prev_z':prev_z.detach().abs().mean(),'prev_available_rate':has_prev.float().mean(),'mean_d':d.detach().mean()}
        return stats

    @torch.no_grad()
    def sample(self,temp=1.0,sample_steps=None,b=8,shape=None,return_all=False,label=None,mask=None,guidance=None,full=False):
        del full
        if guidance is not None: raise NotImplementedError
        if temp<=0: raise ValueError('temperature must be positive')
        device=next(self._denoise_fn.parameters()).device
        if shape is None: shape=(b,int(np.prod(self.shape)),self.codebook_size)
        else: b=int(shape[0])
        x_t=torch.bernoulli(torch.full(shape,.5,device=device)); prev_z=torch.zeros_like(x_t)
        if mask is not None:
            mt=mask['mask'].unsqueeze(0).to(device); latent=mask['latent'].unsqueeze(0).to(device); x_t=latent*mt+x_t*(1-mt)
        sample_steps=self.num_timesteps if sample_steps is None else int(sample_steps)
        steps=np.arange(1,self.num_timesteps+1)
        if sample_steps!=self.num_timesteps: steps=steps[np.linspace(0,self.num_timesteps-1,sample_steps).astype(np.int64)]
        steps=steps[::-1]; all_states=[x_t] if return_all else None; diagnostics=[]
        for i,tv in enumerate(steps):
            pt=torch.full((b,),int(tv),device=device,dtype=torch.long)
            c,d,z=self.full_logits(x_t,prev_z,pt,label=label); z=z/float(temp); clean=torch.sigmoid(z)
            if int(tv)!=1:
                target=torch.full((b,),int(steps[i+1]),device=device,dtype=torch.long)
                x_prob=self._reverse_probability(clean,x_t,pt,target); x_next=torch.bernoulli(x_prob)
            elif self.hard_final: x_next=(clean>.5).float()
            else: x_next=torch.bernoulli(clean)
            if mask is not None: x_next=latent*mt+x_next*(1-mt)
            diagnostics.append({'physical_t':int(tv),'mean_abs_prev_z':float(prev_z.abs().mean()),'mean_abs_c':float(c.abs().mean()),'mean_abs_z':float(z.abs().mean()),'hard_z_change_rate':float(((z>0)!=(prev_z>0)).float().mean()),'Xt_change_rate':float((x_next!=x_t).float().mean())})
            prev_z=z; x_t=x_next
            if return_all: all_states.append(x_t)
        self.last_sampling_diagnostics=diagnostics
        return torch.cat(all_states,dim=0) if return_all else x_t
