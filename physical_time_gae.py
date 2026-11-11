"""GAE trace decay once per physical minute; gamma remains 1 per atomic step."""
from contextlib import contextmanager
import numpy as np
import torch
import atomic_dual.ppo as ppo
from advantage_ablation import actor_advantage


def minute_trace_factors(dones,fleet_size,minute_lambda):
    if fleet_size<1 or not 0<=minute_lambda<=1: raise ValueError('Invalid fleet size or minute lambda')
    terminal=np.asarray(dones,dtype=bool)
    if terminal.ndim!=1 or not len(terminal) or not terminal[-1]:
        raise ValueError('Requires complete terminal trajectories')
    ends=np.flatnonzero(terminal)+1
    lengths=np.diff(np.r_[0,ends])
    if np.any(lengths%fleet_size): raise ValueError('Episode horizon does not contain complete fleet rounds')
    factors=np.ones(len(terminal),dtype=np.float64)
    # Every episode starts at fleet cursor 0 and contains all forced busy Pass.
    factors[(np.arange(len(terminal))+1)%fleet_size==0]=minute_lambda
    factors[terminal]=0.
    return factors


def minute_gae_from_td(td,dones,fleet_size,minute_lambda=.95):
    delta=td.detach().cpu().numpy()
    factors=minute_trace_factors(dones.detach().cpu().numpy(),fleet_size,minute_lambda)
    if delta.shape!=factors.shape: raise ValueError('TD/done shape mismatch')
    result=np.empty_like(delta); carry=0.
    for i in range(len(delta)-1,-1,-1):
        carry=float(delta[i])+factors[i]*carry
        result[i]=carry
    return torch.as_tensor(result,device=td.device)


@contextmanager
def physical_advantage(method,fleet_size,minute_lambda=.95):
    if method!='minute_gae':
        with actor_advantage(method): yield
        return
    original=ppo.td_advantages
    def replacement(rewards,baseline,values,next_values,dones):
        td=original(rewards,baseline,values,next_values,dones)
        return minute_gae_from_td(td,dones,fleet_size,minute_lambda)
    ppo.td_advantages=replacement
    try: yield
    finally: ppo.td_advantages=original
