"""Temporary actor-advantage intervention; production critic/optimizer code unchanged."""
from contextlib import contextmanager
import numpy as np
import torch
import atomic_dual.ppo as ppo


def compute_advantage(method,rewards,baseline,values,next_values,dones):
    td=ppo.td_advantages(rewards,baseline,values,next_values,dones)
    if method=='td0': return td
    if method=='mc':
        adjusted=(rewards-baseline).detach().cpu().numpy()
        terminal=dones.detach().cpu().numpy()
        target=np.empty_like(adjusted); end=0
        for end_idx in np.flatnonzero(terminal):
            target[end:end_idx+1]=np.cumsum(adjusted[end:end_idx+1][::-1],dtype=adjusted.dtype)[::-1]
            end=end_idx+1
        if end!=len(adjusted): raise ValueError('MC requires complete terminal trajectories')
        return torch.as_tensor(target,device=values.device)-values
    if method!='gae095': raise ValueError(method)
    delta=td.detach().cpu().numpy(); terminal=dones.detach().cpu().numpy()
    result=np.empty_like(delta); carry=0.
    if not terminal[-1]: raise ValueError('GAE requires complete terminal trajectories')
    for i in range(len(delta)-1,-1,-1):
        carry=float(delta[i])+.95*(0. if terminal[i] else carry)
        result[i]=carry
    return torch.as_tensor(result,device=values.device)


@contextmanager
def actor_advantage(method):
    """Only intercept the actor advantage, AFTER the unchanged critic fit.

    The MC branch uses complete centered suffix returns; normalization is still
    performed by production update. No critic or optimizer implementation changes.
"""
    if method=='td0':
        yield
        return
    original=ppo.td_advantages
    def replacement(rewards,baseline,values,next_values,dones):
        # compute_advantage calls td_advantages internally: avoid recursive hook.
        ppo.td_advantages=original
        try: return compute_advantage(method,rewards,baseline,values,next_values,dones)
        finally: ppo.td_advantages=replacement
    ppo.td_advantages=replacement
    try: yield
    finally: ppo.td_advantages=original
