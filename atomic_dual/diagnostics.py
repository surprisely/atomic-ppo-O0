"""Diagnostic value estimates and one-step probes; never an optimal-policy oracle."""
import copy
import numpy as np
import torch
from .baselines import choose_action


@torch.no_grad()
def critic_error(trainer,trajectory):
    rewards=np.asarray(trajectory['rewards'],dtype=np.float64)/trainer.cfg.reward_scale
    center=float(rewards.mean()) if trainer.cfg.objective=='centered' else 0.
    targets=np.cumsum((rewards-center)[::-1])[::-1].copy()
    predictions=trainer.critic(torch.as_tensor(trajectory['states'],dtype=torch.float32,device=trainer.device)).squeeze(-1).cpu().numpy()
    return dict(mse=float(np.mean((predictions-targets)**2)),mae=float(np.mean(np.abs(predictions-targets))),
        target_std=float(targets.std()),prediction_std=float(predictions.std()),
        explained_variance=float(1-np.var(targets-predictions)/(np.var(targets)+1e-8)),
        target='within_trajectory_centered_MC' if trainer.cfg.objective=='centered' else 'episodic_MC')


@torch.no_grad()
def action_probe(env,critic,reward_scale,baseline=0.,seed=7):
    """Score legal actions at a fixed state with independent forecast arrivals.

    Compare to immediate reward and a dedicated-first continuation only until
    the current batch closes. Neither is the true long-horizon action value.
    """
    forecast=np.random.default_rng(seed).poisson(env.arrival_rates)
    actions=[]
    device=next(critic.parameters()).device
    for action in np.flatnonzero(env.action_mask()):
        branch=copy.deepcopy(env)
        branch.request_batches=forecast.copy()
        obs,reward,done,_=branch.step(int(action))
        value=0. if done else float(critic(torch.as_tensor(obs,dtype=torch.float32,device=device)).squeeze())
        local_reward=reward
        while not branch.done and branch.t==env.t:
            _,r,_,_=branch.step(choose_action('compatible_stay',branch))
            local_reward+=r
        actions.append(dict(action=int(action),immediate_reward=reward,
            critic_score=reward/reward_scale-baseline+value,same_batch_continuation_reward=local_reward))
    ranked=max(actions,key=lambda a:(a['critic_score'],-a['action']))
    local=max(actions,key=lambda a:(a['same_batch_continuation_reward'],-a['action']))
    return dict(time_step=env.t,atomic_cursor=env._cursor,actions=actions,
        critic_preferred_action=ranked['action'],same_batch_preferred_action=local['action'],
        agrees_with_same_batch=ranked['same_batch_continuation_reward']==local['same_batch_continuation_reward'])


def probe_rollout(env,critic,reward_scale,seed=7,max_probes=4):
    env.reset(seed);probes=[]
    while not env.done and len(probes)<max_probes:
        mask=env.action_mask()
        if mask[:3*env.Z].any() and mask.sum()>1:
            probes.append(action_probe(env,critic,reward_scale,seed=seed+len(probes)))
        env.step(choose_action('compatible_stay',env))
    return dict(probes=probes,n=len(probes),same_batch_agreement=float(np.mean([p['agrees_with_same_batch'] for p in probes])) if probes else None,
                limitation='Forecast one-step critic ranking vs same-batch heuristic continuation; not long-horizon optimality. Stochastic travel uses one keyed realization per action, not an expected action value.')
