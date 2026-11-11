"""Cross-environment Actor batching; each environment remains sequential."""
from __future__ import annotations

import copy
import time
import numpy as np
import torch


def episode_uniforms(seed, horizon):
    """Independent policy stream, indexed by *every* atomic step, including Pass."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x4150504F]))
    return rng.random(horizon, dtype=np.float32)


@torch.no_grad()
def infer_actions(actor, states, masks, uniforms, device):
    """Masked categorical inverse CDF with one externally supplied U[0,1) per row.

    No global Torch/CUDA RNG is consumed, so batch scheduling cannot reassign
    another episode's random draws. Float32 CPU/GPU results need not be bitwise equal.
    """
    x = torch.as_tensor(states, dtype=torch.float32, device=device)
    legal = torch.as_tensor(masks, dtype=torch.bool, device=device)
    logits = actor(x).masked_fill(~legal, -torch.inf)
    logps = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    cdf = logps.exp().cumsum(dim=-1)
    # Normalize rounding at the endpoint without giving masked trailing actions mass.
    cdf = cdf / cdf[:, -1:]
    u = torch.as_tensor(uniforms, dtype=torch.float32, device=device).unsqueeze(-1)
    actions = (u >= cdf).sum(dim=-1)
    selected_logps = logps.gather(1, actions.unsqueeze(-1)).squeeze(-1)
    # One D2H transfer/synchronization for the entire batch, not per environment.
    return torch.stack((actions.float(), selected_logps), dim=-1).cpu().numpy()


class _Episode:
    def __init__(self, template, seed):
        self.env = copy.deepcopy(template)
        obs = self.env.reset(seed)
        self.horizon = self.env.T*self.env.N
        self.index = 0
        self.observations = np.empty((self.horizon+1, len(obs)), dtype=np.float32)
        self.observations[0] = obs
        self.masks = np.empty((self.horizon, self.env.action_dim), dtype=np.bool_)
        self.actions = np.empty(self.horizon, dtype=np.int64)
        self.logps = np.empty(self.horizon, dtype=np.float32)
        self.rewards = np.empty(self.horizon, dtype=np.float32)
        self.dones = np.zeros(self.horizon, dtype=np.bool_)
        self.uniforms = episode_uniforms(seed, self.horizon)

    def step(self, mask, action, logp):
        obs, reward, done, self.info = self.env.step(int(action))
        i = self.index
        self.masks[i], self.actions[i], self.logps[i] = mask, action, logp
        self.rewards[i], self.dones[i] = reward, done
        self.observations[i+1] = 0. if done else obs
        self.index += 1
        if done != (self.index == self.horizon):
            raise AssertionError('Expected a complete fixed T*N episode')

    def next_decision(self):
        # Only skip *network inference*. All forced transitions, rewards, next
        # observations and physical-time boundaries remain in the trajectory.
        while not self.env.done:
            mask = self.env.action_mask()
            if mask.sum() > 1:
                return mask
            self.step(mask, int(np.flatnonzero(mask)[0]), 0.)
        return None

    def trajectory(self):
        return dict(states=self.observations[:-1], next_states=self.observations[1:],
            masks=self.masks, actions=self.actions, logps=self.logps, rewards=self.rewards,
            dones=self.dones, total_profit=float(self.info['profit']),
            served=int(self.info['served']), rejected=int(self.info['rejected']))


class BatchedEpisodeSampler:
    """One frozen Actor and up to batch_size independent CPU environments.

    Advancing only to the next non-forced action avoids GPU calls for busy cars
    and allows different environments to be at different atomic/physical times.
    No actions inside a single environment are computed from stale observations.
    """
    def __init__(self, env, config):
        self.env = env
        self.device = torch.device(config.rollout_device)
        self.batch_size = config.rollout_batch_size
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA rollout requested but CUDA is unavailable')

    def __enter__(self):
        return self

    @torch.no_grad()
    def collect(self, actor, seeds):
        frozen = copy.deepcopy(actor).to(self.device).eval()
        trajectories = []
        self.last_stats = dict(actor_forward_calls=0, actor_inference_samples=0,
                               max_inference_batch=0,actor_inference_seconds=0.)
        for offset in range(0, len(seeds), self.batch_size):
            episodes = [_Episode(self.env, seed) for seed in seeds[offset:offset+self.batch_size]]
            while True:
                pending = []
                for ep in episodes:
                    mask = ep.next_decision()
                    if mask is not None:
                        pending.append((ep, mask))
                if not pending:
                    break
                states = np.stack([ep.observations[ep.index] for ep,_ in pending])
                masks = np.stack([mask for _,mask in pending])
                uniforms = np.asarray([ep.uniforms[ep.index] for ep,_ in pending], dtype=np.float32)
                inference_start=time.perf_counter()
                output = infer_actions(frozen, states, masks, uniforms, self.device)
                self.last_stats['actor_inference_seconds'] += time.perf_counter()-inference_start
                self.last_stats['actor_forward_calls'] += 1
                self.last_stats['actor_inference_samples'] += len(pending)
                self.last_stats['max_inference_batch'] = max(self.last_stats['max_inference_batch'],len(pending))
                for (ep,mask), (action,logp) in zip(pending,output):
                    ep.step(mask, int(action), float(logp))
            trajectories.extend(ep.trajectory() for ep in episodes)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        return trajectories

    def __exit__(self, exc_type, exc, tb):
        pass
