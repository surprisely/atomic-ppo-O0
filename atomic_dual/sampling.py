"""Complete fixed-H rollouts and persistent, frozen-policy CPU episode workers."""
from __future__ import annotations

import copy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch


@torch.no_grad()
def rollout(env, actor, device, seed, greedy=False, uniform=False):
    from .ppo import masked_categorical

    obs = env.reset(seed=seed)
    horizon = env.T * env.N
    if uniform:
        from .batched_sampling import episode_uniforms, infer_actions
        policy_uniforms = episode_uniforms(seed, horizon)
    # Adjacent transitions share an observation. Store it only once, including
    # a zero terminal observation, and expose the original trajectory API as views.
    observations = np.empty((horizon+1, len(obs)), dtype=np.float32)
    masks = np.empty((horizon, env.action_dim), dtype=np.bool_)
    actions = np.empty(horizon, dtype=np.int64)
    logps = np.empty(horizon, dtype=np.float32)
    rewards = np.empty(horizon, dtype=np.float32)
    dones = np.zeros(horizon, dtype=np.bool_)
    observations[0] = obs
    for index in range(horizon):
        mask = env.action_mask()
        if mask.sum() == 1:
            action = int(np.flatnonzero(mask)[0])
            logp = 0.
            if not greedy and not uniform:
                # Preserve the legacy serial multinomial stream, even for Pass.
                probs = torch.as_tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
                torch.multinomial(probs, 1, replacement=True)
        elif uniform and not greedy:
            sampled = infer_actions(actor, obs[None,:], mask[None,:],
                                    policy_uniforms[index:index+1], device)[0]
            action, logp = int(sampled[0]), float(sampled[1])
        else:
            state = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
            dist = masked_categorical(actor(state), mask_tensor)
            chosen = torch.argmax(dist.probs, dim=-1) if greedy else dist.sample()
            action = int(chosen.item())
            logp = float(dist.log_prob(chosen).item())
        obs, reward, done, info = env.step(action)
        masks[index], actions[index], logps[index] = mask, action, logp
        rewards[index], dones[index] = reward, done
        observations[index+1] = 0. if done else obs
        if done != (index == horizon-1):
            raise AssertionError('The average-reward formula requires exactly T*N steps')
    return dict(states=observations[:-1], next_states=observations[1:],
        masks=masks, actions=actions, logps=logps, rewards=rewards, dones=dones,
        total_profit=float(info['profit']), served=int(info['served']), rejected=int(info['rejected']))


def _initialize_worker(env, obs_dim, hidden_dim):
    from .ppo import MLP
    global _worker_env, _worker_actor
    torch.set_num_threads(1)
    _worker_env = env
    _worker_actor = MLP(obs_dim, env.action_dim, hidden_dim).eval()


def _sample_episode(task):
    weights, seed, uniform = task
    _worker_actor.load_state_dict({k: torch.from_numpy(v) for k,v in weights.items()})
    # A task's random stream never depends on process assignment or completion order.
    torch.manual_seed(seed)
    trajectory = rollout(_worker_env, _worker_actor, 'cpu', seed, uniform=uniform)
    # Pickle would otherwise serialize the overlapping views twice.
    observations = trajectory.pop('states').base
    trajectory.pop('next_states')
    return observations, trajectory


class EpisodeSampler:
    """CPU sampling with one deterministic policy RNG per environment seed.

    The caller updates its policy only after collect() returns all complete
    trajectories. A single sampler/pool is reused over the entire train() call.
    """
    def __init__(self, env, config, workers):
        self.env = copy.deepcopy(env)
        self.config = config
        self.workers = workers
        self.pool = None

    def __enter__(self):
        if self.workers > 1:
            obs_dim = len(self.env.reset(0))
            self.pool = ProcessPoolExecutor(max_workers=self.workers,
                mp_context=mp.get_context('spawn'), initializer=_initialize_worker,
                initargs=(self.env, obs_dim, self.config.hidden_dim))
        return self

    def collect(self, actor, seeds):
        uniform = self.config.rollout_seed_mode == 'uniform'
        if self.pool is None:
            cpu_actor = copy.deepcopy(actor).to('cpu').eval()
            trajectories = []
            # CPU Generator only: leave the learner's CPU/CUDA streams untouched.
            with torch.random.fork_rng(devices=[]):
                for seed in seeds:
                    torch.random.default_generator.manual_seed(seed)
                    trajectories.append(rollout(self.env, cpu_actor, 'cpu', seed, uniform=uniform))
            return trajectories
        weights = {k: v.detach().cpu().numpy().copy() for k,v in actor.state_dict().items()}
        trajectories = []
        for observations, trajectory in self.pool.map(_sample_episode, ((weights,s,uniform) for s in seeds)):
            trajectory['states'] = observations[:-1]
            trajectory['next_states'] = observations[1:]
            trajectories.append(trajectory)
        return trajectories

    def __exit__(self, exc_type, exc, tb):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
