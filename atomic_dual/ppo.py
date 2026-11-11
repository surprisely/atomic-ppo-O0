"""Average-reward Atomic-PPO for independent, reset two-hour operating periods.

This implements the primary equations of Atomic-PPO for a fixed H=T*N atoms:
  gain = mean over rollout cycles of total cycle reward
  r_bar = gain / H
  critic MC target = sum from current atom to end of (r - r_bar)
  advantage = (r - r_bar) + h(next) - h(current)
  clipped actor PPO uses masked old/new action probabilities consistently.
At the episode boundary, the continuation value is anchored to zero. This is
consistent with the reset-cycle gain objective, not a claim of steady-state
continuation through an unmodeled 22-hour interval.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import platform
import random
import time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from .environment import DualServiceEnv


@dataclass(frozen=True)
class PPOConfig:
    hidden_dim: int = 128
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    clip_epsilon: float = 0.2
    actor_epochs: int = 3
    critic_epochs: int = 4
    minibatch_size: int = 256
    reward_scale: float = 100.0
    entropy_coef: float = 0.005
    max_grad_norm: float = 1.0
    device: str = 'cpu'
    objective: str = 'centered'
    rollout_workers: int = 1
    rollout_seed_mode: str = 'auto'
    rollout_backend: str = 'episode'
    rollout_device: str = 'cpu'
    rollout_batch_size: int = 30
    rollout_tensor_graph: bool = True
    rollout_travel_rng: str = 'step'

    def __post_init__(self):
        if not isinstance(self.rollout_workers, int) or self.rollout_workers < 1:
            raise ValueError('rollout_workers must be a positive integer')
        if self.rollout_seed_mode not in ('auto', 'legacy', 'episode', 'uniform'):
            raise ValueError('Unknown rollout seed mode')
        if self.rollout_backend not in ('episode','batched','tensor'):
            raise ValueError('Unknown rollout backend')
        if self.rollout_travel_rng not in ('step','event'):
            raise ValueError('Unknown Tensor travel RNG')
        if self.rollout_backend!='tensor' and self.rollout_travel_rng!='step':
            raise ValueError('rollout_travel_rng is a Tensor-backend option')
        if not isinstance(self.rollout_batch_size,int) or self.rollout_batch_size < 1:
            raise ValueError('rollout_batch_size must be positive')
        if torch.device(self.rollout_device).type not in ('cpu','cuda'):
            raise ValueError('rollout_device must be cpu or cuda')
        if self.rollout_backend in ('batched','tensor'):
            if self.rollout_workers != 1:
                raise ValueError('Batched rollout uses one central environment coordinator; set rollout_workers=1')
            if self.rollout_seed_mode not in ('auto','uniform'):
                raise ValueError('Batched rollout cannot use legacy/episode Torch RNG; select auto or uniform')
        elif self.rollout_device != 'cpu':
            raise ValueError('GPU sampling requires rollout_backend=batched')
        if self.rollout_workers > 1 and self.rollout_seed_mode == 'legacy':
            raise ValueError('Parallel sampling cannot preserve the legacy global RNG stream; use episode seeding')
        if self.objective not in ('centered','episodic'): raise ValueError('Unknown reward objective')
        if any(not np.isfinite(v) or v < 0 for v in (self.actor_lr,self.critic_lr,self.entropy_coef)):
            raise ValueError('Learning rates and entropy coefficient must be finite and nonnegative')
        if self.reward_scale <= 0 or not np.isfinite(self.reward_scale) or self.max_grad_norm <= 0 or not np.isfinite(self.max_grad_norm):
            raise ValueError('Positive finite reward scale and gradient bound required')
        if not 0 < self.clip_epsilon < 1:
            raise ValueError('clip_epsilon must be in (0,1)')
        if any(not isinstance(v,int) or v < 1 for v in (self.hidden_dim,self.actor_epochs,self.critic_epochs,self.minibatch_size)):
            raise ValueError('Network size, epochs and minibatch size must be positive integers')


def td_advantages(rewards, baseline, values, next_values, dones):
    return rewards - baseline + torch.where(dones, torch.zeros_like(next_values), next_values) - values


def normalize_actor_advantages(advantages, masks):
    active = masks.sum(-1) > 1
    result = torch.zeros_like(advantages)
    if active.any():
        selected = advantages[active]
        result[active] = (selected-selected.mean())/(selected.std(unbiased=False)+1e-8)
    return result


class MLP(nn.Module):
    def __init__(self, inputs: int, outputs: int, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inputs, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def masked_categorical(logits: torch.Tensor, masks: torch.Tensor) -> Categorical:
    """All probability calculations, both sampling and PPO, use the SAME mask."""
    if masks.shape != logits.shape:
        raise ValueError('Action mask must have shape identical to logits')
    if not torch.all(masks.any(dim=-1)):
        raise ValueError('Every state must have at least one feasible action')
    return Categorical(logits=logits.masked_fill(~masks.bool(), -torch.inf))


class AverageRewardPPO:
    def __init__(self, env: DualServiceEnv, config: PPOConfig = PPOConfig(), seed: int = 0):
        self.env = env
        self.cfg = config
        self.device = torch.device(config.device)
        torch.manual_seed(seed)
        obs = env.reset(seed)
        self.actor = MLP(len(obs), env.action_dim, config.hidden_dim).to(self.device)
        self.critic = MLP(len(obs), 1, config.hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.iteration = 0
        self.training_schedule = None

    def rollout(self, seed: int, greedy: bool = False) -> dict:
        from .sampling import rollout
        return rollout(self.env, self.actor, self.device, seed, greedy)

    def _tensor(self, arr, dtype=torch.float32):
        return torch.as_tensor(arr, dtype=dtype, device=self.device)

    def update(self, trajectories: list[dict]) -> dict:
        if not trajectories:
            raise ValueError('At least one trajectory is required')
        H = self.env.N * self.env.T
        for tr in trajectories:
            if (len(tr['rewards']) != H or not np.isfinite(tr['rewards']).all()
                    or not np.isclose(np.sum(tr['rewards'], dtype=np.float64), tr['total_profit'], rtol=1e-5, atol=1e-4)
                    or not tr['dones'][-1] or np.any(tr['dones'][:-1])):
                raise ValueError('Expected complete finite fixed-H trajectories with consistent profit')
        # Gain is defined per 2-hour operating period. Scale ALL rewards and gain alike.
        gains_raw = np.asarray([tr['total_profit'] for tr in trajectories], dtype=np.float64)
        gain_raw = float(np.mean(gains_raw))
        scaled_gain_per_atom = gain_raw / (H * self.cfg.reward_scale) if self.cfg.objective=='centered' else 0.
        critic_targets = []
        for tr in trajectories:
            adjusted = tr['rewards'] / self.cfg.reward_scale - scaled_gain_per_atom
            critic_targets.append(np.cumsum(adjusted[::-1])[::-1].copy())
        states = self._tensor(np.concatenate([tr['states'] for tr in trajectories]))
        masks = self._tensor(np.concatenate([tr['masks'] for tr in trajectories]), dtype=torch.bool)
        actions = self._tensor(np.concatenate([tr['actions'] for tr in trajectories]), dtype=torch.long)
        old_logps = self._tensor(np.concatenate([tr['logps'] for tr in trajectories]))
        rewards = self._tensor(np.concatenate([tr['rewards'] for tr in trajectories])) / self.cfg.reward_scale
        dones = self._tensor(np.concatenate([tr['dones'] for tr in trajectories]), dtype=torch.bool)
        targets = self._tensor(np.concatenate(critic_targets))
        with torch.no_grad():
            before=self.critic(states).squeeze(-1)
            critic_mse_before=float(((before-targets)**2).mean())
        count = states.shape[0]
        batch_size = max(1, min(self.cfg.minibatch_size, count))
        critic_losses = []
        for _ in range(self.cfg.critic_epochs):
            indices = torch.randperm(count, device=self.device)
            for start in range(0, count, batch_size):
                batch_idx = indices[start:start + batch_size]
                predicted = self.critic(states[batch_idx]).squeeze(-1)
                loss = torch.mean((predicted - targets[batch_idx]) ** 2)
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                self.critic_optimizer.step()
                critic_losses.append(float(loss.item()))
        # Atomic-PPO eq. (15): r - g/(TN) + h(next) - h(current).
        with torch.no_grad():
            values = self.critic(states).squeeze(-1)
            # Avoid a second full concatenated observation matrix (several GiB
            # at K=30). Keep exactly the same supplied next-state observations.
            next_values = torch.cat([self.critic(self._tensor(tr['next_states'])).squeeze(-1)
                                     for tr in trajectories])
            next_values = torch.where(dones, torch.zeros_like(next_values), next_values)
            advantages = td_advantages(rewards, scaled_gain_per_atom, values, next_values, dones)
            advantages = normalize_actor_advantages(advantages, masks)
        actor_indices = torch.nonzero(masks.sum(-1) > 1).flatten()
        actor_count = len(actor_indices)
        actor_losses = []
        diagnostics = []
        self.actor_optimizer.zero_grad(set_to_none=True)
        for _ in range(self.cfg.actor_epochs):
            indices = actor_indices[torch.randperm(actor_count, device=self.device)]
            for start in range(0, actor_count, batch_size):
                batch_idx = indices[start:start + batch_size]
                dist = masked_categorical(self.actor(states[batch_idx]), masks[batch_idx])
                logp = dist.log_prob(actions[batch_idx])
                ratio = torch.exp(logp - old_logps[batch_idx])
                adv = advantages[batch_idx]
                surrogate = torch.minimum(
                    ratio * adv,
                    torch.clamp(ratio, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon) * adv,
                )
                loss = -surrogate.mean() - self.cfg.entropy_coef * dist.entropy().mean()
                with torch.no_grad():
                    logratio = logp-old_logps[batch_idx]
                    diagnostics.append((len(batch_idx), float(dist.entropy().mean()),
                        float(((ratio-1)-logratio).mean()), float(((ratio-1).abs()>self.cfg.clip_epsilon).float().mean())))
                self.actor_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                self.actor_optimizer.step()
                actor_losses.append(float(loss.item()))
        return {
            'reward_baseline_per_atom':scaled_gain_per_atom,
            'profit_std':float(gains_raw.std(ddof=1)) if len(gains_raw)>1 else 0.,
            'profit_min':float(gains_raw.min()),'profit_max':float(gains_raw.max()),
            'critic_mse_before':critic_mse_before,
            'critic_mse_after':float(((values-targets)**2).mean()),
            'critic_explained_variance':float(1-((targets-values).var(unbiased=False)/(targets.var(unbiased=False)+1e-8))),
            'gain_estimate_per_period': gain_raw,
            'mean_served': float(np.mean([tr['served'] for tr in trajectories])),
            'critic_loss': float(np.mean(critic_losses)),
            'actor_loss': float(np.mean(actor_losses)) if actor_losses else 0.,
            'num_atomic_samples': count,
            'num_actor_samples': actor_count,
            'forced_action_fraction': 1-actor_count/count,
            'actor_optimizer_steps': len(actor_losses),
            **{name: (sum(row[0]*row[i] for row in diagnostics)/sum(row[0] for row in diagnostics) if diagnostics else 0.)
               for i,name in enumerate(('actor_entropy','approx_kl','clip_fraction'),start=1)},
        }

    def train(self, iterations: int, episodes_per_iteration: int, seed: int = 0, on_iteration=None) -> list[dict]:
        if iterations < 0 or episodes_per_iteration < 1:
            raise ValueError('Nonnegative iterations and positive episodes required')
        schedule = {'seed':seed,'episodes_per_iteration':episodes_per_iteration}
        batched = self.cfg.rollout_backend in ('batched','tensor')
        tensor_env = self.cfg.rollout_backend == 'tensor'
        uniform = batched or self.cfg.rollout_seed_mode == 'uniform'
        episode_seeded = uniform or self.cfg.rollout_seed_mode == 'episode' or (
            self.cfg.rollout_seed_mode == 'auto' and self.cfg.rollout_workers > 1)
        if episode_seeded:
            schedule['sampling_contract'] = 'episode_cpu_v1'
        if uniform:
            schedule['sampling_contract'] = 'episode_uniform_v1'
            schedule['rollout_backend'] = self.cfg.rollout_backend
        if batched:
            schedule.update(rollout_device=self.cfg.rollout_device,rollout_batch_size=self.cfg.rollout_batch_size)
        if tensor_env:
            schedule.update(travel_noise_contract='event_keyed_v1' if self.cfg.rollout_travel_rng=='event' else 'tensor_step_uniform_v1',
                            rollout_tensor_graph=self.cfg.rollout_tensor_graph)
        if self.training_schedule is not None and self.training_schedule != schedule:
            raise ValueError('Training schedule differs from saved schedule')
        if iterations:
            self.training_schedule = schedule
        results = []
        from .sampling import EpisodeSampler
        from .batched_sampling import BatchedEpisodeSampler
        from .tensor_sampling import TensorEpisodeSampler
        context = (TensorEpisodeSampler(self.env,self.cfg) if tensor_env and iterations else
                   BatchedEpisodeSampler(self.env,self.cfg) if batched and iterations else
                   EpisodeSampler(self.env, self.cfg, min(self.cfg.rollout_workers, episodes_per_iteration))
                   if episode_seeded and iterations else nullcontext())
        with context as sampler:
            for _ in range(iterations):
                started = time.perf_counter()
                seeds = [seed + self.iteration * episodes_per_iteration + episode
                         for episode in range(episodes_per_iteration)]
                trajs = (sampler.collect(self.actor, seeds) if episode_seeded else
                         [self.rollout(s) for s in seeds])
                sampled = time.perf_counter()
                report = self.update(trajs)
                self.last_timings = dict(sampling_seconds=sampled-started,
                                         update_seconds=time.perf_counter()-sampled)
                self.last_sampling_stats = getattr(sampler,'last_stats',{})
                del trajs  # release large previous batches before collecting the next one
                self.iteration += 1
                report['iteration'] = self.iteration
                results.append(report)
                if on_iteration is not None:
                    on_iteration(report)
        return results

    def environment_fingerprint(self, include_replay=True):
        import json
        digest = hashlib.sha256(json.dumps(asdict(self.env.cfg), sort_keys=True).encode())
        if self.env.research_settings():
            digest.update(json.dumps(self.env.research_settings(), sort_keys=True).encode())
        for name in ('arrival_rates','trip_minutes','relocation_minutes','fares','relocation_costs','initial_regions') + (('request_batches',) if include_replay else ()):
            value = getattr(self.env,name)
            digest.update(name.encode())
            if value is not None:
                arr = np.ascontiguousarray(value)
                digest.update(str((arr.shape,arr.dtype.str)).encode())
                digest.update(arr.tobytes())
        return digest.hexdigest()

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(schema_version=1, config=asdict(self.cfg), env_config=asdict(self.env.cfg),
            environment_fingerprint=self.environment_fingerprint(), calibration_fingerprint=self.environment_fingerprint(False),
            seed=self.seed, iteration=self.iteration, training_schedule=self.training_schedule,
            actor=self.actor.state_dict(),critic=self.critic.state_dict(),
            actor_optimizer=self.actor_optimizer.state_dict(),critic_optimizer=self.critic_optimizer.state_dict(),
            rng=dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                numpy=self.rng.bit_generator.state, env=self.env.rng.bit_generator.state,
                python=random.getstate(), numpy_global=np.random.get_state()),
            versions=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__),
            source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                           for name in ('ppo.py','environment.py','sampling.py','batched_sampling.py',
                                        'tensor_environment.py','tensor_sampling.py','event_rng.py','event_tables.py')})
        # Exclusive creation prevents silently replacing prior research checkpoints.
        with path.open('xb') as f:
            torch.save(payload,f)

    @classmethod
    def from_checkpoint(cls, env, path, device='cpu', for_evaluation=False):
        # Only load locally trusted checkpoints: optimizer and RNG states need pickle.
        saved = torch.load(path,map_location='cpu',weights_only=False)
        if saved.get('schema_version') != 1:
            raise ValueError('Unsupported checkpoint schema')
        cfg = dict(saved['config'],device=device)
        trainer = cls(env,PPOConfig(**cfg),seed=saved['seed'])
        key = 'calibration_fingerprint' if for_evaluation else 'environment_fingerprint'
        if saved[key] != trainer.environment_fingerprint(not for_evaluation):
            raise ValueError('Checkpoint environment fingerprint mismatch')
        for name in ('actor','critic','actor_optimizer','critic_optimizer'):
            getattr(trainer,name).load_state_dict(saved[name])
        trainer.iteration, trainer.training_schedule = saved['iteration'],saved['training_schedule']
        rng = saved['rng']
        trainer.rng.bit_generator.state = rng['numpy']
        trainer.env.rng.bit_generator.state = rng['env']
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy_global'])
        torch.set_rng_state(rng['torch'])
        if device.startswith('cuda') and rng['cuda']:
            torch.cuda.set_rng_state_all(rng['cuda'])
        return trainer

    def evaluate(self, seed: int = 1000, greedy: bool = True) -> dict:
        tr = self.rollout(seed=seed, greedy=greedy)
        return {key: tr[key] for key in ('total_profit', 'served', 'rejected')}
