"""One-period, batched-dispatch environment with sequential atomic vehicle decisions.

The environment has complete per-vehicle state. Policy observations are aggregated.
Each episode has exactly horizon_steps * n_vehicles atomic transitions.

Deliberate modeling choices for the first implementation:
- All requests of a step are revealed before the atomic decisions at that step.
- Unserved requests are rejected at the end of that step (zero inter-step patience).
- Fare/repositioning cost is booked when an action starts, including near the horizon.
- Trips and relocations are noninterruptible. After arrival, an idle vehicle may
  select a new action in a later time step.
- Legacy inputs reset to fixed positions; the ADP profile redraws remainder counts.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
import numpy as np
import math
import random


def draw_adp_initial_regions(express, premium, zones, seed):
    """Original uniform base counts plus independent random remainder placements."""
    rng = random.Random(int(seed))
    regions = []
    for n in (express, premium):
        counts = np.full(zones, n // zones, dtype=np.int64)
        for _ in range(n % zones):
            counts[rng.randint(0, zones - 1)] += 1
        regions.extend(np.repeat(np.arange(zones), counts))
    return np.asarray(regions, dtype=np.int64)


class ServiceClass(IntEnum):
    EXPRESS = 0
    PREMIUM = 1


class CarStatus(IntEnum):
    IDLE = 0
    OCCUPIED = 1
    RELOCATING = 2


# Order type indices: 0=Express-only; 1=Premium-only; 2=Dual-compatible.
ORDER_TYPE_NAMES = ("express_only", "premium_only", "dual_compatible")


@dataclass(frozen=True)
class EnvConfig:
    horizon_steps: int = 120
    step_minutes: float = 1.0
    n_express: int = 240
    n_premium: int = 60
    random_order: bool = True
    max_eta_feature: int = 8  # ETA buckets for actor/critic; full ETA retained internally.
    batch_timing: str = 'opening'  # calibrated inputs use closing-window boundaries

    def __post_init__(self) -> None:
        if any(not isinstance(v, (int,np.integer)) for v in (self.horizon_steps,self.n_express,self.n_premium,self.max_eta_feature)):
            raise ValueError('Horizon, fleet counts and ETA buckets must be integers')
        if self.horizon_steps < 1 or not np.isfinite(self.step_minutes) or self.step_minutes <= 0:
            raise ValueError('Positive time horizon and time step are required')
        if self.n_express < 0 or self.n_premium < 0 or self.n_express + self.n_premium < 1:
            raise ValueError('At least one vehicle is required')
        if self.max_eta_feature < 1:
            raise ValueError('max_eta_feature must be positive')
        if self.batch_timing not in ('opening', 'closing'):
            raise ValueError('batch_timing must be opening or closing')


@dataclass
class Car:
    service_class: int
    region: int
    status: int = CarStatus.IDLE
    destination: int = -1
    eta: int = 0


class DualServiceEnv:
    def __init__(
        self,
        config: EnvConfig,
        arrival_rates: np.ndarray,
        trip_minutes: np.ndarray,
        relocation_minutes: np.ndarray,
        fares: np.ndarray,
        relocation_costs: np.ndarray,
        initial_regions: Optional[np.ndarray] = None,
        request_batches: Optional[np.ndarray] = None,
        initialization_mode: str = 'fixed',
        travel_time_model: str = 'deterministic',
        financial_parameters: Optional[dict] = None,
    ) -> None:
        if initialization_mode not in ('fixed', 'adp_random'):
            raise ValueError('Unknown initialization mode')
        if travel_time_model not in ('deterministic', 'adp_exponential'):
            raise ValueError('Unknown travel time model')
        required = ('express_base_fare', 'premium_base_fare', 'express_rate_per_km',
                    'premium_rate_per_km', 'average_speed', 'relocation_cost_factor')
        if travel_time_model == 'adp_exponential' and (financial_parameters is None or
                any(k not in financial_parameters or not np.isfinite(financial_parameters[k])
                    or financial_parameters[k] < 0 for k in required)):
            raise ValueError('Stochastic travel requires finite nonnegative financial parameters')
        self.initialization_mode = initialization_mode
        self.travel_time_model = travel_time_model
        self.financial_parameters = None if financial_parameters is None else dict(financial_parameters)
        self.cfg = config
        self.arrival_rates = np.asarray(arrival_rates, dtype=np.float64)
        if self.arrival_rates.ndim != 4 or self.arrival_rates.shape[-1] != 3:
            raise ValueError('arrival_rates must have shape (T, Z, Z, 3)')
        self.T, self.Z, z2, _ = self.arrival_rates.shape
        if self.T != config.horizon_steps or self.Z != z2:
            raise ValueError('Rates must match horizon_steps and square region dimensions')
        if np.any(self.arrival_rates < 0) or not np.isfinite(self.arrival_rates).all():
            raise ValueError('Rates must be finite and nonnegative')
        if self.Z < 1:
            raise ValueError('At least one region is required')
        self.trip_minutes = np.asarray(trip_minutes, dtype=np.float64)
        self.relocation_minutes = np.asarray(relocation_minutes, dtype=np.float64)
        self.fares = np.asarray(fares, dtype=np.float64)
        self.relocation_costs = np.asarray(relocation_costs, dtype=np.float64)
        for name, arr in [('trip_minutes', self.trip_minutes), ('relocation_minutes', self.relocation_minutes)]:
            if arr.shape != (self.Z, self.Z) or np.any(arr < 0) or not np.isfinite(arr).all():
                raise ValueError(f'{name} must have shape (Z,Z), finite nonnegative')
        for name, arr in [('fares', self.fares), ('relocation_costs', self.relocation_costs)]:
            if arr.shape != (2, self.Z, self.Z) or not np.isfinite(arr).all():
                raise ValueError(f'{name} must have shape (2,Z,Z), finite')
        if np.any(self.relocation_costs < 0):
            raise ValueError('relocation_costs should be nonnegative cost magnitudes')
        self.trip_steps = np.maximum(1, np.ceil(self.trip_minutes / config.step_minutes).astype(int))
        self.relocation_steps = np.maximum(1, np.ceil(self.relocation_minutes / config.step_minutes).astype(int))
        self.N = config.n_express + config.n_premium
        self.action_dim = 4 * self.Z + 1  # 3Z Serve + Z Relocate + 1 Pass
        self.pass_action = 4 * self.Z
        if initial_regions is None:
            self.initial_regions = np.array(
                [i % self.Z for i in range(config.n_express)] +
                [i % self.Z for i in range(config.n_premium)], dtype=np.int64
            )
        else:
            raw_regions = np.asarray(initial_regions)
            if not np.isfinite(raw_regions).all() or np.any(raw_regions != np.floor(raw_regions)):
                raise ValueError('initial_regions must contain finite integers')
            self.initial_regions = np.asarray(initial_regions, dtype=np.int64)
            if self.initial_regions.shape != (self.N,) or np.any(self.initial_regions < 0) or np.any(self.initial_regions >= self.Z):
                raise ValueError('initial_regions must be a length-N array of valid region IDs')
        self.request_batches = None
        if request_batches is not None:
            batches = np.asarray(request_batches)
            if (batches.shape != self.arrival_rates.shape or not np.isfinite(batches).all()
                    or np.any(batches < 0) or np.any(batches != np.floor(batches))):
                raise ValueError('request_batches must be nonnegative integer counts of shape (T,Z,Z,3)')
            self.request_batches = batches.astype(np.int64, copy=True)
        self.cars: list[Car] = []
        self.rng = np.random.default_rng(0)
        self.done = True
        self.obs_dim: Optional[int] = None

    def decode(self, action: int) -> tuple[str, int, int]:
        if not 0 <= action < self.action_dim:
            raise ValueError('Invalid action index')
        if action < 3 * self.Z:
            return ('serve', action // self.Z, action % self.Z)
        if action < 4 * self.Z:
            return ('relocate', -1, action - 3 * self.Z)
        return ('pass', -1, -1)

    def serve_action(self, order_type: int, destination: int) -> int:
        return order_type * self.Z + destination

    def relocate_action(self, destination: int) -> int:
        return 3 * self.Z + destination

    def research_settings(self):
        # Empty legacy contract preserves pre-existing checkpoint/scenario hashes.
        result = {}
        if self.initialization_mode != 'fixed' or self.travel_time_model != 'deterministic':
            result = dict(initialization_mode=self.initialization_mode, travel_time_model=self.travel_time_model,
                          financial_parameters=self.financial_parameters, sampling_contract='event_keyed_v1')
        if getattr(self, 'evaluation_override_sha256', None):
            result['evaluation_override_sha256'] = self.evaluation_override_sha256
        return result

    def sample_initial_regions(self, seed):
        if self.initialization_mode == 'adp_random':
            return draw_adp_initial_regions(self.cfg.n_express, self.cfg.n_premium, self.Z, seed)
        return self.initial_regions.copy()

    def reset(self, seed: int = 0, initial_regions=None) -> np.ndarray:
        self.rng = np.random.default_rng(seed)
        self.episode_seed = int(seed)
        regions = self.sample_initial_regions(seed) if initial_regions is None else np.asarray(initial_regions)
        if (regions.shape != (self.N,) or not np.isfinite(regions).all() or
                np.any(regions != np.floor(regions)) or np.any(regions < 0) or np.any(regions >= self.Z)):
            raise ValueError('Invalid episode initial regions')
        self.episode_initial_regions = regions.astype(np.int64, copy=True)
        self._relocation_counts = np.zeros(self.N, dtype=np.int64)
        self.cars = (
            [Car(ServiceClass.EXPRESS, int(region)) for region in regions[:self.cfg.n_express]] +
            [Car(ServiceClass.PREMIUM, int(region)) for region in regions[self.cfg.n_express:]]
        )
        self.t = 0
        self.atomic_count = 0
        self.profit = 0.0
        self.n_served = np.zeros((2, 3), dtype=np.int64)
        self.n_rejected = 0
        self.n_arrived = 0
        self.done = False
        self._begin_timestamp()
        obs = self.observe()
        self.obs_dim = len(obs)
        return obs

    def _begin_timestamp(self) -> None:
        self.requests = (self.rng.poisson(self.arrival_rates[self.t]).astype(np.int64)
                         if self.request_batches is None else self.request_batches[self.t].copy())
        self.n_arrived += int(self.requests.sum())
        self._batch_arrivals = self.requests.copy()
        self._car_order = (self.rng.permutation(self.N) if self.cfg.random_order else np.arange(self.N))
        self._cursor = 0
        if getattr(self,'_cache_enabled',False): self._refresh_observation_cache()

    def enable_observation_cache(self):
        """Opt-in exact fleet histograms; simulator mutations must go through step/reset."""
        self._cache_enabled = True
        if not self.done: self._refresh_observation_cache()

    def _cache_index(self,car):
        zone=car.region if car.status==CarStatus.IDLE else car.destination
        return (int(car.service_class),int(car.status),zone,min(car.eta,self.cfg.max_eta_feature))

    def _refresh_observation_cache(self):
        self._cached_hist=np.zeros((2,3,self.Z,self.cfg.max_eta_feature+1),dtype=np.float32)
        self._cached_remaining=np.zeros((2,self.Z),dtype=np.float32)
        for car in self.cars: self._cached_hist[self._cache_index(car)]+=1
        for cid in self._car_order[self._cursor:]:
            car=self.cars[int(cid)]
            if car.status==CarStatus.IDLE: self._cached_remaining[int(car.service_class),car.region]+=1
        # Normalized features change at only a few indices per atomic action.
        # Keep raw integer-valued float32 counts too: recompute changed entries
        # by division rather than accumulating rounded increments of 1/N.
        h=self._cached_hist.size;r=self.requests.size;m=self._cached_remaining.size
        self._cached_observation=np.zeros(h+r+m+self.Z+8,dtype=np.float32)
        self._cached_obs_hist=self._cached_observation[:h].reshape(self._cached_hist.shape)
        self._cached_obs_requests=self._cached_observation[h:h+r].reshape(self.requests.shape)
        self._cached_obs_remaining=self._cached_observation[h+r:h+r+m].reshape(self._cached_remaining.shape)
        self._cached_obs_hist[:]=self._cached_hist/self.N
        self._cached_obs_requests[:]=self.requests.astype(np.float32)/self.N
        self._cached_obs_remaining[:]=self._cached_remaining/self.N
        self._cached_car_offset=h+r+m

    def current_car(self) -> Car:
        if self.done:
            raise RuntimeError('The episode has finished')
        return self.cars[int(self._car_order[self._cursor])]

    def action_mask(self) -> np.ndarray:
        car = self.current_car()
        mask = np.zeros(self.action_dim, dtype=np.bool_)
        mask[self.pass_action] = True
        if car.status != CarStatus.IDLE:
            return mask
        origin = car.region
        for q in range(3):
            if q == 0 and car.service_class != ServiceClass.EXPRESS:
                continue
            if q == 1 and car.service_class != ServiceClass.PREMIUM:
                continue
            mask[q * self.Z:(q + 1) * self.Z] = self.requests[origin, :, q] > 0
        for dest in range(self.Z):
            if dest != origin:
                mask[self.relocate_action(dest)] = True
        return mask

    def observe(self) -> np.ndarray:
        if self.done:
            raise RuntimeError('The episode has finished')
        if getattr(self,'_cache_enabled',False):
            obs=self._cached_observation.copy()
            car=self.current_car();offset=self._cached_car_offset
            obs[offset+(car.region if car.status==CarStatus.IDLE else car.destination)]=1.
            obs[offset+self.Z+int(car.service_class)]=1.
            obs[offset+self.Z+2+int(car.status)]=1.
            obs[-3:]=((self.t+(self.cfg.batch_timing=='closing'))/self.T,
                      self._cursor/self.N,min(car.eta,1000)/1000.)
            return obs
        # Uncached reference path, also supports direct simulator-state inspection.
        hist = np.zeros((2, 3, self.Z, self.cfg.max_eta_feature + 1), dtype=np.float32)
        for car in self.cars:
            zone = car.region if car.status == CarStatus.IDLE else car.destination
            bucket = min(car.eta, self.cfg.max_eta_feature)
            hist[int(car.service_class), int(car.status), zone, bucket] += 1.0
        remaining = np.zeros((2, self.Z), dtype=np.float32)
        for car_id in self._car_order[self._cursor:]:
            car = self.cars[int(car_id)]
            if car.status == CarStatus.IDLE:
                remaining[int(car.service_class), car.region] += 1.0
        car = self.current_car()
        region = np.zeros(self.Z, dtype=np.float32)
        region[car.region if car.status == CarStatus.IDLE else car.destination] = 1.0
        car_type = np.eye(2, dtype=np.float32)[int(car.service_class)]
        car_status = np.eye(3, dtype=np.float32)[int(car.status)]
        meta = np.asarray([
            (self.t + (self.cfg.batch_timing == 'closing')) / self.T,
            self._cursor / self.N,
            min(car.eta, 1000) / 1000.0,
        ], dtype=np.float32)
        return np.concatenate([
            (hist / self.N).ravel(),
            (self.requests.astype(np.float32) / max(1., self.N)).ravel(),
            (remaining / self.N).ravel(),
            region, car_type, car_status, meta,
        ]).astype(np.float32)

    def _advance_physical_time(self) -> None:
        self.n_rejected += int(self.requests.sum())
        self.requests.fill(0)
        if self.cfg.batch_timing == 'closing' and self.t == self.T - 1:
            # Final batch is processed AT the horizon: do not simulate beyond it.
            self.t += 1
            self.done = True
            return
        for car in self.cars:
            if car.status != CarStatus.IDLE:
                car.eta -= 1
                if car.eta == 0:
                    car.region = car.destination
                    car.destination = -1
                    car.status = CarStatus.IDLE
        self.t += 1
        if self.t >= self.T:
            self.done = True
        else:
            self._begin_timestamp()

    def _sample_task_minutes(self, mean, key):
        # Draw after action selection. Keying avoids policy-dependent RNG consumption.
        u = np.random.default_rng(np.random.SeedSequence([self.episode_seed, *key])).random()
        return max(-math.log1p(-u) * (mean if mean > 0 else .1), .01)

    def step(self, action: int) -> tuple[Optional[np.ndarray], float, bool, dict]:
        if self.done:
            raise RuntimeError('The episode has finished')
        if not isinstance(action, (int,np.integer)):
            raise ValueError('Action index must be an integer')
        action = int(action)
        mask = self.action_mask()
        if not 0 <= action < self.action_dim or not mask[action]:
            raise ValueError(f'Infeasible atomic action {action}, mask={mask}')
        car = self.current_car()
        origin = car.region
        kind, order_type, dest = self.decode(action)
        task_minutes = task_steps = None
        stochastic = self.travel_time_model == 'adp_exponential'
        if kind in ('serve', 'relocate'):
            cid = int(self._car_order[self._cursor])
            if kind == 'serve':
                ordinal = int(self._batch_arrivals[origin,dest,order_type] - self.requests[origin,dest,order_type])
                key = [0,self.t,origin,dest,order_type,ordinal]
                mean = float(self.trip_minutes[origin,dest])
            else:
                key = [1,cid,int(self._relocation_counts[cid]),origin,dest]
                mean = float(self.relocation_minutes[origin,dest])
                self._relocation_counts[cid] += 1
            task_minutes = self._sample_task_minutes(mean,key) if stochastic else mean
            task_steps = max(1, math.ceil(task_minutes / self.cfg.step_minutes))
            if stochastic:
                p = self.financial_parameters
                prefix = 'express' if car.service_class == ServiceClass.EXPRESS else 'premium'
                actual_fare = p[prefix+'_base_fare'] + p[prefix+'_rate_per_km']*p['average_speed']*task_minutes/60.
        cached=getattr(self,'_cache_enabled',False)
        if cached:
            old_index=self._cache_index(car)
            if car.status==CarStatus.IDLE:
                index=(int(car.service_class),origin)
                self._cached_remaining[index]-=1
                self._cached_obs_remaining[index]=self._cached_remaining[index]/self.N
        reward = 0.0
        if kind == 'serve':
            assert car.status == CarStatus.IDLE and self.requests[origin, dest, order_type] > 0
            self.requests[origin, dest, order_type] -= 1
            if cached:
                self._cached_obs_requests[origin,dest,order_type]=np.float32(self.requests[origin,dest,order_type])/self.N
            self.n_served[int(car.service_class), order_type] += 1
            reward = actual_fare if stochastic else float(self.fares[int(car.service_class), origin, dest])
            car.status = CarStatus.OCCUPIED
            car.destination = dest
            car.eta = task_steps
        elif kind == 'relocate':
            assert car.status == CarStatus.IDLE and dest != origin
            reward = -actual_fare*self.financial_parameters['relocation_cost_factor'] if stochastic else -float(self.relocation_costs[int(car.service_class), origin, dest])
            car.status = CarStatus.RELOCATING
            car.destination = dest
            car.eta = task_steps
        if cached and kind!='pass':
            new_index=self._cache_index(car)
            self._cached_hist[old_index]-=1
            self._cached_hist[new_index]+=1
            self._cached_obs_hist[old_index]=self._cached_hist[old_index]/self.N
            self._cached_obs_hist[new_index]=self._cached_hist[new_index]/self.N
        self.profit += reward
        self.atomic_count += 1
        self._cursor += 1
        if self._cursor == self.N:
            self._advance_physical_time()
        obs = None if self.done else self.observe()
        info = {
            'profit': float(self.profit), 'served': int(self.n_served.sum()),
            'rejected': int(self.n_rejected), 'arrived': int(self.n_arrived),
            'atomic_count': self.atomic_count,
            'task_minutes': task_minutes, 'task_steps': task_steps,
        }
        return obs, reward, self.done, info

    def snapshot(self) -> dict:
        """Detailed simulator state for conservation and correctness checks."""
        return {
            't': self.t, 'cars': [(int(c.service_class), int(c.status), c.region, c.destination, c.eta) for c in self.cars],
            'profit': float(self.profit), 'served_by_class': self.n_served.copy(),
            'rejected': self.n_rejected, 'arrived': self.n_arrived,
            'orders': np.zeros_like(self.arrival_rates[0]) if self.done else self.requests.copy(),
        }
