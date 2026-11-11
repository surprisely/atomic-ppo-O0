"""Synthetic, reproducible inputs strictly for smoke tests, NOT ADP paper data."""
from __future__ import annotations
import numpy as np
from .environment import EnvConfig, DualServiceEnv


def make_synthetic_env(cfg: EnvConfig, zones: int = 3, seed: int = 0) -> DualServiceEnv:
    rng = np.random.default_rng(seed)
    Z, T = zones, cfg.horizon_steps
    minutes = np.full((Z, Z), 4.0, dtype=np.float64)
    np.fill_diagonal(minutes, 1.0)
    reloc_minutes = minutes.copy()
    demand = np.zeros((T, Z, Z, 3), dtype=np.float64)
    passenger_weights = np.array([0.5, 0.2, 0.3])
    for t in range(T):
        variation = 1.0 + 0.25 * np.sin(2.0 * np.pi * t / max(1,T))
        for i in range(Z):
            for j in range(Z):
                base = (0.6 if i != j else 0.2) * variation
                demand[t, i, j] = base * passenger_weights
    # A tiny amount of seeded heterogeneity makes the example nondegenerate.
    demand *= rng.uniform(0.85, 1.15, size=(1, Z, Z, 1))
    fare = np.broadcast_to(np.array([10., 16.])[:, None, None], (2, Z, Z)).copy()
    relocation_cost = np.broadcast_to(np.array([1.5, 2.0])[:, None, None], (2, Z, Z)).copy()
    return DualServiceEnv(cfg, demand, minutes, reloc_minutes, fare, relocation_cost)
