"""Controlled training throughput check; no validation/test selection or tuning."""
import argparse
import copy
import hashlib
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .data import load_environment, sha256
from .ppo import AverageRewardPPO, PPOConfig
from .stage4 import memory_usage


def weights_digest(trainer):
    digest = hashlib.sha256()
    for model in (trainer.actor, trainer.critic):
        for name, value in model.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--period', choices=['morning','evening'], default='evening')
    p.add_argument('--workers', nargs='+', type=int, default=[1,2])
    p.add_argument('--episodes', type=int, default=4)
    p.add_argument('--iterations', type=int, default=2)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if min(a.episodes, a.iterations, *a.workers) < 1:
        p.error('Counts must be positive')
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    data = Path(a.data)/f'{a.period}.npz'
    env = load_environment(data, a.period)
    env.enable_observation_cache()
    rows = []
    report = dict(arguments=vars(a), environment=asdict(env.cfg),
        data_sha256=sha256(data), versions=dict(python=platform.python_version(),
        numpy=np.__version__, torch=torch.__version__),
        source_sha256={name:sha256(Path(__file__).with_name(name)) for name in
                       ('ppo.py','sampling.py','environment.py','benchmark_sampling.py')},
        note='CPU, one Torch thread per process; episode_cpu_v1 for every worker count. '
             'Includes pool startup/shutdown. OS high-water memory is parent-only and cumulative; '
             'concurrent jobs and short batches affect timings.', results=rows)
    for workers in a.workers:
        config = PPOConfig(hidden_dim=64, critic_lr=.001, minibatch_size=512,
                           rollout_workers=workers, rollout_seed_mode='episode')
        trainer = AverageRewardPPO(copy.deepcopy(env), config, seed=7)
        timings = []
        def record(row):
            entry = dict(iteration=trainer.iteration, **trainer.last_timings)
            timings.append(entry)
            print(json.dumps(dict(workers=workers, **entry)), flush=True)
        started = time.perf_counter()
        trainer.train(a.iterations, a.episodes, 7, on_iteration=record)
        row = dict(workers=workers, actual_workers=min(workers,a.episodes),
                   wallclock_seconds=time.perf_counter()-started, iterations=timings,
                   final_weights_sha256=weights_digest(trainer), **memory_usage())
        rows.append(row)
        report['exact_final_weights_equal'] = len({r['final_weights_sha256'] for r in rows}) == 1
        (output/'report.json').write_text(json.dumps(report,indent=2), encoding='utf-8')
        del trainer
    print(json.dumps(dict(output=str(output), exact_final_weights_equal=report['exact_final_weights_equal'])),flush=True)


if __name__ == '__main__':
    main()
