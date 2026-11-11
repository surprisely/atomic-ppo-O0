"""Frozen-policy, full-horizon sampling benchmark including CPU/GPU transfers."""
import argparse
import copy
import gc
import hashlib
import json
import platform
import time
from dataclasses import asdict,replace
from pathlib import Path

import numpy as np
import torch

from .batched_sampling import BatchedEpisodeSampler, infer_actions
from .data import load_environment,sha256
from .ppo import AverageRewardPPO,PPOConfig
from .sampling import EpisodeSampler
from .stage4 import memory_usage


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def signature(trajectories):
    digest=hashlib.sha256()
    logp_digest=hashlib.sha256()
    for tr in trajectories:
        for key in ('states','actions','rewards','masks','dones'):
            digest.update(tr[key].tobytes())
        logp_digest.update(tr['logps'].tobytes())
    return dict(trajectory_sha256=digest.hexdigest(),old_logps_sha256=logp_digest.hexdigest(),
        mean_profit=float(np.mean([t['total_profit'] for t in trajectories])),
        atomic_samples=sum(len(t['actions']) for t in trajectories),
        actor_samples=sum(int((t['masks'].sum(-1)>1).sum()) for t in trajectories))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True)
    p.add_argument('--period',choices=['morning','evening'],default='evening')
    p.add_argument('--checkpoint',help='Locally trusted training checkpoint for a representative frozen policy')
    p.add_argument('--episodes',type=int,default=30)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=30)
    p.add_argument('--cases',nargs='+',choices=['parallel_cpu','uniform_cpu','batched_cpu','batched_cuda'],
                   default=['parallel_cpu','uniform_cpu','batched_cpu','batched_cuda'])
    p.add_argument('--update',action='store_true',help='Also time one unchanged CPU PPO update per sample batch')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if min(a.episodes,a.repeats,a.batch_size)<1:
        p.error('Counts must be positive')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    env=load_environment(Path(a.data)/f'{a.period}.npz',a.period)
    env.enable_observation_cache()
    original=(AverageRewardPPO.from_checkpoint(copy.deepcopy(env),a.checkpoint,device='cpu') if a.checkpoint else
              AverageRewardPPO(copy.deepcopy(env),PPOConfig(hidden_dim=64,critic_lr=.001,minibatch_size=512),seed=7))
    rows=[]
    report=dict(arguments=vars(a),environment=asdict(env.cfg),ppo=asdict(original.cfg),
        checkpoint_sha256=sha256(a.checkpoint) if a.checkpoint else None,
        data_sha256=sha256(Path(a.data)/f'{a.period}.npz'),
        versions=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        source_sha256={name:sha256(Path(__file__).with_name(name)) for name in
                       ('ppo.py','sampling.py','batched_sampling.py','environment.py','benchmark_batched.py')},
        warmup='20 Actor batches; CUDA synchronized before/after timed regions; model copy and transfers included',
        note='CPU environments remain sequential per episode. Parallel modes use two single-thread CPU workers. '
             'parallel_cpu uses legacy episode Torch draws; other modes share episode_uniform_v1. '
             'No validation/test input used. Sampling contexts persist across repeats; first repeat includes spawn. '
             'Parent-only cumulative high-water memory; concurrent user training may affect timings.',results=rows)
    def save():
        (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    save()
    for case in a.cases:
        if case=='batched_cuda' and not torch.cuda.is_available():
            rows.append(dict(case=case,skipped='CUDA unavailable'));save();continue
        batched=case.startswith('batched_')
        cfg=replace(original.cfg,device='cpu',rollout_backend='batched' if batched else 'episode',
                    rollout_workers=1 if batched else 2,rollout_seed_mode='episode' if case=='parallel_cpu' else 'uniform',
                    rollout_device='cuda' if case=='batched_cuda' else 'cpu',rollout_batch_size=a.batch_size)
        sampler=(BatchedEpisodeSampler(env,cfg) if batched else EpisodeSampler(env,cfg,workers=2))
        # Warm up the selected inference device without consuming learner RNG.
        actor=copy.deepcopy(original.actor).to(cfg.rollout_device).eval()
        obs=env.reset(7);states=np.tile(obs,(a.batch_size,1))
        masks=np.tile(env.action_mask(),(a.batch_size,1));uniforms=np.full(a.batch_size,.5,dtype=np.float32)
        for _ in range(20):
            infer_actions(actor,states,masks,uniforms,cfg.rollout_device)
        synchronize();del actor
        if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
        with sampler:
            for repeat in range(a.repeats):
                synchronize();started=time.perf_counter()
                trajectories=sampler.collect(original.actor,list(range(42,42+a.episodes)))
                synchronize();sample_seconds=time.perf_counter()-started
                row=dict(case=case,repeat=repeat,sampling_seconds=sample_seconds,
                         **signature(trajectories),**getattr(sampler,'last_stats',{}),**memory_usage())
                if a.update:
                    # Reset weights, optimizer and minibatch RNG for every case/repeat.
                    learner=copy.deepcopy(original)
                    torch.manual_seed(20261107)
                    started=time.perf_counter();metrics=learner.update(trajectories)
                    synchronize();row['update_seconds']=time.perf_counter()-started
                    row['sampling_and_update_seconds']=sample_seconds+row['update_seconds']
                    row['update_metrics']=metrics
                    del learner
                if torch.cuda.is_available():row['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
                rows.append(row);save()
                print(json.dumps({k:v for k,v in row.items() if k not in ('update_metrics',)}),flush=True)
                del trajectories;gc.collect()
        del sampler
    report['status']='complete';save()
    print(json.dumps(dict(output=str(out),status='complete')),flush=True)


if __name__=='__main__':main()
