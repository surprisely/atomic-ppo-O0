"""Compare resident Tensor simulation with NumPy using identical travel noise."""
import argparse
import copy
import gc
import json
import platform
import time
from dataclasses import asdict,replace
from pathlib import Path

import numpy as np
import torch

from .batched_sampling import BatchedEpisodeSampler
from .benchmark_batched import signature
from .data import load_environment,sha256
from .ppo import AverageRewardPPO,PPOConfig
from .stage4 import memory_usage
from .tensor_environment import noise_reference
from .tensor_sampling import TensorEpisodeSampler


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True)
    p.add_argument('--period',choices=['morning','evening'],default='evening')
    p.add_argument('--checkpoint')
    p.add_argument('--episodes',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=30)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--cases',nargs='+',choices=['numpy_cpu','numpy_cuda','tensor_cuda_eager','tensor_cuda_graph'],
                   default=['numpy_cpu','numpy_cuda','tensor_cuda_graph'])
    p.add_argument('--update',action='store_true')
    p.add_argument('--travel-rng',choices=['step','event'],default='step')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if min(a.episodes,a.batch_size,a.repeats)<1:p.error('Counts must be positive')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    env=load_environment(Path(a.data)/f'{a.period}.npz',a.period);env.enable_observation_cache()
    original=(AverageRewardPPO.from_checkpoint(copy.deepcopy(env),a.checkpoint,device='cpu') if a.checkpoint else
              AverageRewardPPO(copy.deepcopy(env),PPOConfig(hidden_dim=64,critic_lr=.001,minibatch_size=512),7))
    rows=[]
    report=dict(arguments=vars(a),environment=asdict(env.cfg),ppo=asdict(original.cfg),
        versions=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        checkpoint_sha256=sha256(a.checkpoint) if a.checkpoint else None,
        data_sha256=sha256(Path(a.data)/f'{a.period}.npz'),
        source_sha256={n:sha256(Path(__file__).with_name(n)) for n in
                       ('tensor_environment.py','tensor_sampling.py','event_rng.py','event_tables.py','batched_sampling.py','ppo.py','environment.py','benchmark_tensor.py')},
        policy_noise_contract='episode_uniform_v1',travel_noise_contract='event_keyed_v1' if a.travel_rng=='event' else 'tensor_step_uniform_v1',
        note=('Unmodified NumPy event-keyed travel is the reference. ' if a.travel_rng=='event' else
              'NumPy reference uses the Tensor step tape rather than original event-keyed coupling. ')+
             'Each repeat samples the same frozen policy/seeds; optional unchanged CPU PPO update is reset to the same checkpoint/RNG. '
             'Includes preparation, GPU transfers, capture/warmup on first repeat, replay and CPU trajectory export. '
             'CUDA synchronized. User training may run concurrently. Parent-only cumulative memory.',results=rows)
    def save():(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    save()
    for case in a.cases:
        cuda=case!='numpy_cpu'
        if cuda and not torch.cuda.is_available():
            rows.append(dict(case=case,skipped='CUDA unavailable'));save();continue
        tensor=case.startswith('tensor_')
        cfg=replace(original.cfg,device='cpu',rollout_backend='tensor' if tensor else 'batched',rollout_workers=1,
                    rollout_seed_mode='uniform',rollout_device='cuda' if cuda else 'cpu',
                    rollout_batch_size=a.batch_size,rollout_tensor_graph=case=='tensor_cuda_graph',
                    rollout_travel_rng=a.travel_rng if tensor else 'step')
        sampler=(TensorEpisodeSampler(env,cfg) if tensor else BatchedEpisodeSampler(env if a.travel_rng=='event' else noise_reference(env),cfg))
        if cuda:torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        with sampler:
            for repeat in range(a.repeats):
                if cuda:torch.cuda.synchronize()
                start=time.perf_counter()
                trajectories=sampler.collect(original.actor,list(range(42,42+a.episodes)))
                if cuda:torch.cuda.synchronize()
                seconds=time.perf_counter()-start
                row=dict(case=case,repeat=repeat,sampling_seconds=seconds,**signature(trajectories),
                         **sampler.last_stats,**memory_usage())
                if a.update:
                    learner=copy.deepcopy(original);torch.manual_seed(20261107)
                    start=time.perf_counter();metrics=learner.update(trajectories)
                    row.update(update_seconds=time.perf_counter()-start,update_metrics=metrics)
                    row['sampling_and_update_seconds']=seconds+row['update_seconds']
                    del learner
                if cuda:row['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
                rows.append(row);save()
                print(json.dumps({k:v for k,v in row.items() if k!='update_metrics'}),flush=True)
                del trajectories;gc.collect()
        del sampler;gc.collect()
        if cuda:torch.cuda.empty_cache()
    report['status']='complete';save()
    print(json.dumps(dict(output=str(out),status='complete')),flush=True)


if __name__=='__main__':main()
