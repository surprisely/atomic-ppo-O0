"""Isolated real-workload GPU update/memory check; production update is unchanged."""
import argparse
import gc
import json
import platform
import subprocess
import threading
import time
from dataclasses import asdict,replace
from pathlib import Path

import numpy as np
import torch

from .data import load_environment,sha256
from .ppo import AverageRewardPPO
from .tensor_sampling import TensorEpisodeSampler


def gpu_memory():
    torch.cuda.synchronize()
    free,total=torch.cuda.mem_get_info()
    stats=torch.cuda.memory_stats()
    return dict(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        cuda_free_bytes=free,cuda_total_bytes=total,
        allocation_retries=stats.get('num_alloc_retries',0),oom_count=stats.get('num_ooms',0))


class Monitor:
    """Management-only sampling; no Torch/CUDA work on the background thread."""
    def __init__(self):
        self.phase='setup';self.rows=[];self.stop=threading.Event()
    def run(self):
        while not self.stop.is_set():
            try:
                r=subprocess.run(['nvidia-smi','--id=0','--query-gpu=memory.total,memory.used,memory.free,utilization.gpu',
                                  '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
                if r.returncode==0:
                    total,used,free,util=[int(v.strip()) for v in r.stdout.strip().split(',')]
                    self.rows.append(dict(monotonic_seconds=time.perf_counter(),phase=self.phase,
                        total_mib=total,used_mib=used,free_mib=free,utilization_percent=util))
            except (OSError,ValueError,subprocess.TimeoutExpired):pass
            self.stop.wait(.5)
    def __enter__(self):
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.stop.set();self.thread.join(timeout=4)


def run_case(env,checkpoint,batch,episodes,iterations,save,rows):
    record=dict(rollout_batch_size=batch,episodes=episodes,iterations=[],status='running')
    rows.append(record);save()
    with Monitor() as monitor:
        try:
            learner=AverageRewardPPO.from_checkpoint(env,checkpoint,device='cuda')
            learner.cfg=replace(learner.cfg,device='cuda',rollout_backend='tensor',rollout_device='cuda',
                rollout_workers=1,rollout_seed_mode='uniform',rollout_travel_rng='event',rollout_batch_size=batch)
            record['config']=asdict(learner.cfg)
            before=gpu_memory();record['before']=before
            with TensorEpisodeSampler(learner.env,learner.cfg) as sampler:
                for repeat in range(iterations):
                    seeds=list(range(42+(learner.iteration)*episodes,42+(learner.iteration+1)*episodes))
                    monitor.phase=f'iteration_{repeat+1}_sampling'
                    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
                    trajectories=sampler.collect(learner.actor,seeds)
                    torch.cuda.synchronize();sample_seconds=time.perf_counter()-started
                    entry=dict(iteration=repeat+1,episode_seed_first=seeds[0],episode_seed_last=seeds[-1],
                        sampling_seconds=sample_seconds,sampling_memory=gpu_memory(),sampling_stats=sampler.last_stats)
                    record['iterations'].append(entry);save()
                    monitor.phase=f'iteration_{repeat+1}_update'
                    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
                    metrics=learner.update(trajectories)
                    torch.cuda.synchronize()
                    entry.update(update_seconds=time.perf_counter()-started,update_memory=gpu_memory(),metrics=metrics)
                    entry['sampling_and_update_seconds']=sample_seconds+entry['update_seconds']
                    print(json.dumps(dict(batch=batch,iteration=repeat+1,sampling_seconds=sample_seconds,
                        update_seconds=entry['update_seconds'],update_memory=entry['update_memory'])),flush=True)
                    learner.iteration+=1
                    del trajectories;gc.collect();save()
            record['after']=gpu_memory()
            record['allocation_retries_added']=record['after']['allocation_retries']-before['allocation_retries']
            record['ooms_added']=record['after']['oom_count']-before['oom_count']
            record['status']='complete'
        except Exception as error:
            record.update(status='failed',error_type=type(error).__name__,error=str(error)[:2500],phase=monitor.phase)
            try:record['failure_memory']=gpu_memory()
            except Exception:pass
            print(json.dumps(dict(batch=batch,status='failed',phase=monitor.phase,error_type=type(error).__name__,error=str(error)[:700])),flush=True)
        finally:
            record['nvml_samples']=monitor.rows
            if monitor.rows:
                record['nvml_min_free_mib']=min(v['free_mib'] for v in monitor.rows)
                record['nvml_max_used_mib']=max(v['used_mib'] for v in monitor.rows)
            save()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--period',default='evening',choices=['morning','evening'])
    p.add_argument('--episodes',type=int,default=30);p.add_argument('--iterations',type=int,default=2)
    p.add_argument('--rollout-batches',nargs='+',type=int,default=[30,15])
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if min(a.episodes,a.iterations,*a.rollout_batches)<1:p.error('Counts must be positive')
    if not torch.cuda.is_available():p.error('CUDA unavailable')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    env=load_environment(Path(a.data)/f'{a.period}.npz',a.period);env.enable_observation_cache()
    obs=env.reset(42);h=env.T*env.N;obs_bytes=a.episodes*h*len(obs)*4
    rows=[]
    report=dict(arguments=vars(a),environment=asdict(env.cfg),observation_dim=len(obs),
        atomic_samples=a.episodes*h,observation_matrix_bytes=obs_bytes,
        versions=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__),
        gpu=torch.cuda.get_device_name(0),gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        checkpoint_sha256=sha256(a.checkpoint),data_sha256=sha256(Path(a.data)/f'{a.period}.npz'),
        source_sha256={n:sha256(Path(__file__).with_name(n)) for n in
            ('probe_gpu_update.py','ppo.py','tensor_environment.py','tensor_sampling.py','event_rng.py','event_tables.py')},
        note='Isolated, unsaved GPU updates from a trusted checkpoint. All K episodes retained; only simultaneous '
             'GPU rollout environment count varies. Production update and minibatch/epoch/dtype unchanged. '
             'Resident sampler Graph/buffers are held during update. CPU export and H2D copies retained. '
             'CUDA synchronized; Torch peaks and 0.5s NVML management samples recorded. '
             'One small model, fixed data/horizon and two updates do not prove arbitrary long-run capacity.',results=rows)
    def save():(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    save()
    for batch in a.rollout_batches:
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        run_case(env,a.checkpoint,batch,a.episodes,a.iterations,save,rows)
        gc.collect();torch.cuda.empty_cache()
    report['status']='complete';save()
    print(json.dumps(dict(output=str(out),cases=[dict(batch=r['rollout_batch_size'],status=r['status']) for r in rows])),flush=True)


if __name__=='__main__':main()
