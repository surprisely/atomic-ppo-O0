"""Chronological, multi-seed scale-training experiment with frozen checkpoint selection."""
from __future__ import annotations
import argparse
import copy
import json
import platform
import time
from dataclasses import replace,asdict
from pathlib import Path
import numpy as np
import torch
from .data import load_environment,sha256
from .ppo import AverageRewardPPO,PPOConfig
from .evaluation import make_scenario,save_scenario,evaluate_scenario,write_report
from .diagnostics import critic_error,probe_rollout


def select_iteration(history):
    if not history or any(r['split']!='validation' or not np.isfinite(r['validation_profit_mean']) for r in history):
        raise ValueError('Selection requires finite validation rows exclusively')
    return int(max(history,key=lambda r:(r['validation_profit_mean'],-r['iteration']))['iteration'])


def memory_usage():
    """OS high-water working set on Windows, RSS high-water on POSIX."""
    try:
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in
                ('peak_working_set','working_set','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile')]
        c=Counters();c.cb=ctypes.sizeof(c)
        current=ctypes.windll.kernel32.GetCurrentProcess;current.restype=ctypes.c_void_p
        get=ctypes.windll.psapi.GetProcessMemoryInfo
        get.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),wintypes.DWORD]
        if not get(current(),ctypes.byref(c),c.cb): raise OSError('GetProcessMemoryInfo failed')
        return dict(process_peak_working_set_bytes=int(c.peak_working_set),memory_source='Windows GetProcessMemoryInfo')
    except (AttributeError,ImportError,OSError):
        try:
            import resource
            return dict(process_peak_working_set_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),memory_source='POSIX ru_maxrss KiB')
        except ImportError:
            return dict(process_peak_working_set_bytes=None,memory_source='unavailable')


def _write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def _append(path,value):
    with Path(path).open('a',encoding='utf-8') as f: f.write(json.dumps(value,allow_nan=False)+'\n')


def verified_artifact(data,path):
    root=Path(data).resolve();path=Path(path).resolve()
    key=path.relative_to(root).as_posix()
    manifest=json.loads((root/'manifest.json').read_text())
    recorded={k.replace('\\','/'):v for k,v in manifest.get('outputs_sha256',{}).items()}
    if recorded.get(key)!=sha256(path): raise ValueError(f'Dataset artifact fingerprint mismatch: {key}')
    return path


def verified_replay(data,period,step_minutes,split,date):
    data=Path(data)
    verified_artifact(data,data/f'{period}.npz')
    replay=verified_artifact(data,data/f'{period}_replays/{split}_{date}.npz')
    with np.load(replay,allow_pickle=False) as f: meta=json.loads(str(f['metadata_json']))
    if meta.get('split')!=split or meta.get('date')!=date:
        raise ValueError('Replay split/date mismatch')
    return load_environment(data/f'{period}.npz',period,step_minutes,replay).request_batches


def _scenarios(data,period,env,split,dates,output):
    from .data import configure_evaluation_environment
    configure_evaluation_environment(env,data/f'{period}.npz',split)
    folder=output/f'{split}_scenarios';folder.mkdir()
    scenarios=[]
    for date in dates:
        requests=verified_replay(data,period,env.cfg.step_minutes,split,date)
        seed = 20261106 if env.initialization_mode=='fixed' else int(date.replace('-',''))
        s=make_scenario(env,seed,period,sha256(data/f'{period}.npz'),requests=requests,data_split=split,date=date)
        save_scenario(s,folder/f'{date}.npz');scenarios.append(s)
    return scenarios


def run(args):
    torch.set_num_threads(1)
    if args.iterations<1 or args.episodes<1 or args.eval_every<1: raise ValueError('Positive iterations/episodes/eval interval required')
    if len(set(args.seeds))!=len(args.seeds) or len(set(args.objectives))!=len(args.objectives): raise ValueError('Duplicate seeds/objectives')
    data,out=Path(args.data),Path(args.output)
    dataset=json.loads((data/'manifest.json').read_text())
    if dataset.get('status')!='complete' or dataset.get('calibration_cutoff_exclusive')!='2025-10-25':
        raise ValueError('Stage4 requires completed training-only calibration')
    validation_dates=args.validation_dates or dataset['validation_dates']
    test_dates=args.test_dates or dataset['test_dates']
    if not validation_dates or not set(validation_dates)<=set(dataset['validation_dates']): raise ValueError('Unknown validation dates')
    if not set(test_dates)<=set(dataset['test_dates']): raise ValueError('Unknown test dates')
    verified_artifact(data,data/f'{args.period}.npz')
    with np.load(data/f'{args.period}.npz',allow_pickle=False) as f: meta=json.loads(str(f['metadata_json']))
    env=load_environment(data/f'{args.period}.npz',args.period,meta['step_minutes'])
    env.cfg=replace(env.cfg,max_eta_feature=args.eta_buckets,random_order=not args.fixed_order)
    env.enable_observation_cache()
    evaluator=copy.deepcopy(env)
    out.mkdir(parents=True,exist_ok=False)
    val=_scenarios(data,args.period,evaluator,'validation',validation_dates,out)
    cfg=PPOConfig(hidden_dim=args.hidden_dim,actor_lr=args.actor_lr,critic_lr=args.critic_lr,
        minibatch_size=args.minibatch_size,actor_epochs=args.actor_epochs,critic_epochs=args.critic_epochs,
        reward_scale=args.reward_scale,device=args.device,
        rollout_workers=getattr(args,'rollout_workers',1),
        rollout_seed_mode=getattr(args,'rollout_seed_mode','auto'),
        rollout_backend=getattr(args,'rollout_backend','episode'),
        rollout_device=getattr(args,'rollout_device','cpu'),
        rollout_batch_size=getattr(args,'rollout_batch_size',30),
        rollout_tensor_graph=getattr(args,'rollout_tensor_graph',True),
        rollout_travel_rng=getattr(args,'rollout_travel_rng','step'))
    manifest=dict(status='running',arguments=vars(args),environment=asdict(env.cfg),ppo=asdict(cfg),
        rollout_travel_noise_contract=('event_keyed_v1' if cfg.rollout_travel_rng=='event' else 'tensor_step_uniform_v1') if cfg.rollout_backend=='tensor' else
            ('event_keyed_v1' if env.travel_time_model=='adp_exponential' else 'deterministic'),
        research_settings=env.research_settings(),
        dataset_manifest_sha256=sha256(data/'manifest.json'),calibration_sha256=sha256(data/f'{args.period}.npz'),
        validation_scenario_ids=[s['scenario_id'] for s in val],
        versions=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__),
        source_sha256={f:sha256(Path(__file__).with_name(f)) for f in
            ('stage4.py','stage4_data.py','ppo.py','sampling.py','batched_sampling.py','tensor_environment.py',
             'tensor_sampling.py','event_rng.py','event_tables.py','environment.py','evaluation.py','diagnostics.py','baselines.py')})
    _write(out/'manifest.json',manifest)
    selected=[];all_history={}
    for objective in args.objectives:
        for seed in args.seeds:
            name=f'{objective}_seed{seed}'
            folder=out/name;folder.mkdir()
            trainer=AverageRewardPPO(copy.deepcopy(env),replace(cfg,objective=objective),seed=seed)
            history=[]
            train_start=time.perf_counter()
            last=train_start
            def validate():
                validation_start=time.perf_counter()
                cp=folder/f'iteration_{trainer.iteration:06d}.pt'
                trainer.save_checkpoint(cp)
                results=[]
                for scenario in val:
                    row=evaluate_scenario(evaluator,scenario,'ppo_greedy',trainer.actor,checkpoint_hash=sha256(cp))
                    row.update(training_seed=seed,objective=objective,iteration=trainer.iteration)
                    _append(folder/'validation_episodes.jsonl',row);results.append(row['profit_total'])
                entry=dict(split='validation',iteration=trainer.iteration,validation_profit_mean=float(np.mean(results)),
                    validation_profit_std=float(np.std(results,ddof=1)) if len(results)>1 else None,checkpoint=str(cp),
                    validation_seconds=time.perf_counter()-validation_start)
                history.append(entry);_append(folder/'validation.jsonl',entry)
                print(json.dumps(dict(run=name,**entry)),flush=True)
            validate()
            def record(row):
                nonlocal last
                now=time.perf_counter()
                row=dict(row,**trainer.last_timings,**trainer.last_sampling_stats,training_seed=seed,objective=objective,wallclock_iteration_seconds=now-last,**memory_usage(),
                    cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(
                        torch.device(args.device if args.device.startswith('cuda') else cfg.rollout_device)))
                        if args.device.startswith('cuda') or cfg.rollout_device.startswith('cuda') else None)
                _append(folder/'train.jsonl',row)
                if trainer.iteration%args.eval_every==0 or trainer.iteration==args.iterations: validate()
                last=time.perf_counter()
            last=time.perf_counter()
            trainer.train(args.iterations,args.episodes,seed,on_iteration=record)
            iteration=select_iteration(history)
            selection=dict(name=name,objective=objective,training_seed=seed,iteration=iteration,
                checkpoint=str(folder/f'iteration_{iteration:06d}.pt'),validation_profit_mean=next(r['validation_profit_mean'] for r in history if r['iteration']==iteration),
                wallclock_train_and_validation_seconds=time.perf_counter()-train_start)
            selected.append(selection);all_history[name]=history
            _write(folder/'selection.json',selection)
            del trainer
    # All choices are frozen and written before any test scenario is loaded.
    _write(out/'selection_frozen.json',selected)
    selection_hash=sha256(out/'selection_frozen.json')
    evaluation=val if args.validation_only else _scenarios(data,args.period,evaluator,'test',test_dates,out)
    results=[]
    for scenario in evaluation:
        for algorithm in ('greedy_fare','compatible_stay','demand_relocate'):
            row=evaluate_scenario(evaluator,scenario,algorithm)
            results.append(row);_append(out/'evaluation_episodes.jsonl',row)
    for selection in selected:
        trainer=AverageRewardPPO.from_checkpoint(copy.deepcopy(env),selection['checkpoint'],device=args.device,for_evaluation=True)
        for scenario in evaluation:
            row=evaluate_scenario(evaluator,scenario,'ppo_greedy',trainer.actor,checkpoint_hash=sha256(selection['checkpoint']))
            row.update(algorithm=selection['name'],training_seed=selection['training_seed'],objective=selection['objective'],
                selected_iteration=selection['iteration'],selection_sha256=selection_hash)
            # CSV schema is unified below; raw JSONL retains provenance fields.
            results.append(row);_append(out/'evaluation_episodes.jsonl',row)
        diagnostic_env=copy.deepcopy(env);diagnostic_env.request_batches=val[0]['requests'].copy()
        trainer.env=diagnostic_env
        torch.manual_seed(20261106)
        trajectory=trainer.rollout(20261106,greedy=True)
        _write(out/selection['name']/'diagnostics.json',dict(critic=critic_error(trainer,trajectory),
            action_ranking=probe_rollout(diagnostic_env,trainer.critic,trainer.cfg.reward_scale),
            data_split='validation',date=validation_dates[0],selection_sha256=selection_hash))
    # Consistent CSV columns even though baseline rows have no training seed.
    keys=list(dict.fromkeys(k for row in results for k in row))
    rows=[{k:r.get(k) for k in keys} for r in results]
    report=write_report(rows,out)
    seed_means=[]
    baseline=float(np.mean([r['profit_total'] for r in results if r['algorithm']=='greedy_fare']))
    for objective in args.objectives:
        values=[float(np.mean([r['profit_total'] for r in results if r['algorithm']==f'{objective}_seed{seed}'])) for seed in args.seeds]
        seed_means.append(dict(objective=objective,n_training_seeds=len(values),seed_means=values,
            mean=float(np.mean(values)),std_across_seed_means=float(np.std(values,ddof=1)) if len(values)>1 else None,
            mean_paired_difference_vs_greedy=float(np.mean(values)-baseline)))
    _write(out/'training_seed_summary.json',seed_means)
    manifest.update(status='complete',selection_sha256=selection_hash,**memory_usage(),
        evaluated_dates=validation_dates if args.validation_only else test_dates,
        test_used_for_selection=False,outputs_sha256={str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'})
    _write(out/'manifest.json',manifest)
    print(json.dumps(dict(output=str(out),training_seed_summary=seed_means)),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--period',choices=['morning','evening'],required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[7,11,19])
    p.add_argument('--objectives',nargs='+',choices=['centered','episodic'],default=['centered','episodic'])
    p.add_argument('--iterations',type=int,default=6);p.add_argument('--episodes',type=int,default=2)
    p.add_argument('--eval-every',type=int,default=3)
    p.add_argument('--validation-dates',nargs='+');p.add_argument('--test-dates',nargs='+')
    p.add_argument('--validation-only',action='store_true')
    p.add_argument('--device',default='cpu');p.add_argument('--hidden-dim',type=int,default=64)
    p.add_argument('--rollout-workers',type=int,default=1,help='Persistent CPU episode workers; >1 enables independent episode policy RNGs')
    p.add_argument('--rollout-backend',choices=['episode','batched','tensor'],default='episode')
    p.add_argument('--rollout-tensor-graph',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--rollout-travel-rng',choices=['step','event'],default='step',help='Tensor travel coupling: original NumPy event keys or legacy Tensor step tape')
    p.add_argument('--rollout-device',default='cpu',help='Sampling device: Actor for batched, environment+Actor for tensor; training uses --device')
    p.add_argument('--rollout-batch-size',type=int,default=30)
    p.add_argument('--rollout-seed-mode',choices=['auto','legacy','episode','uniform'],default='auto',
        help='auto preserves legacy serial RNG; parallel uses episode_cpu_v1; episode also allows a serial reference')
    p.add_argument('--actor-lr',type=float,default=3e-4);p.add_argument('--critic-lr',type=float,default=1e-3)
    p.add_argument('--minibatch-size',type=int,default=512);p.add_argument('--eta-buckets',type=int,default=8)
    p.add_argument('--actor-epochs',type=int,default=3);p.add_argument('--critic-epochs',type=int,default=4)
    p.add_argument('--reward-scale',type=float,default=100.)
    p.add_argument('--fixed-order',action='store_true')
    run(p.parse_args())


if __name__=='__main__': main()
