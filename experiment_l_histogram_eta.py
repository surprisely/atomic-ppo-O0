"""Original vehicle histogram / MLP ETA and OD comparisons using minute GAE.

L0: 0..7,8+; L1: one bin per integer ETA. All group
checkpoints are new models, never old iteration-220 network transplants.
"""
import argparse
import copy
import json
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from atomic_dual.data import load_environment, sha256
from atomic_dual.evaluation import load_scenario, evaluate_scenario, scenario_digest
from atomic_dual.ppo import PPOConfig, masked_categorical
from experiment_c_action_forks import clone_snapshot
from experiment_d_critic_consistency import rollout_branch
from experiment_ef_value_advantages import read_complete_jsonl
from diagnose_policy_drift import write_csv
from physical_time_gae import physical_advantage
from histogram_eta import HistogramSpec, HistogramEnv, HistogramPPO, HistogramSampler, parameter_hash, bucket_projection_hash, exact_eta_bound
from datetime import datetime, timezone


def append(path,row):
    with path.open('a',encoding='utf-8') as f: f.write(json.dumps(row,allow_nan=False)+'\n')


def groups(eta_max):
    return dict(L0=HistogramSpec('bucket',8),L1=HistogramSpec('exact',eta_max))


def od_groups():
    return dict(O0=HistogramSpec('bucket',8),O1=HistogramSpec('bucket',8,'marginal','marginal'),
        O2=HistogramSpec('bucket',8,'marginal','full'))


def benchmark_update(env,cfg,args):
    spec=od_groups()[args.benchmark_update]
    seed=args.seeds[0]
    cpu=HistogramPPO(HistogramEnv(clone_snapshot(env),spec),replace(cfg,device='cpu'),seed)
    gpu=HistogramPPO(HistogramEnv(clone_snapshot(env),spec),replace(cfg,device='cuda'),seed)
    if parameter_hash(cpu.actor,cpu.critic)!=parameter_hash(gpu.actor,gpu.critic):
        raise AssertionError('Benchmark requires identical initial parameters')
    print(json.dumps(dict(stage='sampling',group=args.benchmark_update,episodes=args.episodes)),flush=True)
    with HistogramSampler(cpu.env,cpu.cfg,min(args.workers,args.episodes)) as sampler:
        started=time.perf_counter()
        trajectories=sampler.collect(cpu.actor,range(seed,seed+args.episodes))
        sampling=time.perf_counter()-started

    updates={}
    with physical_advantage('minute_gae',env.N,args.minute_lambda):
        for trainer in (cpu,gpu):
            device=trainer.device.type
            print(json.dumps(dict(stage='update',device=device,sampling_seconds=sampling)),flush=True)
            if device=='cuda':
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started=time.perf_counter()
            trainer.update(trajectories)
            updates[device]=time.perf_counter()-started

    cpu_total=sampling+updates['cpu']; cuda_total=sampling+updates['cuda']
    saved=cpu_total-cuda_total
    print(json.dumps(dict(group=args.benchmark_update,seed=seed,episodes=args.episodes,
        gpu=torch.cuda.get_device_name(),sampling_seconds=sampling,update_seconds=updates,
        iteration_seconds=dict(cpu=cpu_total,cuda=cuda_total),seconds_saved=saved,
        iteration_percent_saved=100*saved/cpu_total,update_speedup=updates['cpu']/updates['cuda'],
        peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30),indent=2),flush=True)


def benchmark_rollout(env,cfg,args):
    from atomic_dual.batched_sampling import BatchedEpisodeSampler
    from atomic_dual.benchmark_batched import signature
    from atomic_dual.tensor_sampling import TensorEpisodeSampler

    group=args.benchmark_rollout
    spec=od_groups()[group]
    seed=args.seeds[0]
    seeds=list(range(seed,seed+args.episodes))
    learner_cfg=replace(cfg,device='cuda')
    frozen=HistogramPPO(HistogramEnv(clone_snapshot(env),spec),replace(cfg,device='cpu'),seed).actor
    reference=None
    print(json.dumps(dict(group=group,seed=seed,episodes=len(seeds),gpu=torch.cuda.get_device_name(),
        policy_rng='episode_uniform_v1',travel_rng='event_keyed_v1',
        initial_actor_sha256=parameter_hash(frozen))),flush=True)
    for tensor in ((True,) if args.compare_graph_cache else (False,True)):
        sampler_cfg=replace(cfg,rollout_backend='tensor' if tensor else 'batched',rollout_workers=1,
            rollout_seed_mode='uniform',rollout_device='cuda' if tensor else 'cpu',
            rollout_batch_size=len(seeds),rollout_tensor_graph=tensor,rollout_travel_rng='event' if tensor else 'step')
        sampler=TensorEpisodeSampler(env,sampler_cfg) if tensor else BatchedEpisodeSampler(env,sampler_cfg)
        with sampler, physical_advantage('minute_gae',env.N,args.minute_lambda):
            if args.compare_graph_cache:
                def make_learner():
                    return HistogramPPO(HistogramEnv(clone_snapshot(env),spec),learner_cfg,seed)
                benchmark_graph_cache(sampler,frozen,seeds,make_learner)
                return
            for stage in (('cuda_cold','cuda_reuse') if tensor else ('cpu_batched',)):
                trainer=HistogramPPO(HistogramEnv(clone_snapshot(env),spec),learner_cfg,seed)
                print(json.dumps(dict(stage=stage,event='sampling')),flush=True)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                free_at_sampling=torch.cuda.mem_get_info()[0]
                reserved_at_sampling=torch.cuda.memory_reserved()
                started=time.perf_counter()
                trajectories=sampler.collect(frozen,seeds)
                torch.cuda.synchronize()
                sampling=time.perf_counter()-started
                sampling_peak=torch.cuda.max_memory_allocated()
                summary=signature(trajectories)
                logps=np.concatenate([tr['logps'] for tr in trajectories])
                if reference is None:
                    reference=(summary['trajectory_sha256'],logps)
                same_trajectory=summary['trajectory_sha256']==reference[0]
                logp_error=float(np.max(np.abs(logps-reference[1]))) if same_trajectory else None
                print(json.dumps(dict(stage=stage,event='update',sampling_seconds=sampling,
                    same_trajectory=same_trajectory,max_abs_logp_error=logp_error)),flush=True)
                started=time.perf_counter()
                trainer.update(trajectories)
                update=time.perf_counter()-started
                print(json.dumps(dict(stage=stage,sampling_seconds=sampling,update_seconds=update,
                    iteration_seconds=sampling+update,same_trajectory=same_trajectory,max_abs_logp_error=logp_error,
                    free_cuda_at_sampling_gib=free_at_sampling/2**30,
                    reserved_cuda_at_sampling_gib=reserved_at_sampling/2**30,
                    sampling_peak_cuda_gib=sampling_peak/2**30,
                    iteration_peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,
                    iteration_peak_cuda_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                    **summary,**sampler.last_stats)),flush=True)
                del trajectories,trainer


def benchmark_graph_cache(sampler,actor,seeds,make_learner):
    from atomic_dual.benchmark_batched import signature

    def memory_gib():
        free,_=torch.cuda.mem_get_info()
        return dict(free=free/2**30,allocated=torch.cuda.memory_allocated()/2**30,
            reserved=torch.cuda.memory_reserved()/2**30)

    reference=None
    for stage in ('cold','no_update','after_update','after_update_empty_cache'):
        update_seconds=None
        cache_release_seconds=None
        memory_before_release=None
        if stage.startswith('after_update'):
            print(json.dumps(dict(stage=stage,event='update')),flush=True)
            learner=make_learner()
            torch.cuda.synchronize()
            started=time.perf_counter()
            learner.update(trajectories)
            update_seconds=time.perf_counter()-started
            del learner
        if stage!='cold':
            del trajectories
        if stage=='after_update_empty_cache':
            memory_before_release=memory_gib()
            started=time.perf_counter()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            cache_release_seconds=time.perf_counter()-started

        print(json.dumps(dict(stage=stage,event='sampling')),flush=True)
        torch.cuda.synchronize()
        memory_before_sampling=memory_gib()
        torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        trajectories=sampler.collect(actor,seeds)
        torch.cuda.synchronize()
        sampling_seconds=time.perf_counter()-started
        stats=sampler.last_stats
        if stage!='cold' and (stats['tensor_capture_seconds'] or stats['tensor_event_capacity_recaptures']):
            raise AssertionError('Cache comparison requires reusing the original CUDA Graph')
        digest=signature(trajectories)['trajectory_sha256']
        if reference is None: reference=digest
        if digest!=reference: raise AssertionError('Frozen-policy trajectories changed between stages')
        print(json.dumps(dict(stage=stage,sampling_seconds=sampling_seconds,update_seconds=update_seconds,
            cache_release_seconds=cache_release_seconds,memory_before_release_gib=memory_before_release,
            memory_before_sampling_gib=memory_before_sampling,trajectory_sha256=digest,
            sampling_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,**stats)),flush=True)


def benchmark_cpu_batch(env,cfg,args):
    from atomic_dual.batched_sampling import BatchedEpisodeSampler
    from atomic_dual.benchmark_batched import signature

    group=args.benchmark_cpu_batch
    spec=od_groups()[group]
    seed=args.seeds[0]
    learner_cfg=replace(cfg,device='cuda')
    reference=[]
    results={}
    print(json.dumps(dict(group=group,seed=seed,iterations=args.iterations,episodes=args.episodes,
        episode_workers=min(args.workers,args.episodes),gpu=torch.cuda.get_device_name(),
        policy_rng='episode_uniform_v1',travel_rng='event_keyed_v1')),flush=True)
    for case in ('cpu_episode','cpu_batched'):
        trainer=HistogramPPO(HistogramEnv(clone_snapshot(env),spec),learner_cfg,seed)
        init_hash=parameter_hash(trainer.actor,trainer.critic)
        if case=='cpu_episode':
            reference_init=init_hash
        elif init_hash!=reference_init:
            raise AssertionError('Benchmark requires identical initial parameters')
        if case=='cpu_episode':
            sampler=HistogramSampler(trainer.env,replace(cfg,rollout_seed_mode='uniform'),min(args.workers,args.episodes))
        else:
            sampler_cfg=replace(cfg,rollout_backend='batched',rollout_workers=1,
                rollout_seed_mode='uniform',rollout_device='cpu',rollout_batch_size=args.episodes)
            sampler=BatchedEpisodeSampler(trainer.env,sampler_cfg)
        rows=[]
        with sampler, physical_advantage('minute_gae',env.N,args.minute_lambda):
            for index in range(args.iterations):
                first_seed=seed+index*args.episodes
                print(json.dumps(dict(case=case,iteration=index+1,event='sampling')),flush=True)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started=time.perf_counter()
                trajectories=sampler.collect(trainer.actor,list(range(first_seed,first_seed+args.episodes)))
                sampling=time.perf_counter()-started
                print(json.dumps(dict(case=case,iteration=index+1,event='update',sampling_seconds=sampling)),flush=True)
                started=time.perf_counter()
                metrics=trainer.update(trajectories)
                update=time.perf_counter()-started
                digest=signature(trajectories)['trajectory_sha256']
                del trajectories
                if case=='cpu_episode': reference.append(digest)
                row=dict(case=case,iteration=index+1,first_episode_seed=first_seed,
                    sampling_seconds=sampling,update_seconds=update,iteration_seconds=sampling+update,
                    train_profit=metrics['gain_estimate_per_period'],actor_samples=metrics['num_actor_samples'],
                    same_trajectory_as_cpu_episode=digest==reference[index],trajectory_sha256=digest,
                    peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                rows.append(row)
                print(json.dumps(row),flush=True)
        results[case]={
            'first_iteration_seconds':rows[0]['iteration_seconds'],
            'total_seconds':sum(row['iteration_seconds'] for row in rows),
            'warm_mean_seconds':{key:float(np.mean([row[key] for row in rows[1:]]))
                for key in ('sampling_seconds','update_seconds','iteration_seconds')}}
        del trainer
    episode=results['cpu_episode']['warm_mean_seconds']['iteration_seconds']
    batched=results['cpu_batched']['warm_mean_seconds']['iteration_seconds']
    print(json.dumps(dict(summary=results,warm_seconds_saved=episode-batched,
        warm_percent_saved=100*(episode-batched)/episode,initial_parameters_sha256=reference_init)),flush=True)


def adapt_scenario(env,scenario,policy_seed=None):
    s=dict(scenario,research_settings=env.research_settings())
    if policy_seed is not None:
        s['policy_seed']=policy_seed
    s['scenario_id']=scenario_digest(s)
    return s


@torch.inference_mode()
def probe_states(trainer,snapshots,actor):
    result=[]
    for idx,base in snapshots.items():
        env=HistogramEnv(clone_snapshot(base),trainer.env.representation)
        p=masked_categorical(actor(torch.as_tensor(env.observe())),torch.as_tensor(env.action_mask())).probs
        result.append(dict(bank_row=idx,t=env.t,remaining_minutes=(env.T-env.t-int(env.cfg.batch_timing=='closing'))*env.cfg.step_minutes,
            serve_probability=float(p[:3*env.Z].sum()),pass_probability=float(p[-1]),
            entropy=float(-(p[p>0]*p[p>0].log()).sum())))
    return result


def paired_probe(snapshot,action,trainer,seeds,critic=None):
    """Use actor that generated this critic's most recent supervised targets."""
    if critic is None: critic=trainer.critic
    branches=[rollout_branch(snapshot,a,trainer.fit_actor,critic,'sampled',seeds,
        trainer.audit_gain,trainer.cfg.reward_scale) for a in (action,snapshot.pass_action)]
    s,p=branches
    if s['exogenous_sha256']!=p['exogenous_sha256']: raise AssertionError('Unpaired future dynamics')
    # Equal complete remaining horizons: gain baseline cancels analytically.
    # Keep float32-target differences separately to expose numerical residuals.
    mc=s['q_centered_raw']-p['q_centered_raw']
    td=s['td_raw']-p['td_raw']
    numeric_mc=s['q_training_target_raw']-p['q_training_target_raw']
    return dict(delta_mc=mc,delta_td=td,error_td_minus_mc=td-mc,
        delta_mc_float32=numeric_mc,float32_residual=numeric_mc-mc,
        continuation='critic_fitting_actor_before_latest_actor_update')


def summarize_q(rows):
    states=[]
    for idx in sorted({r['bank_row'] for r in rows}):
        rs=[r for r in rows if r['bank_row']==idx]; mc=np.array([r['delta_mc'] for r in rs])
        errors=np.array([r['error_td_minus_mc'] for r in rs]); td=float(np.mean([r['delta_td'] for r in rs]))
        se=float(mc.std(ddof=1)/np.sqrt(len(mc))) if len(mc)>1 else None
        ese=float(errors.std(ddof=1)/np.sqrt(len(errors))) if len(errors)>1 else None
        low=None if se is None else float(mc.mean()-1.96*se); high=None if se is None else float(mc.mean()+1.96*se)
        states.append(dict(bank_row=idx,n_replicates=len(mc),mc_mean=float(mc.mean()),td_mean=td,
            error=float(errors.mean()),abs_relative_error=abs(float(errors.mean())),
            error_ci95_low=None if ese is None else float(errors.mean()-1.96*ese),
            error_ci95_high=None if ese is None else float(errors.mean()+1.96*ese),
            mc_ci95_low=low,mc_ci95_high=high,
            sign_wrong=bool(np.sign(mc.mean())!=np.sign(td)),resolved=bool(low is not None and (low>0 or high<0))))
    resolved=[r for r in states if r['resolved']]
    return states,dict(n_states=len(states),mean_relative_error=float(np.mean([r['error'] for r in states])),
        mean_abs_relative_error=float(np.mean([r['abs_relative_error'] for r in states])),
        sign_error_rate=float(np.mean([r['sign_wrong'] for r in states])),n_resolved=len(resolved),
        sign_error_rate_resolved=float(np.mean([r['sign_wrong'] for r in resolved])) if resolved else None,
        max_abs_float32_residual=max(abs(r['float32_residual']) for r in rows))


def run(args):
    if args.experiment=='od' and args.l0_only:
        raise ValueError('--l0-only applies to the ETA experiment')
    if args.early_stop_patience<0 or (args.early_stop_patience and not args.l0_only):
        raise ValueError('Early stopping requires --l0-only and a nonnegative patience')
    benchmark=args.benchmark_update or args.benchmark_rollout or args.benchmark_cpu_batch
    if args.od_group and (args.experiment!='od' or benchmark):
        raise ValueError('--od-group requires an OD training run')
    batched=args.rollout_backend=='batched'
    if batched and (benchmark or args.experiment!='od' or args.workers!=1):
        raise ValueError('--rollout-backend batched requires OD training and --workers 1')
    if args.compare_graph_cache and not args.benchmark_rollout:
        raise ValueError('--compare-graph-cache requires --benchmark-rollout')
    if args.benchmark_cpu_batch and args.iterations<2:
        raise ValueError('--benchmark-cpu-batch requires at least two iterations')
    if benchmark and (len(args.seeds)!=1 or args.resume_from or args.dry_run):
        raise ValueError('Benchmark requires one seed and a fresh run')
    if not benchmark and args.output is None:
        raise ValueError('--output is required for training and dry runs')
    if (args.device=='cuda' or benchmark) and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable in this Python environment')
    if args.device=='cuda' and args.experiment=='eta' and not (args.l0_only or benchmark):
        raise ValueError('CUDA updates require --experiment od or --l0-only')
    if args.continue_l0 and (not args.l0_only or not args.resume_from or args.early_stop_patience):
        raise ValueError('--continue-l0 requires --l0-only, --resume-from, and early stopping disabled')
    torch.set_num_threads(1)
    root=Path(args.run); original=json.loads((root/'manifest.json').read_text())
    env=load_environment(Path(original['arguments']['data'])/'evening.npz','evening',original['environment']['step_minutes'])
    env.cfg=replace(env.cfg,**original['environment']); env.enable_observation_cache()
    if env.cfg.step_minutes!=1 or env.cfg.max_eta_feature!=8: raise ValueError('Requires integer-minute environment and original 8+ cap')
    ppo_settings=dict(original['ppo'],hidden_dim=args.hidden_dim,device=args.device,rollout_workers=args.workers,
        rollout_seed_mode='uniform' if batched else 'episode',rollout_backend=args.rollout_backend)
    if batched: ppo_settings.update(rollout_device='cpu',rollout_batch_size=args.episodes)
    cfg=PPOConfig(**ppo_settings)
    if args.benchmark_update:
        benchmark_update(env,cfg,args)
        return
    if args.benchmark_rollout:
        benchmark_rollout(env,cfg,args)
        return
    if args.benchmark_cpu_batch:
        benchmark_cpu_batch(env,cfg,args)
        return
    out=Path(args.output)
    scenarios=[load_scenario(p) for p in sorted((root/'validation_scenarios').glob('*.npz'))]
    if [s['scenario_id'] for s in scenarios]!=original['validation_scenario_ids']: raise ValueError('Scenario mismatch')
    snaproot=Path(args.snapshots); file=snaproot/'trusted_local_snapshots.pt'
    sm=json.loads((snaproot/'manifest.json').read_text())
    if sha256(file)!=sm['outputs_sha256'][file.name]: raise ValueError('Snapshot mismatch')
    saved=torch.load(file,map_location='cpu',weights_only=False)['snapshots']
    indices=sorted(saved) if args.q_states is None else args.q_states
    snapshots={i:clone_snapshot(saved[i]) for i in indices}
    if args.experiment=='od':
        eta_max=8
        specs=od_groups()
        if args.od_group:
            specs={args.od_group:specs[args.od_group]}
    else:
        eta_max=exact_eta_bound(env,saved.values())
        specs=groups(eta_max)
        if args.l0_only: specs={'L0':specs['L0']}
    with np.load(Path(args.bank)/'action_probabilities.npz',allow_pickle=False) as f: actions=f['150'].argmax(-1)
    for i,s in snapshots.items():
        if not s.action_mask()[actions[i]] or actions[i]>=3*s.Z: raise ValueError('Probe Serve must be feasible')
    source_names=('experiment_l_histogram_eta.py','histogram_eta.py','physical_time_gae.py','advantage_ablation.py',
        'atomic_dual/ppo.py','atomic_dual/environment.py','atomic_dual/sampling.py','atomic_dual/evaluation.py','atomic_dual/data.py',
        'atomic_dual/event_rng.py','atomic_dual/event_tables.py','atomic_dual/batched_sampling.py','diagnose_policy_drift.py',
        'experiment_c_action_forks.py','experiment_d_critic_consistency.py','experiment_ef_value_advantages.py')
    arguments={k:v for k,v in vars(args).items() if k not in ('output','resume_from','dry_run','continue_l0')}
    if args.od_group is None:
        arguments.pop('od_group')
    contract=dict(arguments=arguments,
        source_sha256={n:sha256(n) for n in source_names},snapshots_sha256=sha256(file),
        bank_actions_sha256=sha256(Path(args.bank)/'action_probabilities.npz'),run_manifest_sha256=sha256(root/'manifest.json'),
        calibration_sha256=sha256(Path(original['arguments']['data'])/'evening.npz'),
        validation_scenario_ids=original['validation_scenario_ids'],ppo=asdict(cfg),representations={k:asdict(v) for k,v in specs.items()})
    tail_dim=3*env.Z**2+3*env.Z+8
    batch_gib_per_feature=(env.T*env.N+1)*4*args.episodes/2**30
    manifest=dict(status='running',contract=contract,initializations=[],
        common_architecture='Original histogram followed by Linear/Tanh/Linear/Tanh/Linear MLP. L0 uses the original cached dense observation and ordinary MLP. L1 uses sparse count storage and an algebraically equivalent sparse first Linear.',
        baseline_scope='L0 matches old histogram features including the existing current-car ETA scalar; full 243 OD. No added candidate quotes.',
        vehicle_records='Grouped counts by class/status/zone/ETA. L0:0..7,8+; L1:one bin per integer through proven implementation bound, no overflow/tail bucket.',
        eta_max=eta_max,eta_bound_rule='53-bit PCG uniform inverse CDF, max=-log(2^-53)*maximum mean, ceil to minutes +1 safety bin; include snapshots.',
        capacity='Same hidden64 MLP; L1 first Linear necessarily has more parameters. Initial >=8 coefficients copied from common L0, all other coefficients and learner RNG matched.',
        od='Both L0 and L1 retain original full Z*Z*3 OD; no OD ablation in this runner.',
        gae=dict(gamma=1,lambda_unit='physical minute',minute_lambda=args.minute_lambda),
        critic_target='Unchanged complete gain-centered MC, true terminal zero. Q references saved PRE-actor-update fitting policy.',
        probe_scope='Existing 15 Serve150-to-Pass300 full snapshots; diagnostic subset, not population weighted. Each state receives equal weight.',
        inference='Report per-state MC repeat intervals; do not count fork repetitions as independent training seeds.',
        initialization='All arms from scratch; same initialization seed, training episode seeds, budgets and validation exogenous data.',
        memory_gib_per_batch={name:((2*env.N if spec.eta=='exact' else 2*3*env.Z*9)+tail_dim)*batch_gib_per_feature
                              for name,spec in specs.items()})
    if args.experiment=='od':
        manifest.update(
            common_architecture='L0 dense histogram and Linear/Tanh/Linear/Tanh/Linear MLP in all arms.',
            baseline_scope='O0 is the original L0 with full OD.',
            vehicle_records='L0 buckets 0..7,8+ in all arms.',
            eta_bound_rule='ETA 8+ bucket',
            capacity='Common hidden width; marginal OD reduces first-layer parameters.',
            od='Full OD or origin and destination marginals by passenger class. Marginal projection occurs before the first Linear; the rollout stores the shared full observation. Feasibility masks are unchanged.',
            initialization='From scratch, same episode seeds and learner RNG. Marginal networks reuse the full network initialization for non-OD inputs and later layers, with a subset of the full OD coefficients assigned to the marginal inputs.')
    if args.resume_from:
        previous=json.loads((Path(args.resume_from)/'manifest.json').read_text())
        previous_contract=previous['contract']
        if args.continue_l0:
            if previous['status']!='complete':
                raise ValueError('Finish the source run before continuing L0')
            old_arguments=previous_contract['arguments']
            compatible_arguments=dict(device='cpu',rollout_backend='episode',benchmark_update=None,
                benchmark_rollout=None,benchmark_cpu_batch=None,compare_graph_cache=False,
                validation_algorithm='ppo_greedy')
            compatible_arguments.update(old_arguments,iterations=args.iterations,early_stop_patience=0)
            compatible_contract=dict(previous_contract,arguments=compatible_arguments,source_sha256=contract['source_sha256'])
            if compatible_contract!=contract:
                mismatches=[key for key in contract if compatible_contract.get(key)!=contract[key]]
                raise ValueError(f'Continuation requires matching training and input settings: {mismatches}')
            for seed in args.seeds:
                source=Path(args.resume_from)/f'L0_seed{seed}'
                commits=sorted(source.glob('iteration_*.commit.json'))
                if not commits or json.loads(commits[-1].read_text())['iteration']>=args.iterations:
                    raise ValueError('Target iterations must exceed the last committed L0 iteration')
            manifest['continuation']=dict(source=str(Path(args.resume_from).resolve()),
                previous_source_sha256=previous_contract['source_sha256'])
        elif previous_contract!=contract:
            raise ValueError('Resume contract differs; keep arguments and source unchanged')
    initial={}
    for seed in args.seeds:
        for name,spec in specs.items():
            tr=HistogramPPO(HistogramEnv(clone_snapshot(env),spec),cfg,seed)
            item=dict(seed=seed,group=name,packed_obs_dim=len(tr.env.observe()),dense_obs_dim=tr.env.hist_dim+tr.env.tail_dim,actor_parameters=sum(p.numel() for p in tr.actor.parameters()),
                critic_parameters=sum(p.numel() for p in tr.critic.parameters()),initial_sha256=parameter_hash(tr.actor,tr.critic))
            if args.experiment=='eta':
                canonical=bucket_projection_hash(tr.env,tr.actor,tr.critic)
                item['canonical_initial_sha256']=canonical
                if seed in initial and initial[seed]!=canonical: raise AssertionError('Unmatched common initialization')
                initial[seed]=canonical
            else:
                item.update(actor_input_dim=tr.actor.net[0].in_features,critic_input_dim=tr.critic.net[0].in_features,
                    actor_od_dim=6*env.Z if spec.actor_od=='marginal' else 3*env.Z**2,
                    critic_od_dim=6*env.Z if spec.critic_od=='marginal' else 3*env.Z**2)
            manifest['initializations'].append(item)
    if args.dry_run:
        print(json.dumps(dict(status='dry_run_passed',initializations=manifest['initializations'],
            training_iterations=args.iterations,episodes=args.episodes,validation_algorithm=args.validation_algorithm,
            q_states=indices,q_replicates=args.q_replicates,
            batch_observations_gib=manifest['memory_gib_per_batch'],groups=list(specs)),indent=2)); return
    out.mkdir(parents=True,exist_ok=False)
    def save_manifest(): (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    save_manifest(); comparisons=[]
    for seed in args.seeds:
        for name,spec in specs.items():
            folder=out/f'{name}_seed{seed}'; folder.mkdir(); history=[]
            trainer=HistogramPPO(HistogramEnv(clone_snapshot(env),spec),cfg,seed); trainer.audit_gain=0.
            if args.resume_from:
                src=Path(args.resume_from)/folder.name
                commits=sorted(src.glob('iteration_*.commit.json')) if src.exists() else []
                if commits:
                    last=json.loads(commits[-1].read_text()); end=last['iteration']
                    for marker in commits:
                        info=json.loads(marker.read_text())
                        for fn,h in info['files_sha256'].items():
                            if sha256(src/fn)!=h: raise ValueError('Resume checkpoint corruption')
                            shutil.copy2(src/fn,folder/fn)
                        shutil.copy2(marker,folder/marker.name)
                    committed={r['iteration'] for r in read_complete_jsonl(src/'validation.jsonl')} if (src/'validation.jsonl').exists() else set()
                    qcommitted={r['iteration'] for r in read_complete_jsonl(src/'q_summary.jsonl')} if (src/'q_summary.jsonl').exists() else set()
                    for fn in ('train.jsonl','validation.jsonl','validation_episodes.jsonl','fixed_states.jsonl','q_pairs.jsonl','q_states.jsonl','q_summary.jsonl'):
                        if (src/fn).exists():
                            for r in read_complete_jsonl(src/fn):
                                if (r['iteration']<=end and (fn=='train.jsonl' or r['iteration'] in committed)
                                        and (not fn.startswith('q_') or r['iteration'] in qcommitted)):
                                    append(folder/fn,r)
                    trainer=HistogramPPO.from_checkpoint(HistogramEnv(clone_snapshot(env),spec),folder/last['checkpoint'],device=cfg.device)
                    trainer.fit_actor.load_state_dict(torch.load(folder/last['fit_actor'],map_location='cpu',weights_only=True))
                    trainer.audit_gain=last['audit_gain']
                    history=read_complete_jsonl(folder/'validation.jsonl') if (folder/'validation.jsonl').exists() else []
            def progress(row):
                r=dict(row,group=name,seed=seed,timestamp=datetime.now(timezone.utc).isoformat())
                append(folder/'progress.jsonl',r); print(json.dumps(r),flush=True)
            trainer.progress_callback=progress
            def save_checkpoint():
                it=trainer.iteration; cp=folder/f'iteration_{it:06d}.pt'; fit=folder/f'iteration_{it:06d}.fit_actor.pt'
                marker=folder/f'iteration_{it:06d}.commit.json'
                if not marker.exists():
                    tmp=cp.with_suffix('.pending'); trainer.save_checkpoint(tmp); tmp.replace(cp)
                    tmp=fit.with_suffix('.pending')
                    with tmp.open('xb') as f: torch.save(trainer.fit_actor.state_dict(),f)
                    tmp.replace(fit)
                    marker.write_text(json.dumps(dict(iteration=it,checkpoint=cp.name,fit_actor=fit.name,audit_gain=trainer.audit_gain,
                        representation=asdict(spec),minute_lambda=args.minute_lambda,
                        files_sha256={p.name:sha256(p) for p in (cp,fit)})),encoding='utf-8')
                return cp
            def q_evaluate(tr,it):
                if not args.q_replicates: return
                critic=copy.deepcopy(tr.critic).cpu().eval() if tr.device.type=='cuda' else tr.critic
                rows=[]
                for idx,base in snapshots.items():
                    snap=HistogramEnv(clone_snapshot(base),spec)
                    for rep in range(args.q_replicates):
                        seeds=dict(zip(('exogenous','travel','policy'),map(int,np.random.SeedSequence([args.q_seed,idx,rep]).generate_state(3))))
                        r=dict(paired_probe(snap,int(actions[idx]),tr,seeds,critic),iteration=it,group=name,seed=seed,bank_row=idx,replicate=rep)
                        append(folder/'q_pairs.jsonl',r); rows.append(r)
                if rows:
                    states,summary=summarize_q(rows)
                    for r in states: append(folder/'q_states.jsonl',dict(r,iteration=it,group=name,seed=seed))
                    append(folder/'q_summary.jsonl',dict(summary,iteration=it,group=name,seed=seed))
            def evaluate():
                it=trainer.iteration; cp=save_checkpoint(); results=[]
                evaluator=HistogramEnv(clone_snapshot(env),spec)
                actor=copy.deepcopy(trainer.actor).cpu().eval() if trainer.device.type=='cuda' else trainer.actor
                for s in scenarios:
                    policy_seed=None
                    if args.validation_algorithm=='ppo_sampled':
                        policy_seed=int(np.random.SeedSequence([seed,s['policy_seed']]).generate_state(1)[0])
                    scenario=adapt_scenario(evaluator,s,policy_seed)
                    r=evaluate_scenario(evaluator,scenario,args.validation_algorithm,actor,checkpoint_hash=sha256(cp))
                    r.update(iteration=it,group=name,seed=seed,exogenous_scenario_id=s['scenario_id'])
                    append(folder/'validation_episodes.jsonl',r); results.append(r['profit_total'])
                for r in probe_states(trainer,snapshots,actor): append(folder/'fixed_states.jsonl',dict(r,iteration=it,group=name,seed=seed))
                if it>0 and (it%args.q_every==0 or it==args.iterations): q_evaluate(trainer,it)
                row=dict(iteration=it,group=name,seed=seed,validation_profit_mean=float(np.mean(results)),
                    validation_profit_std=float(np.std(results,ddof=1)))
                append(folder/'validation.jsonl',row); history.append(row); print(json.dumps(row),flush=True)
            if not any(r['iteration']==trainer.iteration for r in history): evaluate()
            def should_stop():
                if not args.early_stop_patience: return False
                best_index=max(range(len(history)),key=lambda i:history[i]['validation_profit_mean'])
                return len(history)-1-best_index>=args.early_stop_patience

            def record(row):
                trainer.audit_gain=row['gain_estimate_per_period']; append(folder/'train.jsonl',dict(row,group=name,seed=seed))
                print(json.dumps(dict(group=name,seed=seed,iteration=trainer.iteration,train_profit=trainer.audit_gain)),flush=True)
                if trainer.iteration%args.eval_every==0 or trainer.iteration==args.iterations:
                    evaluate()
                    if should_stop(): return False
            with physical_advantage('minute_gae',env.N,args.minute_lambda):
                if not should_stop():
                    trainer.train(args.iterations-trainer.iteration,args.episodes,seed,on_iteration=record)
            final_iteration=trainer.iteration
            early_stopped=final_iteration<args.iterations and should_stop()
            if early_stopped:
                progress(dict(stage='early_stop',iteration=final_iteration,patience=args.early_stop_patience))
            if args.continue_l0:
                if final_iteration!=args.iterations:
                    raise RuntimeError('Continuation ended before the target iteration')
                actor_path=folder/f'iteration_{final_iteration:06d}.actor.pt'
                torch.save(trainer.actor.state_dict(),actor_path)
            best=max(history,key=lambda r:(r['validation_profit_mean'],-r['iteration']))
            qhistory=read_complete_jsonl(folder/'q_summary.jsonl') if (folder/'q_summary.jsonl').exists() else []
            if early_stopped and args.q_replicates and not any(r['iteration']==final_iteration for r in qhistory):
                q_evaluate(trainer,final_iteration)
                qhistory=read_complete_jsonl(folder/'q_summary.jsonl') if (folder/'q_summary.jsonl').exists() else []
            if args.q_replicates and best['iteration']>0 and not any(r['iteration']==best['iteration'] for r in qhistory):
                it=best['iteration']; info=json.loads((folder/f'iteration_{it:06d}.commit.json').read_text())
                selected=HistogramPPO.from_checkpoint(HistogramEnv(clone_snapshot(env),spec),folder/info['checkpoint'],device=cfg.device)
                selected.fit_actor.load_state_dict(torch.load(folder/info['fit_actor'],map_location='cpu',weights_only=True)); selected.audit_gain=info['audit_gain']
                q_evaluate(selected,it)
            selection=dict(best,final_iteration=final_iteration,early_stopped=early_stopped,checkpoint=str((folder/f"iteration_{best['iteration']:06d}.pt").resolve()))
            (folder/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
            comparison=dict(group=name,seed=seed,eta=spec.eta,actor_od=spec.actor_od,critic_od=spec.critic_od,
                final_iteration=final_iteration,early_stopped=early_stopped,final_profit=history[-1]['validation_profit_mean'],best_profit=best['validation_profit_mean'],best_iteration=best['iteration'])
            qhistory=read_complete_jsonl(folder/'q_summary.jsonl') if (folder/'q_summary.jsonl').exists() else []
            for label,it in (('final',final_iteration),('best',best['iteration'])):
                qr=next((r for r in qhistory if r['iteration']==it),None)
                if qr: comparison.update({label+'_'+k:v for k,v in qr.items() if k not in ('group','seed','iteration')})
            comparisons.append(comparison); write_csv(out/'comparison.csv',comparisons)
    manifest.update(status='complete'); save_manifest(); print(json.dumps(dict(output=str(out),comparison=comparisons)),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',choices=['eta','od'],default='eta')
    p.add_argument('--od-group',choices=['O0','O1','O2'],help='Train only the selected OD group')
    p.add_argument('--l0-only',action='store_true',help='Train only the dense L0 baseline')
    p.add_argument('--early-stop-patience',type=int,default=0,
        help='L0-only consecutive validation checks without a new best mean profit (0 disables)')
    p.add_argument('--run',default='runs/adp_aligned_evening_centered_seeds7_9_parallel_20261107_013749')
    p.add_argument('--output')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu',help='PPO update device; rollout and evaluation use CPU')
    p.add_argument('--rollout-backend',choices=['episode','batched'],default='episode',
        help='CPU rollout backend; batched OD training requires --workers 1')
    benchmarks=p.add_mutually_exclusive_group()
    benchmarks.add_argument('--benchmark-update',choices=['O0','O1','O2'],
        help='Time one CPU and one CUDA PPO update on the same CPU-sampled batch, then exit')
    benchmarks.add_argument('--benchmark-rollout',choices=['O0','O1','O2'],
        help='Compare CPU batched sampling with cold/reused CUDA Graph sampling and CUDA PPO updates')
    benchmarks.add_argument('--benchmark-cpu-batch',choices=['O0','O1','O2'],
        help='Compare multi-iteration CPU episode and CPU batched sampling with CUDA PPO updates')
    p.add_argument('--compare-graph-cache',action='store_true',
        help='With --benchmark-rollout, compare Graph reuse before/after PPO and after releasing unused CUDA cache')
    p.add_argument('--seeds',nargs='+',type=int,default=[42]); p.add_argument('--iterations',type=int,default=300)
    p.add_argument('--episodes',type=int,default=30); p.add_argument('--workers',type=int,default=2)
    p.add_argument('--hidden-dim',type=int,default=64)
    p.add_argument('--minute-lambda',type=float,default=.95); p.add_argument('--eval-every',type=int,default=10)
    p.add_argument('--validation-algorithm',choices=['ppo_greedy','ppo_sampled'],default='ppo_greedy',
        help='Policy action rule for validation and checkpoint selection')
    p.add_argument('--q-every',type=int,default=100); p.add_argument('--q-replicates',type=int,default=20)
    p.add_argument('--q-states',nargs='+',type=int); p.add_argument('--q-seed',type=int,default=20261110)
    p.add_argument('--snapshots',default='analysis/experiment_c_seed42_fixed_20261108_010144')
    p.add_argument('--bank',default='analysis/seed42_policy_drift_150_300')
    p.add_argument('--continue-l0',action='store_true',
        help='Continue a complete L0 run to a new target without early stopping, accepting source revisions')
    p.add_argument('--resume-from'); p.add_argument('--dry-run',action='store_true')
    return p


if __name__=='__main__':
    p=parser(); a=p.parse_args()
    if min(a.iterations,a.episodes,a.workers,a.hidden_dim,a.eval_every,a.q_every)<1 or a.q_replicates<0 or not 0<=a.minute_lambda<=1:
        p.error('Invalid positive counts or lambda')
    if len(set(a.seeds))!=len(a.seeds): p.error('Duplicate seeds')
    if a.q_states is not None and len(set(a.q_states))!=len(a.q_states): p.error('Duplicate states')
    run(a)
