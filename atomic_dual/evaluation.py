"""Frozen exogenous scenarios, paired evaluation and reproducible baseline reports."""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .baselines import BASELINES, VERSION, choose_action
from .data import load_environment, sha256, configure_evaluation_environment
from .ppo import AverageRewardPPO

METRICS = ('profit_total','revenue_total','relocation_cost_total','requests_arrived','requests_served',
    'requests_rejected','service_rate','served_Eonly','served_Ponly','served_D_by_E','served_D_by_P',
    'dual_share_E','dual_share_P','express_utilization','premium_utilization',
    'relocation_count','relocation_minutes','relocation_elapsed_minutes','wallclock_eval_seconds')


def scenario_digest(scenario):
    h = hashlib.sha256()
    for name in ('period','data_split','date','seed','calibration_sha256','env_config','policy_seed'):
        h.update(json.dumps([name,scenario[name]],sort_keys=True).encode())
    if 'research_settings' in scenario:
        h.update(json.dumps(scenario['research_settings'],sort_keys=True).encode())
    for name in ('requests','orders','initial_regions'):
        arr = np.ascontiguousarray(scenario[name],dtype=np.int64)
        h.update(str((name,arr.shape)).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def make_scenario(env, seed, period, calibration_hash, requests=None, orders=None, data_split='simulated', date=None):
    streams = np.random.SeedSequence(seed).spawn(3)
    demand_rng,order_rng = [np.random.default_rng(s) for s in streams[:2]]
    result = dict(period=period,data_split=data_split,date=date,seed=int(seed),
        calibration_sha256=calibration_hash,env_config=asdict(env.cfg),
        requests=(demand_rng.poisson(env.arrival_rates).astype(np.int64) if requests is None else np.array(requests,copy=True)),
        orders=((np.stack([order_rng.permutation(env.N) for _ in range(env.T)]) if env.cfg.random_order
                 else np.tile(np.arange(env.N),(env.T,1))) if orders is None else np.array(orders,copy=True)),
        initial_regions=env.sample_initial_regions(seed),policy_seed=int(streams[2].generate_state(1)[0]))
    if env.research_settings(): result['research_settings'] = env.research_settings()
    result['scenario_id'] = scenario_digest(result)
    return result


def save_scenario(scenario,path):
    meta = {k:v for k,v in scenario.items() if not isinstance(v,np.ndarray)}
    with Path(path).open('xb') as f:
        np.savez_compressed(f,requests=scenario['requests'],orders=scenario['orders'],
            initial_regions=scenario['initial_regions'],metadata_json=json.dumps(meta,sort_keys=True))


def load_scenario(path):
    with np.load(path,allow_pickle=False) as f:
        result = json.loads(str(f['metadata_json']))
        result.update({k:f[k].copy() for k in ('requests','orders','initial_regions')})
    if result['scenario_id'] != scenario_digest(result): raise ValueError('Scenario fingerprint mismatch')
    return result


def evaluate_scenario(env, scenario, algorithm, actor=None, lookahead_steps=10, checkpoint_hash=None):
    if scenario['env_config'] != asdict(env.cfg): raise ValueError('Scenario environment mismatch')
    if scenario.get('research_settings',{}) != env.research_settings(): raise ValueError('Scenario research settings mismatch')
    requests,orders = scenario['requests'],scenario['orders']
    if (requests.shape != env.arrival_rates.shape or np.any(requests < 0) or not np.isfinite(requests).all()
            or np.any(requests != np.floor(requests)) or orders.shape != (env.T,env.N)
            or not np.array_equal(np.sort(orders,axis=1),np.tile(np.arange(env.N),(env.T,1)))
            or not np.array_equal(scenario['initial_regions'],env.sample_initial_regions(scenario['seed']))
            or scenario['scenario_id'] != scenario_digest(scenario)):
        raise ValueError('Corrupt scenario arrays or fingerprint')
    if algorithm not in BASELINES and algorithm not in ('ppo_greedy','ppo_sampled'):
        raise ValueError('Unknown algorithm')
    if algorithm.startswith('ppo') and actor is None: raise ValueError('PPO requires a checkpoint actor')
    env.request_batches = requests.astype(np.int64,copy=True)
    env.reset(scenario['seed'], initial_regions=scenario['initial_regions'])
    env._car_order = orders[0].copy()
    policy_rng = np.random.default_rng(scenario['policy_seed'])
    revenue=cost=relocation_minutes=elapsed_relocation=0.
    relocations=forced=0
    occupied = np.zeros(2)
    start = time.perf_counter()
    while not env.done:
        mask = env.action_mask()
        forced += int(mask.sum()==1)
        if algorithm in BASELINES:
            action = choose_action(algorithm,env,lookahead_steps)
        elif mask.sum()==1:
            probs=mask.astype(float)
            action=int(np.argmax(probs)) if algorithm=='ppo_greedy' else int(policy_rng.choice(env.action_dim,p=probs))
        else:
            with torch.no_grad():
                device = next(actor.parameters()).device
                logits = actor(torch.as_tensor(env.observe(),device=device))
                probs = logits.masked_fill(~torch.as_tensor(mask,device=device),-torch.inf).softmax(-1).cpu().numpy().astype(float)
                probs /= probs.sum()
                action = int(np.argmax(probs)) if algorithm=='ppo_greedy' else int(policy_rng.choice(env.action_dim,p=probs))
        car = env.current_car()
        k,origin,t = int(car.service_class),car.region,env.t
        kind,q,dest = env.decode(action)
        intervals = env.T-t-(env.cfg.batch_timing=='closing')
        _,reward,_,info = env.step(action)
        if kind=='serve':
            revenue += reward
            occupied[k] += min(info['task_steps'],intervals)*env.cfg.step_minutes
        elif kind=='relocate':
            cost -= reward
            relocations += 1
            relocation_minutes += info['task_minutes']
            elapsed_relocation += min(info['task_steps'],intervals)*env.cfg.step_minutes
        if not env.done and env._cursor==0:
            env._car_order = orders[env.t].copy()
    wallclock = time.perf_counter()-start
    served = env.n_served
    dual = int(served[:,2].sum())
    arrived_by_type = requests.sum(axis=(0,1,2))
    if env.atomic_count != env.T*env.N or env.n_arrived != int(requests.sum()) or env.n_arrived != int(served.sum())+env.n_rejected:
        raise AssertionError('Atomic or order conservation failed')
    if not np.isclose(env.profit,revenue-cost): raise AssertionError('Financial ledger mismatch')
    policy_config = dict(algorithm=algorithm,version=VERSION,lookahead_steps=lookahead_steps,checkpoint_sha256=checkpoint_hash)
    config = dict(env=asdict(env.cfg),policy=policy_config,calibration=scenario['calibration_sha256'])
    if env.research_settings(): config['research_settings']=env.research_settings()
    config_hash = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    return dict(experiment_id='stage3',algorithm=algorithm,algorithm_version=VERSION,config_hash=config_hash,
        scenario_id=scenario['scenario_id'],calibration_sha256=scenario['calibration_sha256'],checkpoint_sha256=checkpoint_hash,
        period=scenario['period'],seed=scenario['seed'],policy_seed=scenario['policy_seed'],
        data_split=scenario['data_split'],date_or_synthetic_trial_id=scenario['date'] or f"simulated_seed_{scenario['seed']}",
        profit_total=float(env.profit),revenue_total=revenue,relocation_cost_total=cost,
        requests_arrived=env.n_arrived,requests_served=int(served.sum()),requests_rejected=env.n_rejected,
        service_rate=float(served.sum()/env.n_arrived) if env.n_arrived else 0.,
        arrived_Eonly=int(arrived_by_type[0]),arrived_Ponly=int(arrived_by_type[1]),arrived_D=int(arrived_by_type[2]),
        rejected_Eonly=int(arrived_by_type[0]-served[:,0].sum()),
        rejected_Ponly=int(arrived_by_type[1]-served[:,1].sum()),rejected_D=int(arrived_by_type[2]-served[:,2].sum()),
        served_Eonly=int(served[:,0].sum()),served_Ponly=int(served[:,1].sum()),
        served_D_by_E=int(served[0,2]),served_D_by_P=int(served[1,2]),
        dual_share_E=float(served[0,2]/dual) if dual else None,dual_share_P=float(served[1,2]/dual) if dual else None,
        express_utilization=float(occupied[0]/(env.cfg.n_express*env.T*env.cfg.step_minutes)) if env.cfg.n_express else None,
        premium_utilization=float(occupied[1]/(env.cfg.n_premium*env.T*env.cfg.step_minutes)) if env.cfg.n_premium else None,
        relocation_count=relocations,relocation_minutes=relocation_minutes,relocation_elapsed_minutes=elapsed_relocation,
        atomic_steps=env.atomic_count,forced_action_fraction=forced/env.atomic_count,wallclock_eval_seconds=wallclock,
        wallclock_train_seconds=None)


def _stats(values):
    values = [float(v) for v in values if v is not None]
    return dict(n=len(values),mean=float(np.mean(values)) if values else None,
                std=float(np.std(values,ddof=1)) if len(values)>1 else None)


def summarize(rows,reference='greedy_fare'):
    grouped = {}
    for row in rows:
        key = (row['period'],row['data_split'],row['algorithm'])
        bucket = grouped.setdefault(key,{})
        if row['scenario_id'] in bucket: raise ValueError('Duplicate scenario/algorithm row')
        bucket[row['scenario_id']] = row
    summary,paired = [],[]
    for (period,split,algorithm),bucket in sorted(grouped.items()):
        summary.append(dict(period=period,data_split=split,algorithm=algorithm,
            metrics={name:_stats([row[name] for row in bucket.values()]) for name in METRICS if all(name in row for row in bucket.values())}))
        if algorithm==reference: continue
        ref = grouped.get((period,split,reference))
        if ref is None or bucket.keys()!=ref.keys(): raise ValueError('Paired methods require identical complete scenario sets')
        for sid in bucket:
            for name in ('calibration_sha256','requests_arrived'):
                if bucket[sid].get(name)!=ref[sid].get(name): raise ValueError('Paired inputs differ')
        paired.append(dict(period=period,data_split=split,algorithm=algorithm,reference=reference,
            difference='algorithm_minus_reference',**_stats([bucket[sid]['profit_total']-ref[sid]['profit_total'] for sid in sorted(bucket)])))
    return dict(summary=summary,paired_profit_differences=paired)


def write_report(rows,output):
    output = Path(output)
    result = summarize(rows)
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    with (output/'episodes.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    reference={(r['period'],r['data_split'],r['scenario_id']):r for r in rows if r['algorithm']=='greedy_fare'}
    with (output/'paired.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['period','data_split','scenario_id','algorithm','reference','profit_difference'])
        writer.writeheader()
        for r in rows:
            if r['algorithm']=='greedy_fare': continue
            ref=reference[(r['period'],r['data_split'],r['scenario_id'])]
            writer.writerow(dict(period=r['period'],data_split=r['data_split'],scenario_id=r['scenario_id'],
                algorithm=r['algorithm'],reference='greedy_fare',profit_difference=r['profit_total']-ref['profit_total']))
    lines = ['# Paired scenario evaluation','',
        'Calibrated simulation or explicitly selected replay; n counts scenarios, not training seeds. SD is sample SD.', '',
        '| Period | Split | Algorithm | n | Profit mean | Profit SD | Service rate mean |',
        '|---|---|---|---:|---:|---:|---:|']
    for s in result['summary']:
        profit = s['metrics']['profit_total']; service=s['metrics']['service_rate']['mean']
        sd = f"{profit['std']:.3f}" if profit['std'] is not None else 'NA'
        lines.append(f"| {s['period']} | {s['data_split']} | {s['algorithm']} | {profit['n']} | {profit['mean']:.3f} | {sd} | {service:.4f} |")
    lines += ['', '| Period | Split | Algorithm − Greedy | n paired | Mean profit difference | SD |', '|---|---|---|---:|---:|---:|']
    for p in result['paired_profit_differences']:
        sd = f"{p['std']:.3f}" if p['std'] is not None else 'NA'
        lines.append(f"| {p['period']} | {p['data_split']} | {p['algorithm']} | {p['n']} | {p['mean']:.3f} | {sd} |")
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data')
    p.add_argument('--period',choices=['morning','evening'])
    p.add_argument('--summarize-jsonl',help='Regenerate tables from existing evaluation rows without simulation')
    p.add_argument('--seeds',nargs='+',type=int,default=[7,11,19])
    p.add_argument('--algorithms',nargs='+',choices=list(BASELINES)+['ppo_greedy','ppo_sampled'],default=list(BASELINES))
    p.add_argument('--historical-dates',nargs='+',help='Explicit November dates; omitted means simulated calibration trials')
    p.add_argument('--checkpoint')
    p.add_argument('--device',default='cpu')
    p.add_argument('--lookahead-steps',type=int,default=10)
    p.add_argument('--scenario-dir',help='Replay frozen scenarios from an earlier evaluation')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    if args.summarize_jsonl:
        rows=[json.loads(line) for line in Path(args.summarize_jsonl).read_text(encoding='utf-8').splitlines() if line.strip()]
        if not rows: p.error('No evaluation rows')
        output=Path(args.output)
        output.mkdir(parents=True,exist_ok=False)
        write_report(rows,output)
        (output/'source.json').write_text(json.dumps(dict(jsonl=str(Path(args.summarize_jsonl).resolve()),sha256=sha256(args.summarize_jsonl)),indent=2))
        return
    if not args.data or not args.period: p.error('--data and --period required for evaluation')
    if len(set(args.seeds))!=len(args.seeds) or len(set(args.algorithms))!=len(args.algorithms): p.error('Duplicate seeds or algorithms')
    if 'greedy_fare' not in args.algorithms: p.error('Include greedy_fare for paired reference')
    if args.lookahead_steps<1: p.error('Positive lookahead required')
    torch.set_num_threads(1)
    data,output=Path(args.data),Path(args.output)
    path=data/f'{args.period}.npz'
    with np.load(path,allow_pickle=False) as f: meta=json.loads(str(f['metadata_json']))
    env=load_environment(path,args.period,meta['step_minutes'])
    checkpoint_hash=sha256(args.checkpoint) if args.checkpoint else None
    actor=None
    if any(a.startswith('ppo') for a in args.algorithms):
        if not args.checkpoint: p.error('PPO algorithms require --checkpoint')
        actor=AverageRewardPPO.from_checkpoint(env,args.checkpoint,device=args.device,for_evaluation=True).actor.eval()
    scenarios=[]
    if args.scenario_dir:
        if args.historical_dates: p.error('Use either frozen scenarios or historical dates')
        scenarios=[load_scenario(f) for f in sorted(Path(args.scenario_dir).glob('*.npz'))]
        if not scenarios: p.error('No frozen scenarios')
        if any(s['calibration_sha256']!=sha256(path) or s['period']!=args.period for s in scenarios): p.error('Frozen scenario calibration mismatch')
        splits={s['data_split'] for s in scenarios}
        if 'test' in splits and len(splits)>1: p.error('Test scenarios require a separate evaluation invocation')
        if 'test' in splits: configure_evaluation_environment(env,path,'test')
    else:
        if args.historical_dates: configure_evaluation_environment(env,path,'test')
        for date,seed in itertools.product(args.historical_dates or [None],args.seeds):
            requests=None
            if date:
                replay=data/f'{args.period}_replays/test_{date}.npz'
                requests=load_environment(path,args.period,meta['step_minutes'],replay).request_batches
            scenarios.append(make_scenario(env,seed,args.period,sha256(path),requests=requests,
                data_split='test' if date else 'simulated_from_train_calibration',date=date))
    output.mkdir(parents=True,exist_ok=False)
    (output/'scenarios').mkdir()
    for i,s in enumerate(scenarios): save_scenario(s,output/'scenarios'/f'{i:04d}.npz')
    manifest=dict(arguments=vars(args),calibration_sha256=sha256(path),policy_version=VERSION,
        checkpoint_sha256=checkpoint_hash,versions=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__),
        scenario_ids=[s['scenario_id'] for s in scenarios],status='running',
        source_sha256={name:sha256(Path(__file__).with_name(name)) for name in ('baselines.py','evaluation.py','environment.py','ppo.py')})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    rows=[]
    for s,algorithm in itertools.product(scenarios,args.algorithms):
        row=evaluate_scenario(env,s,algorithm,actor,args.lookahead_steps,checkpoint_hash if algorithm.startswith('ppo') else None)
        rows.append(row)
        with (output/'episodes.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps({k:row[k] for k in ('algorithm','seed','period','profit_total','requests_served','wallclock_eval_seconds')}),flush=True)
    write_report(rows,output)
    manifest.update(status='complete',outputs_sha256={str(f.relative_to(output)):sha256(f) for f in sorted(output.rglob('*')) if f.is_file() and f.name!='manifest.json'})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__': main()
