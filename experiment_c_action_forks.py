"""Restore experiment-A states; paired finite-horizon Serve/Pass counterfactuals.

Future validation demand/order tapes are held fixed. Replicates resample ONLY
future event-keyed travel durations (including their coupled fares/costs).
Existing in-progress trips retain their sampled ETA and previously booked revenue.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from atomic_dual.data import load_environment, sha256
from atomic_dual.evaluation import load_scenario
from atomic_dual.ppo import AverageRewardPPO
from diagnose_policy_drift import write_csv


def reservoir_ordinals(total, size, seed):
    rng=np.random.default_rng(seed)
    result=[]
    for seen in range(1,total+1):
        pos=seen-1 if seen<=size else int(rng.integers(seen))
        if pos<size:
            if pos==len(result): result.append(seen)
            else: result[pos]=seen
    return result


def greedy(actor, state, mask):
    with torch.inference_mode():
        logits=actor(torch.as_tensor(state))
        return int(logits.masked_fill(~torch.as_tensor(mask),-torch.inf).softmax(-1).argmax())


def step_with_order(env, action, orders):
    _,reward,done,info=env.step(action)
    if not done and env._cursor==0:
        env._car_order=orders[env.t].copy()
    return reward,info


def clone_snapshot(snapshot):
    """Deepcopy separates NumPy views: rebuild cache from authoritative state."""
    env=copy.deepcopy(snapshot)
    if getattr(env,'_cache_enabled',False) and not env.done:
        env._refresh_observation_cache()
    return env


def branch(snapshot, forced_action, actor, orders, travel_seed):
    """Clone before acting. Return raw remaining profit, NOT total episode profit."""
    env=clone_snapshot(snapshot)
    env.episode_seed=int(travel_seed)
    start_profit=env.profit
    start_served=int(env.n_served.sum())
    revenue=cost=0.
    relocations=0
    first_reward,info=step_with_order(env,forced_action,orders)
    if env.decode(forced_action)[0]=='serve': revenue+=first_reward
    elif env.decode(forced_action)[0]=='relocate': cost-=first_reward; relocations+=1
    next_state=None if env.done else env.observe().copy()
    first_minutes=info['task_minutes']
    while not env.done:
        mask=env.action_mask()
        action=int(mask.argmax()) if mask.sum()==1 else greedy(actor,env.observe(),mask)
        kind=env.decode(action)[0]
        reward,_=step_with_order(env,action,orders)
        if kind=='serve': revenue+=reward
        elif kind=='relocate': cost-=reward; relocations+=1
    result=dict(remaining_profit=float(env.profit-start_profit),remaining_revenue=revenue,
                remaining_relocation_cost=cost,remaining_served=int(env.n_served.sum())-start_served,
                remaining_relocations=relocations,first_reward=first_reward,
                first_task_minutes=first_minutes)
    if not np.isclose(result['remaining_profit'],revenue-cost):
        raise AssertionError('Remaining reward ledger mismatch')
    return result,next_state


def mean_se(values):
    a=np.asarray(values,dtype=float)
    return float(a.mean()),float(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else None


def run(args):
    torch.set_num_threads(1)
    root=Path(args.run); bank=Path(args.bank); out=Path(args.output)
    manifest=json.loads((root/'manifest.json').read_text())
    a_meta=json.loads((bank/'results.json').read_text())
    with np.load(bank/'fixed_states.npz',allow_pickle=False) as f:
        states=f['states']; masks=f['masks']; metadata=f['metadata']
    with np.load(bank/'action_probabilities.npz',allow_pickle=False) as f:
        a150=f['150'].argmax(-1); a300=f['300'].argmax(-1)
    zones=(masks.shape[1]-1)//4
    eligible=(a150<3*zones)&(a300==4*zones)
    if not eligible.any(): raise ValueError('No Serve150/Pass300 states')
    scenarios=[load_scenario(p) for p in sorted((root/'validation_scenarios').glob('*.npz'))]
    if ([s['scenario_id'] for s in scenarios]!=a_meta['scenario_ids'] or
        a_meta['scenario_ids']!=manifest['validation_scenario_ids'] or
        any(s['data_split']!='validation' for s in scenarios)):
        raise ValueError('Validation scenario provenance mismatch')
    calibration=Path(manifest['arguments']['data'])/'evening.npz'
    if sha256(calibration)!=manifest['calibration_sha256']: raise ValueError('Calibration mismatch')
    env=load_environment(calibration,'evening',manifest['environment']['step_minutes'])
    env.cfg=replace(env.cfg,**manifest['environment']); env.enable_observation_cache()
    folder=root/'centered_seed42'
    cps={i:folder/f'iteration_{i:06d}.pt' for i in (150,220,300)}
    for i,cp in cps.items():
        if sha256(cp)!=a_meta['checkpoints'][str(i)]['sha256']:
            raise ValueError(f'Experiment A checkpoint mismatch: {i}')
    trainers={i:AverageRewardPPO.from_checkpoint(env,cp,for_evaluation=True) for i,cp in cps.items()}
    for tr in trainers.values(): tr.actor.eval(); tr.critic.eval()
    # Select uniformly within each source/date group, not by size of action margin.
    rng=np.random.default_rng(args.seed)
    chosen=[]
    for source,date in sorted({tuple(row[:2]) for row in metadata}):
        candidates=np.flatnonzero(eligible&(metadata[:,0]==source)&(metadata[:,1]==date))
        if len(candidates): chosen.extend(rng.choice(candidates,min(args.states_per_group,len(candidates)),replace=False).tolist())
    source_counts={}
    import csv
    with (bank/'trajectory_episodes.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            source_counts[int(row['iteration']),row['date_or_synthetic_trial_id']]=int(row['active'])
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(status='restoring',experiment_version='cache_rebound_v2',arguments=vars(args),n_eligible=int(eligible.sum()),selected_bank_rows=sorted(chosen),
        calibration_sha256=sha256(calibration),checkpoint_sha256={str(i):sha256(cp) for i,cp in cps.items()},
        bank_sha256={p:sha256(bank/p) for p in ('results.json','fixed_states.npz','action_probabilities.npz','trajectory_episodes.csv')},
        source_sha256={p:sha256(p) for p in ('experiment_c_action_forks.py','atomic_dual/environment.py','atomic_dual/ppo.py')},
        estimand='Conditional raw undiscounted remaining system profit; frozen validation demand/order tapes; resampled future travel only.',
        terminal='Original fixed horizon, gamma=1, terminal value=0, assignment-time booking including final instant.',
        continuation='Masked greedy checkpoint 150 or 300, matching experiments A/B; not sampled-policy Q.',
        opportunity_cost='Implicit in full-fleet future revenue/cost changes; no extra penalty added.',
        critic_probe='Raw units: reward difference + reward_scale*(h(nextServe)-h(nextPass)); terminal h=0; gain and h(current) cancel.',
        uncertainty='Per-state paired Monte Carlo SE/normal CI measures future-travel noise only; selected states/dates are dependent.')
    def save_protocol():
        (out/'manifest.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    save_protocol(); snapshots={}
    if args.snapshots_from:
        prior=Path(args.snapshots_from)
        previous=json.loads((prior/'manifest.json').read_text())
        for key in ('calibration_sha256','checkpoint_sha256','bank_sha256','selected_bank_rows'):
            if previous[key]!=protocol[key]: raise ValueError(f'Snapshot provenance mismatch: {key}')
        snapshot_path=prior/'trusted_local_snapshots.pt'
        if sha256(snapshot_path)!=previous['outputs_sha256']['trusted_local_snapshots.pt']:
            raise ValueError('Snapshot file fingerprint mismatch')
        saved=torch.load(snapshot_path,map_location='cpu',weights_only=False)
        snapshots={idx:clone_snapshot(s) for idx,s in saved['snapshots'].items()}
        for idx,snapshot in snapshots.items():
            if not np.array_equal(snapshot.observe(),states[idx]) or not np.array_equal(snapshot.action_mask(),masks[idx]):
                raise AssertionError(f'Restored snapshot mismatch: {idx}')
        protocol['reused_snapshot_sha256']=sha256(snapshot_path)
        print(json.dumps(dict(reused_snapshots=len(snapshots),cache_rebuilt=True)),flush=True)
    restore_groups=[] if args.snapshots_from else sorted({tuple(metadata[idx,:2]) for idx in chosen})
    for source,date_idx in restore_groups:
        scenario=scenarios[date_idx]
        ids=np.flatnonzero((metadata[:,0]==source)&(metadata[:,1]==date_idx))
        ordinals=reservoir_ordinals(source_counts[source,scenario['date']],
            a_meta['sampling']['per_source_date'],a_meta['sampling']['seed_base']+source*10+date_idx)
        if len(ids)!=len(ordinals): raise AssertionError('Reservoir size mismatch')
        wanted={ordinals[pos]:int(idx) for pos,idx in enumerate(ids) if idx in chosen}
        env.request_batches=scenario['requests'].copy()
        env.reset(scenario['seed'],initial_regions=scenario['initial_regions'])
        env._car_order=scenario['orders'][0].copy()
        seen=0
        while not env.done:
            mask=env.action_mask()
            if mask.sum()>1:
                seen+=1
                if seen in wanted:
                    idx=wanted[seen]
                    if not np.array_equal(env.observe(),states[idx]) or not np.array_equal(mask,masks[idx]):
                        raise AssertionError(f'Restored observation/mask mismatch: bank row {idx}')
                    if greedy(trainers[150].actor,env.observe(),mask)!=int(a150[idx]) or greedy(trainers[300].actor,env.observe(),mask)!=env.pass_action:
                        raise AssertionError('Restored action disagreement mismatch')
                    snapshots[idx]=clone_snapshot(env)
                action=greedy(trainers[source].actor,env.observe(),mask)
            else: action=int(mask.argmax())
            step_with_order(env,action,scenario['orders'])
        if seen!=source_counts[source,scenario['date']]: raise AssertionError('Active-state count mismatch')
        print(json.dumps(dict(restored_source=int(source),date=scenario['date'],restored=len(wanted))),flush=True)
    if set(snapshots)!=set(chosen): raise AssertionError('Incomplete restoration')
    # Trusted local snapshots allow follow-up probes without reconstructing observations.
    torch.save(dict(snapshots=snapshots,selected_bank_rows=chosen),out/'trusted_local_snapshots.pt')
    protocol.update(status='simulating',all_restored_states_exact=True); save_protocol()
    samples=[]; summaries=[]
    with torch.inference_mode():
        for idx in sorted(chosen):
            snapshot=snapshots[idx]; scenario=scenarios[metadata[idx,1]]
            for continuation in (150,300):
                state_samples=[]
                for rep in range(args.replicates):
                    travel_seed=int(np.random.SeedSequence([args.seed,idx,rep]).generate_state(1)[0])
                    serve,ns=branch(snapshot,int(a150[idx]),trainers[continuation].actor,scenario['orders'],travel_seed)
                    stay,np_=branch(snapshot,snapshot.pass_action,trainers[continuation].actor,scenario['orders'],travel_seed)
                    row=dict(bank_row=idx,source_iteration=int(metadata[idx,0]),date=scenario['date'],
                        t=snapshot.t,cursor=snapshot._cursor,service_class=int(snapshot.current_car().service_class),
                        continuation=continuation,replicate=rep,travel_seed=travel_seed,serve_action=int(a150[idx]),
                        q_serve=serve['remaining_profit'],q_pass=stay['remaining_profit'],
                        delta_q=serve['remaining_profit']-stay['remaining_profit'])
                    for name,values in (('serve',serve),('pass',stay)):
                        row.update({name+'_'+key:value for key,value in values.items()})
                    for critic_iteration in (150,300):
                        critic=trainers[critic_iteration].critic
                        hs=0. if ns is None else float(critic(torch.as_tensor(ns)).squeeze())
                        hp=0. if np_ is None else float(critic(torch.as_tensor(np_)).squeeze())
                        row[f'critic_{critic_iteration}_delta_td_raw']=serve['first_reward']-stay['first_reward']+trainers[critic_iteration].cfg.reward_scale*(hs-hp)
                    samples.append(row); state_samples.append(row)
                    with (out/'paired_rollouts.jsonl').open('a',encoding='utf-8') as f:
                        f.write(json.dumps(row,allow_nan=False)+'\n')
                dq,se=mean_se([r['delta_q'] for r in state_samples])
                summary={k:state_samples[0][k] for k in ('bank_row','source_iteration','date','t','cursor','service_class','continuation','serve_action')}
                summary.update(replicates=args.replicates,delta_q_mean=dq,delta_q_mc_se=se,
                    delta_q_mc_ci95_low=None if se is None else dq-1.96*se,
                    delta_q_mc_ci95_high=None if se is None else dq+1.96*se,
                    q_serve_mean=float(np.mean([r['q_serve'] for r in state_samples])),
                    q_pass_mean=float(np.mean([r['q_pass'] for r in state_samples])))
                for ci in (150,300): summary[f'critic_{ci}_delta_td_raw_mean']=float(np.mean([r[f'critic_{ci}_delta_td_raw'] for r in state_samples]))
                summaries.append(summary)
                write_csv(out/'state_summary.csv',summaries)
                print(json.dumps(summary),flush=True)
    write_csv(out/'paired_rollouts.csv',samples)
    aggregate=[]
    for continuation in (150,300):
        subset=[r for r in summaries if r['continuation']==continuation]
        aggregate.append(dict(continuation=continuation,n_states=len(subset),
            mean_delta_q=float(np.mean([r['delta_q_mean'] for r in subset])),
            fraction_positive_mean=float(np.mean([r['delta_q_mean']>0 for r in subset])),
            fraction_mc_ci_above_zero=float(np.mean([r['delta_q_mc_ci95_low'] is not None and r['delta_q_mc_ci95_low']>0 for r in subset])),
            fraction_mc_positive_but_critic300_negative=float(np.mean([r['delta_q_mean']>0 and r['critic_300_delta_td_raw_mean']<0 for r in subset]))))
    (out/'aggregate.json').write_text(json.dumps(aggregate,indent=2),encoding='utf-8')
    protocol.update(status='complete',outputs_sha256={p.name:sha256(p) for p in out.iterdir() if p.name!='manifest.json'})
    save_protocol(); print(json.dumps(dict(output=str(out),aggregate=aggregate)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--bank',default='analysis/seed42_policy_drift_150_300')
    parser.add_argument('--output',required=True)
    parser.add_argument('--states-per-group',type=int,default=1,help='Uniform eligible states per source policy/date (15 groups).')
    parser.add_argument('--replicates',type=int,default=8)
    parser.add_argument('--seed',type=int,default=20261108)
    parser.add_argument('--snapshots-from',help='Prior local experiment-C output directory; verify and rebuild its trusted snapshots, skipping source replay.')
    args=parser.parse_args()
    if min(args.states_per_group,args.replicates)<1 or args.seed<0: parser.error('Positive counts and nonnegative seed required')
    run(args)
