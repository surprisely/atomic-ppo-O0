"""Frozen-network critic consistency probes under training-like future dynamics.

D1: sampled P300, float32 centered MC training-target formula.
D2: greedy P300, same formula. D3: SAME greedy trajectories, raw net profit.
Current validation snapshot is fixed; future demand/order use native training RNG.
Critic300 was fitted before Actor300 update (Actor299), so D1 is not exact
training-policy evaluation unless a separately available fitted-policy checkpoint
is supplied via --fit-checkpoint. No networks are updated by this script.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from atomic_dual.data import sha256
from atomic_dual.ppo import AverageRewardPPO, masked_categorical
from experiment_c_action_forks import clone_snapshot, greedy
from diagnose_policy_drift import write_csv


def verify_cache(env):
    if not getattr(env,'_cache_enabled',False) or env.done: return
    cached=env.observe()
    env._cache_enabled=False
    try: reference=env.observe()
    finally: env._cache_enabled=True
    if not np.array_equal(cached,reference): raise AssertionError('Cached observation mismatch')


@torch.inference_mode()
def rollout_branch(snapshot, forced_action, actor, critic, mode, seeds, gain, scale, record_trace=False):
    env=clone_snapshot(snapshot)
    verify_cache(env)
    # Native _begin_timestamp draws Poisson arrivals then a fleet permutation.
    # No future historical validation replay is visible to the branch.
    env.request_batches=None
    env.rng=np.random.default_rng(seeds['exogenous'])
    env.episode_seed=int(seeds['travel'])
    generator=torch.Generator(device='cpu').manual_seed(int(seeds['policy']))
    remaining=env.T*env.N-env.atomic_count
    start=env.profit; rewards=[]; revenue=cost=0.
    h0=float(critic(torch.as_tensor(env.observe())).squeeze())
    first_reward=None; next_value=None
    trace_observations=[]; trace_values=[]
    def flush_trace():
        if trace_observations:
            trace_values.extend(critic(torch.as_tensor(np.stack(trace_observations))).squeeze(-1).cpu().numpy().tolist())
            trace_observations.clear()
    exogenous_hash=hashlib.sha256()
    action=forced_action
    while not env.done:
        if record_trace:
            trace_observations.append(env.observe())
            if len(trace_observations)>=2048: flush_trace()
        kind=env.decode(action)[0]
        _,reward,done,_=env.step(action)
        rewards.append(reward)
        if kind=='serve': revenue+=reward
        elif kind=='relocate': cost-=reward
        if not done and env._cursor==0:
            exogenous_hash.update(np.asarray([env.t],dtype='<i8').tobytes())
            exogenous_hash.update(env._batch_arrivals.astype('<i8').tobytes())
            exogenous_hash.update(env._car_order.astype('<i8').tobytes())
        if first_reward is None:
            first_reward=reward
            verify_cache(env)
            next_value=0. if done else float(critic(torch.as_tensor(env.observe())).squeeze())
        if done: break
        mask=env.action_mask()
        if mode=='sampled':
            # Same masked categorical distribution and multinomial kernel as training.
            # Consume one draw even on forced busy Pass, as the episode sampler does.
            if mask.sum()==1:
                probs=torch.as_tensor(mask,dtype=torch.float32)
            else:
                probs=masked_categorical(actor(torch.as_tensor(env.observe()).unsqueeze(0)),
                    torch.as_tensor(mask).unsqueeze(0)).probs[0]
            action=int(torch.multinomial(probs,1,replacement=True,generator=generator).item())
        elif mode=='greedy':
            action=int(mask.argmax()) if mask.sum()==1 else greedy(actor,env.observe(),mask)
        else: raise ValueError(mode)
    if len(rewards)!=remaining: raise AssertionError('Remaining atomic horizon differs')
    raw=env.profit-start
    if not np.isclose(raw,revenue-cost,atol=1e-7,rtol=1e-10): raise AssertionError('Reward ledger mismatch')
    g= gain/(env.T*env.N*scale)
    # Match ppo.update float32 rewards, scaled adjustment and reverse cumsum.
    adjusted=np.asarray(rewards,dtype=np.float32)/scale-g
    training_target=float(np.cumsum(adjusted[::-1],dtype=np.float32)[-1])
    td=float(np.float32(first_reward)/scale-g+next_value-h0)
    result=dict(q_raw=float(raw),q_centered_raw=float(raw-remaining*gain/(env.T*env.N)),
        q_training_target_raw=training_target*scale,q_training_target_scaled=training_target,
        td_raw=td*scale,td_scaled=td,first_reward=first_reward,
        remaining_atoms=remaining,revenue=revenue,relocation_cost=cost,
        exogenous_sha256=exogenous_hash.hexdigest())
    if record_trace:
        flush_trace()
        values=np.asarray(trace_values+[0.],dtype=np.float64)
        values[0]=h0
        if len(values)>1: values[1]=next_value
        if len(values)!=remaining+1: raise AssertionError('Trace value horizon mismatch')
        result['trace']=dict(rewards_raw=np.asarray(rewards,dtype=np.float64),
            adjusted_scaled=adjusted.copy(),values_scaled=values)
    return result


def paired_row(state_id, rep, arm, serve, stay):
    if serve['exogenous_sha256']!=stay['exogenous_sha256']:
        raise AssertionError('Serve/Pass did not share future exogenous demand/order')
    raw=serve['q_raw']-stay['q_raw']
    centered=serve['q_centered_raw']-stay['q_centered_raw']
    if not np.isclose(raw,centered,atol=1e-7,rtol=1e-10): raise AssertionError('Gain baseline failed to cancel')
    mc=raw if arm=='D3' else serve['q_training_target_raw']-stay['q_training_target_raw']
    td=serve['td_raw']-stay['td_raw']
    return dict(bank_row=state_id,replicate=rep,arm=arm,delta_mc=mc,delta_td=td,error_td_minus_mc=td-mc,
        delta_raw=raw,delta_centered_analytic=centered,
        float32_centering_residual=serve['q_training_target_raw']-stay['q_training_target_raw']-raw,
        q_serve_raw=serve['q_raw'],q_pass_raw=stay['q_raw'],
        q_serve_training_target=serve['q_training_target_raw'],q_pass_training_target=stay['q_training_target_raw'],
        serve_first_reward=serve['first_reward'],pass_first_reward=stay['first_reward'],
        exogenous_sha256=serve['exogenous_sha256'])


def stats(values):
    a=np.asarray(values,dtype=float)
    se=float(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else None
    return dict(mean=float(a.mean()),se=se,ci95_low=None if se is None else float(a.mean()-1.96*se),
                ci95_high=None if se is None else float(a.mean()+1.96*se))


def summarize(rows, bootstrap=5000):
    summaries=[]; aggregates=[]
    arms=list(dict.fromkeys(r['arm'] for r in rows))
    for arm in arms:
        ids=sorted({r['bank_row'] for r in rows if r['arm']==arm})
        group=[]
        for idx in ids:
            rs=[r for r in rows if r['arm']==arm and r['bank_row']==idx]
            record=dict(arm=arm,bank_row=idx,replicates=len(rs))
            for name,key in (('mc','delta_mc'),('td','delta_td'),('error','error_td_minus_mc')):
                record.update({name+'_'+k:v for k,v in stats([r[key] for r in rs]).items()})
            # Float32 reverse cumsum can turn an exact zero action gap into +/-1e-4.
            tolerance=max(1e-3,max(abs(r['float32_centering_residual']) for r in rs))
            sign=lambda value: int(np.sign(value)) if abs(value)>tolerance else 0
            record.update(sign_tolerance_raw=tolerance,mc_sign=sign(record['mc_mean']),td_sign=sign(record['td_mean']))
            record['sign_agrees']=bool(record['mc_sign']==record['td_sign'])
            record['direction_resolved']=bool(record['mc_ci95_low'] is not None and
                (record['mc_ci95_low']>tolerance or record['mc_ci95_high']<-tolerance) and
                (record['td_ci95_low']>tolerance or record['td_ci95_high']<-tolerance))
            record['absolute_mean_error']=abs(record['error_mean'])
            summaries.append(record); group.append(rs)
        # CIs for fixed selected-state averages: resample paired replicates WITHIN state.
        # Do not treat the hand-selected states as population-independent replicates.
        rng=np.random.default_rng(20261108)
        boot_error=np.zeros(bootstrap); boot_abs=np.zeros(bootstrap); boot_mc=np.zeros(bootstrap)
        for rs in group:
            e=np.array([r['error_td_minus_mc'] for r in rs]); q=np.array([r['delta_mc'] for r in rs])
            inds=rng.integers(len(rs),size=(bootstrap,len(rs)))
            be=e[inds].mean(-1)
            boot_error+=be/len(ids); boot_abs+=np.abs(be)/len(ids); boot_mc+=q[inds].mean(-1)/len(ids)
        subset=[r for r in summaries if r['arm']==arm]
        record=dict(arm=arm,n_states=len(ids),mc_mean=float(np.mean([r['mc_mean'] for r in subset])),
            td_mean=float(np.mean([r['td_mean'] for r in subset])),
            mean_error=float(np.mean([r['error_mean'] for r in subset])),
            mean_absolute_error=float(np.mean([r['absolute_mean_error'] for r in subset])),
            sign_agreement_rate=float(np.mean([r['sign_agrees'] for r in subset])),
            max_abs_float32_centering_residual=float(max(abs(r['float32_centering_residual']) for rs in group for r in rs)))
        for name,values in (('mean_error',boot_error),('mean_absolute_error',boot_abs),('mc_mean',boot_mc)):
            low,high=np.quantile(values,[.025,.975]); record[name+'_ci95_low']=float(low); record[name+'_ci95_high']=float(high)
        resolved=[r for r in subset if r['direction_resolved']]
        record['n_direction_resolved']=len(resolved)
        record['sign_agreement_rate_resolved']=float(np.mean([r['sign_agrees'] for r in resolved])) if resolved else None
        aggregates.append(record)
    return summaries,aggregates


def run(args):
    torch.set_num_threads(1)
    prior=Path(args.snapshots_from); root=Path(args.run); out=Path(args.output)
    original=json.loads((prior/'manifest.json').read_text())
    if original.get('experiment_version')!='cache_rebound_v2' or original['status']!='complete':
        raise ValueError('Use a completed cache-rebound-v2 experiment-C directory')
    snapshot_path=prior/'trusted_local_snapshots.pt'
    if sha256(snapshot_path)!=original['outputs_sha256'][snapshot_path.name]: raise ValueError('Snapshot fingerprint mismatch')
    saved=torch.load(snapshot_path,map_location='cpu',weights_only=False)
    states={idx:clone_snapshot(saved['snapshots'][idx]) for idx in args.states}
    folder=root/'centered_seed42'; cp=folder/'iteration_000300.pt'
    if sha256(cp)!=original['checkpoint_sha256']['300']: raise ValueError('P300 checkpoint mismatch')
    probe=next(iter(states.values()))
    # from_checkpoint initializes/resets its environment: never hand it a probe state.
    trainer=AverageRewardPPO.from_checkpoint(clone_snapshot(probe),cp,for_evaluation=True)
    if trainer.cfg.objective!='centered': raise ValueError('Requires centered training objective')
    trainer.actor.eval(); trainer.critic.eval()
    fit=None
    if args.fit_checkpoint:
        fit=AverageRewardPPO.from_checkpoint(clone_snapshot(probe),args.fit_checkpoint,for_evaluation=True)
        if fit.iteration!=299: raise ValueError('Fitted-policy checkpoint must be iteration 299')
        fit.actor.eval()
    # The snapshot remains fixed, including all known requests/order and in-flight ETA.
    p150=AverageRewardPPO.from_checkpoint(clone_snapshot(probe),folder/'iteration_000150.pt',for_evaluation=True)
    actions={}
    for idx,s in states.items():
        verify_cache(s)
        if s.request_batches is None: raise ValueError('Expected historical validation snapshot')
        actions[idx]=greedy(p150.actor,s.observe(),s.action_mask())
        if s.decode(actions[idx])[0]!='serve' or greedy(trainer.actor,s.observe(),s.action_mask())!=s.pass_action:
            raise ValueError(f'State {idx} no longer has Serve150/Pass300 disagreement')
    training_rows=[json.loads(l) for l in (folder/'train.jsonl').read_text().splitlines()]
    train_row=next(r for r in training_rows if r['iteration']==300)
    gain=train_row['gain_estimate_per_period']; scale=trainer.cfg.reward_scale
    # Read source checkpoint metadata to require unchanged model parameters at end.
    digest_before=sha256(cp)
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(status='running',arguments=vars(args),experiment_version='native_future_v1',
        snapshot_sha256=sha256(snapshot_path),checkpoint300_sha256=digest_before,
        checkpoint150_sha256=sha256(folder/'iteration_000150.pt'),
        fitted_policy_checkpoint_sha256=sha256(args.fit_checkpoint) if fit else None,
        training_log_sha256=sha256(folder/'train.jsonl'),gain_raw=gain,reward_scale=scale,
        source_sha256={name:sha256(name) for name in ('experiment_d_critic_consistency.py','experiment_c_action_forks.py',
            'atomic_dual/environment.py','atomic_dual/ppo.py','atomic_dual/sampling.py')},
        arms=dict(D1='Sampled P300, float32 centered training-target formula',
            D2='Greedy P300, same centered target',D3='SAME greedy trajectories as D2, raw profit',
            D1_fit='Optional sampled Actor299, evaluated with Critic300'),
        pairing='Same future demand/order generator, event-keyed travel seed, policy RNG seed across action branches and modes; full fixed remaining horizon.',
        dynamics='Keep current requests/order/ETA. Future: native training Poisson arrivals and random fleet permutations; no future validation demand replay.',
        target='Sum (float32 reward / scale - logged gain300 / (T*N*scale)); terminal value zero; raw currency reported.',
        centering='Shared fixed-H baseline cancels analytically in action differences; report float32 cumulative rounding residual.',
        limitations=['Critic300 fitted on Actor299 trajectories, checkpoint300 contains updated Actor300. Actor299 unavailable unless supplied separately.',
            'Selected full simulator states come from validation trajectories; critic sees aggregate observations, not every latent detail.',
            'Gain is logged training-batch estimate, frozen across all branches; actual future gains may differ but baseline cancels in delta.',
            'CIs quantify Monte Carlo noise for the fixed three selected states, not state-population or training-seed uncertainty.'],
        interval='Per-state normal mean +/-1.96 paired SE; aggregate paired bootstrap within each fixed state.',
        sign_rule='Treat abs(mean)<=max(0.001 raw currency, observed float32 centering residual) as zero; also report agreement only among states with both mean CIs excluding numerical zero.')
    def save(): (out/'manifest.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    save(); rows=[]
    modes=[('D1','sampled',trainer.actor),('D2','greedy',trainer.actor)]
    if fit: modes.append(('D1_fit','sampled',fit.actor))
    for idx,snapshot in states.items():
        for rep in range(args.replicates):
            seed_values=np.random.SeedSequence([args.seed,idx,rep]).generate_state(3)
            seeds=dict(zip(('exogenous','travel','policy'),map(int,seed_values)))
            for arm,mode,actor in modes:
                serve=rollout_branch(snapshot,actions[idx],actor,trainer.critic,mode,seeds,gain,scale)
                stay=rollout_branch(snapshot,snapshot.pass_action,actor,trainer.critic,mode,seeds,gain,scale)
                batch=[paired_row(idx,rep,arm,serve,stay)]
                if arm=='D2': batch.append(paired_row(idx,rep,'D3',serve,stay))
                for row in batch:
                    row.update(seeds=seeds,t=snapshot.t,cursor=snapshot._cursor,
                        service_class=int(snapshot.current_car().service_class),serve_action=actions[idx])
                    rows.append(row)
                    with (out/'paired_rollouts.jsonl').open('a',encoding='utf-8') as f:
                        f.write(json.dumps(row,allow_nan=False)+'\n')
            if (rep+1)%args.progress_every==0 or rep+1==args.replicates:
                partial=[r for r in rows if r['bank_row']==idx]
                summaries,_=summarize(partial,bootstrap=200)
                print(json.dumps(dict(bank_row=idx,completed_replicates=rep+1,
                    estimates=[{k:r[k] for k in ('arm','mc_mean','td_mean','error_mean')} for r in summaries])),flush=True)
        state_summary,aggregate=summarize(rows,args.bootstrap_samples)
        write_csv(out/'state_summary.csv',state_summary)
        (out/'aggregate.json').write_text(json.dumps(aggregate,indent=2),encoding='utf-8')
    write_csv(out/'paired_rollouts.csv',[{**r,'seeds':json.dumps(r['seeds'],sort_keys=True)} for r in rows])
    if sha256(cp)!=digest_before: raise AssertionError('Source checkpoint changed')
    protocol.update(status='complete',no_network_updates=True,
        outputs_sha256={p.name:sha256(p) for p in out.iterdir() if p.name!='manifest.json'})
    save(); print(json.dumps(dict(output=str(out),aggregate=aggregate)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--snapshots-from',required=True,help='Trusted local completed fixed experiment-C directory.')
    parser.add_argument('--output',required=True)
    parser.add_argument('--states',type=int,nargs='+',default=[13126,11778,7372])
    parser.add_argument('--replicates',type=int,default=100)
    parser.add_argument('--seed',type=int,default=20261108)
    parser.add_argument('--fit-checkpoint',help='Optional Actor299 checkpoint; adds D1_fit without updating parameters.')
    parser.add_argument('--bootstrap-samples',type=int,default=5000)
    parser.add_argument('--progress-every',type=int,default=10)
    args=parser.parse_args()
    if min(args.replicates,args.progress_every,args.bootstrap_samples)<1 or args.seed<0 or len(args.states)!=len(set(args.states)):
        parser.error('Positive counts, nonnegative seed and unique state IDs required')
    run(args)
