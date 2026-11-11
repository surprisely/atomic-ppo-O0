"""E: successor-value bias decomposition. F: TD(0)/GAE/MC on paired P300 traces.

Actual training uses centered TD(0), NOT GAE. Gamma=1 per atomic decision,
zero terminal value, full remaining fixed horizon. GAE is diagnostic only.
"""
from __future__ import annotations
import argparse
import json
import zipfile
from pathlib import Path
import numpy as np
import torch
from atomic_dual.data import sha256
from atomic_dual.ppo import AverageRewardPPO
from experiment_c_action_forks import clone_snapshot, greedy
from experiment_d_critic_consistency import rollout_branch, stats, summarize, verify_cache
from diagnose_policy_drift import write_csv

METHODS={'TD0_actual':0.,'GAE_0.5':.5,'GAE_0.95':.95,'GAE_1':1.}


def analyze_trace(trace, scale):
    """Use float64 accumulation of the SAME float32 adjusted reward terms as training."""
    rewards=np.asarray(trace['adjusted_scaled'],dtype=np.float64)*scale
    values=np.asarray(trace['values_scaled'],dtype=np.float64)*scale
    if len(values)!=len(rewards)+1 or values[-1]!=0.:
        raise ValueError('Requires full terminal trace with zero terminal value')
    deltas=rewards+values[1:]-values[:-1]
    mc=float(rewards.sum()-values[0])
    advantages={name:float(np.dot(np.power(lam,np.arange(len(deltas))),deltas)) for name,lam in METHODS.items()}
    advantages['MC']=mc
    if not np.isclose(advantages['GAE_1'],mc,atol=1e-7,rtol=1e-10):
        raise AssertionError('GAE(1) must telescope to complete MC advantage')
    successor_mc=float(rewards[1:].sum())
    return dict(reward_raw=float(trace['rewards_raw'][0]),reward_adjusted=float(rewards[0]),
        value_current=float(values[0]),value_next=float(values[1]),mc_next=successor_mc,
        successor_bias=float(values[1]-successor_mc),advantages=advantages,
        mc_total=float(rewards.sum()))


def analyze_pair(idx,rep,serve,stay,scale):
    if serve['exogenous_sha256']!=stay['exogenous_sha256']: raise AssertionError('Pairing failed')
    s=analyze_trace(serve['trace'],scale); p=analyze_trace(stay['trace'],scale)
    if s['value_current']!=p['value_current']: raise AssertionError('Initial critic baseline differs')
    e=dict(bank_row=idx,replicate=rep)
    for name,values in (('serve',s),('pass',p)):
        e.update({name+'_'+k:v for k,v in values.items() if k!='advantages'})
    e.update(immediate_difference=s['reward_adjusted']-p['reward_adjusted'],
        future_value_difference=s['value_next']-p['value_next'],
        future_mc_difference=s['mc_next']-p['mc_next'],
        b_serve=s['successor_bias'],b_pass=p['successor_bias'],
        delta_td=s['advantages']['TD0_actual']-p['advantages']['TD0_actual'],
        delta_mc=s['mc_total']-p['mc_total'])
    e['error_td_minus_mc']=e['delta_td']-e['delta_mc']
    e['bias_difference']=e['b_serve']-e['b_pass']
    e['identity_residual']=e['error_td_minus_mc']-e['bias_difference']
    if abs(e['identity_residual'])>1e-7: raise AssertionError('E decomposition identity failed')
    # Separate absolute errors from relative ones: both may be under/over-valued.
    f=[]
    for method in (*METHODS,'MC'):
        difference=s['advantages'][method]-p['advantages'][method]
        f.append(dict(bank_row=idx,replicate=rep,arm=method,delta_mc=e['delta_mc'],
            delta_td=difference,error_td_minus_mc=difference-e['delta_mc'],float32_centering_residual=0.))
    return e,f


E_FIELDS=('serve_reward_raw','pass_reward_raw','serve_reward_adjusted','pass_reward_adjusted',
    'serve_value_next','pass_value_next','serve_mc_next','pass_mc_next',
    'b_serve','b_pass','immediate_difference','future_value_difference','future_mc_difference',
    'delta_td','delta_mc','error_td_minus_mc','bias_difference')


def summarize_e(rows,bootstrap):
    ids=sorted({r['bank_row'] for r in rows})
    state_rows=[]; aggregate=dict(n_states=len(ids))
    rng=np.random.default_rng(20261108)
    boot={key:np.zeros(bootstrap) for key in E_FIELDS}
    for idx in ids:
        group=[r for r in rows if r['bank_row']==idx]
        record=dict(bank_row=idx,replicates=len(group))
        inds=rng.integers(len(group),size=(bootstrap,len(group)))
        for key in E_FIELDS:
            values=np.array([r[key] for r in group])
            record.update({key+'_'+k:v for k,v in stats(values).items()})
            boot[key]+=values[inds].mean(-1)/len(ids)
        state_rows.append(record)
    for key in E_FIELDS:
        aggregate[key+'_mean']=float(np.mean([r[key+'_mean'] for r in state_rows]))
        low,high=np.quantile(boot[key],[.025,.975])
        aggregate[key+'_ci95_low']=float(low); aggregate[key+'_ci95_high']=float(high)
    aggregate['max_identity_residual']=float(max(abs(r['identity_residual']) for r in rows))
    return state_rows,aggregate


def summarize_f(rows,bootstrap):
    states,aggregate=summarize(rows,bootstrap)
    def renamed(row):
        return {('method' if k=='arm' else 'estimate_'+k[3:] if k.startswith('td_') else k):v for k,v in row.items()}
    return list(map(renamed,states)),list(map(renamed,aggregate))


def read_complete_jsonl(path):
    """Preserve source files; ignore only an incomplete final JSON record."""
    lines=path.read_text(encoding='utf-8').splitlines()
    rows=[]
    for i,line in enumerate(lines):
        if not line.strip(): continue
        try: rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i!=len(lines)-1: raise
    return rows


def recover_pairs(prior, ids, replicates, seed):
    """Recover committed E/F/trace triples; an incomplete pair is replayed once."""
    old=json.loads((prior/'manifest.json').read_text())
    if not old['arguments']['save_traces']: raise ValueError('Resume requires saved traces')
    e_by={}; f_by={}
    for row in read_complete_jsonl(prior/'E_pairs.jsonl'):
        key=(row['bank_row'],row['replicate'])
        if key in e_by: raise ValueError(f'Duplicate E pair: {key}')
        e_by[key]=row
    f_path=prior/'F_pairs.jsonl'
    for row in read_complete_jsonl(f_path) if f_path.exists() else []:
        key=(row['bank_row'],row['replicate']); method=row['arm']
        if method in f_by.setdefault(key,{}): raise ValueError(f'Duplicate F pair: {key}/{method}')
        f_by[key][method]=row
    es=[]; fs=[]; traces={}
    for key,e in e_by.items():
        idx,rep=key
        if idx not in ids or not 0<=rep<replicates: raise ValueError(f'Unexpected prior pair: {key}')
        expected_seeds=dict(zip(('exogenous','travel','policy'),map(int,np.random.SeedSequence([seed,idx,rep]).generate_state(3))))
        if e['seeds']!=expected_seeds: raise ValueError(f'Prior seed mismatch: {key}')
        methods=f_by.get(key,{})
        if set(methods)!=set((*METHODS,'MC')): continue
        token=f'{idx}:{rep}'
        trace=Path(old.get('reused_trace_paths',{}).get(token,prior/'traces'/f'state_{idx}_rep_{rep:04d}.npz'))
        try:
            with np.load(trace,allow_pickle=False) as data:
                if not {'serve_values_scaled','pass_values_scaled','serve_adjusted_scaled','pass_adjusted_scaled'}<=set(data.files): continue
        except (OSError,ValueError,EOFError,zipfile.BadZipFile): continue
        if abs(e['identity_residual'])>1e-7 or not np.isclose(e['error_td_minus_mc'],e['b_serve']-e['b_pass'],atol=1e-7,rtol=0):
            raise ValueError(f'Prior E identity mismatch: {key}')
        for name in ('GAE_1','MC'):
            if not np.isclose(methods[name]['delta_td'],e['delta_mc'],atol=1e-7,rtol=0):
                raise ValueError(f'Prior GAE/MC mismatch: {key}')
        if any(not np.isclose(r['delta_mc'],e['delta_mc'],atol=1e-7,rtol=0) for r in methods.values()):
            raise ValueError(f'Prior MC reference mismatch: {key}')
        es.append(e); fs.extend(methods[name] for name in (*METHODS,'MC'))
        traces[token]=str(trace.resolve())
    return es,fs,traces


def run(args):
    torch.set_num_threads(1)
    prior=Path(args.snapshots_from); root=Path(args.run); out=Path(args.output)
    c_manifest=json.loads((prior/'manifest.json').read_text())
    if c_manifest.get('experiment_version')!='cache_rebound_v2' or c_manifest['status']!='complete':
        raise ValueError('Requires completed fixed experiment-C snapshots')
    snapshot_path=prior/'trusted_local_snapshots.pt'
    if sha256(snapshot_path)!=c_manifest['outputs_sha256'][snapshot_path.name]: raise ValueError('Snapshot hash mismatch')
    saved=torch.load(snapshot_path,map_location='cpu',weights_only=False)
    ids=(list(args.states)+sorted(set(saved['snapshots'])-set(args.states))) if args.all_states else args.states
    snapshots={idx:clone_snapshot(saved['snapshots'][idx]) for idx in ids}
    folder=root/'centered_seed42'; cp300=folder/'iteration_000300.pt'; cp150=folder/'iteration_000150.pt'
    for i,cp in ((150,cp150),(300,cp300)):
        if sha256(cp)!=c_manifest['checkpoint_sha256'][str(i)]: raise ValueError('Checkpoint hash mismatch')
    initial=next(iter(snapshots.values()))
    tr=AverageRewardPPO.from_checkpoint(clone_snapshot(initial),cp300,for_evaluation=True)
    ref=AverageRewardPPO.from_checkpoint(clone_snapshot(initial),cp150,for_evaluation=True)
    if tr.cfg.objective!='centered': raise ValueError('Expected centered target')
    tr.actor.eval(); tr.critic.eval(); ref.actor.eval()
    training_rows=[json.loads(l) for l in (folder/'train.jsonl').read_text().splitlines()]
    gain=next(r for r in training_rows if r['iteration']==300)['gain_estimate_per_period']
    scale=tr.cfg.reward_scale
    actions={}
    for idx,s in snapshots.items():
        verify_cache(s)
        action=greedy(ref.actor,s.observe(),s.action_mask())
        if s.decode(action)[0]!='serve' or greedy(tr.actor,s.observe(),s.action_mask())!=s.pass_action:
            raise AssertionError(f'Not Serve150/Pass300: {idx}')
        actions[idx]=action
    check={}; d_manifest=None
    if args.check_d_output:
        droot=Path(args.check_d_output); d_manifest=json.loads((droot/'manifest.json').read_text())
        if (d_manifest['status']!='complete' or d_manifest['checkpoint300_sha256']!=sha256(cp300) or
            d_manifest['snapshot_sha256']!=sha256(snapshot_path) or d_manifest['gain_raw']!=gain or
            d_manifest['arguments']['seed']!=args.seed): raise ValueError('D provenance does not match')
        raw_path=droot/'paired_rollouts.jsonl'
        if sha256(raw_path)!=d_manifest['outputs_sha256'][raw_path.name]: raise ValueError('D output hash mismatch')
        check={(r['bank_row'],r['replicate']):r for r in
            map(json.loads,raw_path.read_text().splitlines()) if r['arm']=='D1'}
    e_rows=[]; f_rows=[]; reused_traces={}; resume_manifest=None
    if args.resume_from:
        resume_root=Path(args.resume_from)
        resume_manifest=json.loads((resume_root/'manifest.json').read_text())
        if resume_manifest.get('experiment_version')!='EF_sampled_trace_v1': raise ValueError('Resume version mismatch')
        expected=dict(checkpoint300_sha256=sha256(cp300),snapshot_sha256=sha256(snapshot_path),gain_raw=gain,reward_scale=scale,selected_bank_rows=ids)
        for key,value in expected.items():
            if resume_manifest[key]!=value: raise ValueError(f'Resume configuration mismatch: {key}')
        for key in ('replicates','seed','save_traces'):
            if resume_manifest['arguments'][key]!=getattr(args,key): raise ValueError(f'Resume argument mismatch: {key}')
        for name in ('atomic_dual/ppo.py','atomic_dual/environment.py','experiment_d_critic_consistency.py','experiment_c_action_forks.py'):
            if resume_manifest['source_sha256'][name]!=sha256(name): raise ValueError(f'Resume dynamics source changed: {name}')
        e_rows,f_rows,reused_traces=recover_pairs(resume_root,ids,args.replicates,args.seed)
    done={(r['bank_row'],r['replicate']) for r in e_rows}
    checked=sum(key in check for key in done)
    print(json.dumps(dict(recovered_pairs=len(done),remaining_pairs=len(ids)*args.replicates-len(done),
        remaining_fork_rollouts=2*(len(ids)*args.replicates-len(done)))),flush=True)
    if args.resume_check_only: return
    out.mkdir(parents=True,exist_ok=False)
    if args.save_traces: (out/'traces').mkdir()
    protocol=dict(status='running',arguments=vars(args),experiment_version='EF_sampled_trace_v1',
        selected_bank_rows=ids,checkpoint300_sha256=sha256(cp300),snapshot_sha256=sha256(snapshot_path),
        gain_raw=gain,reward_scale=scale,actual_training_advantage='Centered TD(0) then normalization over active batch decisions; NO GAE.',
        methods=METHODS,lambda_unit='One atomic vehicle decision, not one physical minute.',
        bootstrap_value='Zero at true terminal; complete remaining atomic horizon, gamma=1.',
        rewards='Float32 adjusted reward terms exactly as training, accumulated in float64 for identities; original float32 cumsum rounding separately recorded.',
        dynamics='Random P300, native training future Poisson demand/order; paired seeds match D1.',
        E_identity='TD action gap - MC action gap = bServe - bPass, gamma=1.',
        F_identity='GAE(1) = complete MC advantage; TD0_actual is the current implementation, 0.5/0.95 are hypothetical alternatives.',
        interpretation='E absolute biases use frozen logged training gain300; relative bias invariant to shared centering baseline. F measures advantage direction; no actor update/gradient causation claimed.',
        limitations='Actor300 is after update; Critic300 fitted Actor299, unavailable. Full-state probes and aggregated critic have different conditioning.',
        source_sha256={name:sha256(name) for name in ('experiment_ef_value_advantages.py',
            'experiment_d_critic_consistency.py','experiment_c_action_forks.py','atomic_dual/ppo.py','atomic_dual/environment.py')})
    if resume_manifest is not None:
        protocol.update(resumed_from=str(Path(args.resume_from).resolve()),recovered_pairs=len(done),
            resume_manifest_sha256=sha256(Path(args.resume_from)/'manifest.json'),
            reused_trace_paths=reused_traces,
            reused_trace_sha256={key:sha256(path) for key,path in reused_traces.items()})
    def save(): (out/'manifest.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    save()
    # Write recovered records only to the NEW output directory, never modify the source run.
    if e_rows:
        with (out/'E_pairs.jsonl').open('w',encoding='utf-8') as f:
            for row in e_rows: f.write(json.dumps(row,allow_nan=False)+'\n')
        with (out/'F_pairs.jsonl').open('w',encoding='utf-8') as f:
            for row in f_rows: f.write(json.dumps(row,allow_nan=False)+'\n')
    def reports():
        es,ea=summarize_e(e_rows,args.bootstrap_samples)
        fs,fa=summarize_f(f_rows,args.bootstrap_samples)
        write_csv(out/'E_state_summary.csv',es); write_csv(out/'F_state_summary.csv',fs)
        (out/'aggregate.json').write_text(json.dumps(dict(E=ea,F=fa),indent=2),encoding='utf-8')
        return ea,fa
    for idx,snapshot in snapshots.items():
        for rep in range(args.replicates):
            if (idx,rep) in done: continue
            seeds=dict(zip(('exogenous','travel','policy'),map(int,np.random.SeedSequence([args.seed,idx,rep]).generate_state(3))))
            serve=rollout_branch(snapshot,actions[idx],tr.actor,tr.critic,'sampled',seeds,gain,scale,record_trace=True)
            stay=rollout_branch(snapshot,snapshot.pass_action,tr.actor,tr.critic,'sampled',seeds,gain,scale,record_trace=True)
            if (idx,rep) in check:
                old=check[idx,rep]
                comparisons=((serve['q_raw']-stay['q_raw'],old['delta_raw'],1e-7),
                    (serve['td_raw']-stay['td_raw'],old['delta_td'],1e-7),
                    (serve['q_training_target_raw']-stay['q_training_target_raw'],old['delta_mc'],1e-7))
                if any(not np.isclose(a,b,atol=tol,rtol=0) for a,b,tol in comparisons):
                    raise AssertionError(f'D1 replay mismatch: {idx}/{rep}')
                checked+=1
            e,fs=analyze_pair(idx,rep,serve,stay,scale)
            e.update(seeds=seeds,t=snapshot.t,cursor=snapshot._cursor,
                training_float32_delta=serve['q_training_target_raw']-stay['q_training_target_raw'])
            e['training_cumsum_rounding_residual']=e['training_float32_delta']-e['delta_mc']
            e_rows.append(e); f_rows.extend(fs)
            with (out/'E_pairs.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(e,allow_nan=False)+'\n')
            with (out/'F_pairs.jsonl').open('a',encoding='utf-8') as f:
                for row in fs: f.write(json.dumps(row,allow_nan=False)+'\n')
            if args.save_traces:
                arrays={name+'_'+key:value for name,result in (('serve',serve),('pass',stay)) for key,value in result['trace'].items()}
                np.savez_compressed(out/'traces'/f'state_{idx}_rep_{rep:04d}.npz',**arrays)
            if (rep+1)%args.progress_every==0 or rep+1==args.replicates:
                print(json.dumps(dict(bank_row=idx,completed_replicates=rep+1,D1_replay_checks=checked)),flush=True)
        ea,fa=reports()
    write_csv(out/'E_pairs.csv',[{**r,'seeds':json.dumps(r['seeds'],sort_keys=True)} for r in e_rows])
    write_csv(out/'F_pairs.csv',f_rows)
    protocol.update(status='complete',D1_matching_pairs_checked=checked,all_E_identities_passed=True,
        all_GAE1_MC_checks_passed=True,no_network_updates=True,
        outputs_sha256={p.relative_to(out).as_posix():sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'})
    save(); print(json.dumps(dict(output=str(out),E=ea,F=fa)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True); p.add_argument('--snapshots-from',required=True)
    p.add_argument('--output',required=True); p.add_argument('--check-d-output')
    p.add_argument('--resume-from',help='Recover complete pairs from prior EF output; write remaining work and full summaries into a NEW --output directory.')
    p.add_argument('--resume-check-only',action='store_true',help='Validate recovery and print counts without creating output or simulating.')
    p.add_argument('--states',nargs='+',type=int,default=[11778,13126,7372])
    p.add_argument('--all-states',action='store_true',help='Use all 15 C snapshots instead of --states.')
    p.add_argument('--replicates',type=int,default=100); p.add_argument('--seed',type=int,default=20261108)
    p.add_argument('--save-traces',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--bootstrap-samples',type=int,default=5000); p.add_argument('--progress-every',type=int,default=10)
    args=p.parse_args()
    if min(args.replicates,args.bootstrap_samples,args.progress_every)<1 or args.seed<0 or len(args.states)!=len(set(args.states)):
        p.error('Positive counts, nonnegative seed, unique states required')
    run(args)
