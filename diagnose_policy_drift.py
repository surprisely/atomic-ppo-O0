"""Read-only checkpoint diagnostics on frozen validation scenarios (never test)."""
from __future__ import annotations
import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from atomic_dual.data import load_environment, sha256
from atomic_dual.evaluation import load_scenario, evaluate_scenario
from atomic_dual.ppo import AverageRewardPPO


def distributions(actor, states, masks):
    with torch.inference_mode():
        logits = actor(torch.from_numpy(states)).double()
        logp = logits.masked_fill(~torch.from_numpy(masks), -torch.inf).log_softmax(-1)
        # Set infeasible entries to zero only AFTER softmax, to avoid 0*(-inf).
        p = logp.exp()
        return p.numpy(), logp.masked_fill(~torch.from_numpy(masks), 0.).numpy()


def compare(p, logp, ref, reflog):
    return dict(entropy=float(-(p*logp).sum(-1).mean()),
                kl_150_to_k=float((ref*(reflog-logp)).sum(-1).mean()),
                argmax_changed=float((p.argmax(-1) != ref.argmax(-1)).mean()))


def write_csv(path, rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)


def run(args):
    torch.set_num_threads(1)
    root, out = Path(args.run), Path(args.output)
    manifest=json.loads((root/'manifest.json').read_text())
    runfolder=root/'centered_seed42'
    iterations=[150,170,190,220,260,300]
    checkpoints={i:runfolder/f'iteration_{i:06d}.pt' for i in iterations}
    for cp in checkpoints.values():
        if not cp.exists(): raise FileNotFoundError(cp)
    scenarios=[load_scenario(p) for p in sorted((root/'validation_scenarios').glob('*.npz'))]
    assert len(scenarios)==5 and all(s['data_split']=='validation' for s in scenarios)
    assert [s['scenario_id'] for s in scenarios]==manifest['validation_scenario_ids']
    calibration=Path(manifest['arguments']['data'])/'evening.npz'
    assert sha256(calibration)==manifest['calibration_sha256']
    env=load_environment(calibration,'evening',manifest['environment']['step_minutes'])
    env.cfg=replace(env.cfg,**manifest['environment'])
    env.enable_observation_cache()
    actors={i:AverageRewardPPO.from_checkpoint(env,cp,for_evaluation=True).actor.eval()
            for i,cp in checkpoints.items()}
    out.mkdir(parents=True,exist_ok=False)
    state_rows=[]; mask_rows=[]; metadata=[]; episodes=[]
    source_iterations=[150,220,300]
    original=[json.loads(l) for l in (runfolder/'validation_episodes.jsonl').read_text().splitlines()]
    for iteration in source_iterations:
        for date_idx,scenario in enumerate(scenarios):
            rng=np.random.default_rng(20261107+iteration*10+date_idx)
            reservoir=[]; seen=0
            counts=dict(pass_count=0,serve_count=0,relocate_count=0,active=0,
                        idle_elapsed_minutes=0.)
            destinations=np.zeros(env.Z,dtype=int)
            step=env.step
            def instrument(action):
                nonlocal seen
                mask=env.action_mask()
                if mask.sum()>1:
                    seen+=1
                    pos=seen-1 if seen<=args.samples_per_scenario else int(rng.integers(seen))
                    if pos<args.samples_per_scenario:
                        item=(env.observe().copy(),mask.copy(),
                              [iteration,date_idx,env.t,int(env.current_car().service_class),
                               env.current_car().region,int(mask[:3*env.Z].any())])
                        if pos==len(reservoir): reservoir.append(item)
                        else: reservoir[pos]=item
                    kind,_,dest=env.decode(action)
                    counts['active']+=1
                    counts[kind+'_count']+=1
                    if kind=='relocate': destinations[dest]+=1
                    if kind=='pass' and env.T-env.t-(env.cfg.batch_timing=='closing')>0:
                        counts['idle_elapsed_minutes']+=env.cfg.step_minutes
                return step(action)
            env.step=instrument
            try:
                row=evaluate_scenario(env,scenario,'ppo_greedy',actors[iteration],
                                      checkpoint_hash=sha256(checkpoints[iteration]))
            finally:
                env.step=step
            previous=next(r for r in original if r['iteration']==iteration and r['scenario_id']==scenario['scenario_id'])
            # Replay equivalence checks both financial outcome and behavioral totals.
            for key in ('profit_total','requests_served','relocation_count','relocation_elapsed_minutes'):
                if not np.isclose(row[key],previous[key],rtol=0,atol=1e-7):
                    raise AssertionError((iteration,scenario['date'],key,row[key],previous[key]))
            row.update(iteration=iteration,**counts,
                       stay_fraction=counts['pass_count']/counts['active'],
                       no_relocation_fraction=1-counts['relocate_count']/counts['active'],
                       relocation_fraction=counts['relocate_count']/counts['active'])
            for dest,count in enumerate(destinations): row[f'relocate_destination_{dest}_count']=int(count)
            occupied=(row['express_utilization']*env.cfg.n_express+
                      row['premium_utilization']*env.cfg.n_premium)*env.T*env.cfg.step_minutes
            capacity=env.N*(env.T-(env.cfg.batch_timing=='closing'))*env.cfg.step_minutes
            assert np.isclose(occupied+row['relocation_elapsed_minutes']+counts['idle_elapsed_minutes'],capacity)
            row['occupied_elapsed_minutes']=occupied
            row['empty_driving_elapsed_minutes']=row['relocation_elapsed_minutes']
            episodes.append(row)
            for state,mask,meta in reservoir:
                state_rows.append(state); mask_rows.append(mask); metadata.append(meta)
            print(json.dumps(dict(iteration=iteration,date=scenario['date'],profit=row['profit_total'],
                                  replay_matches=True,sampled_states=len(reservoir))),flush=True)
    states=np.stack(state_rows); masks=np.stack(mask_rows); meta=np.array(metadata)
    np.savez_compressed(out/'fixed_states.npz',states=states,masks=masks,metadata=meta)
    ref,reflog=distributions(actors[150],states,masks)
    summaries=[]; destinations=[]; full_probs={}
    for iteration,actor in actors.items():
        p,logp=distributions(actor,states,masks)
        full_probs[str(iteration)]=p
        groups={'pooled':np.ones(len(meta),dtype=bool)}
        groups.update({f'source_{s}':meta[:,0]==s for s in source_iterations})
        groups.update({f'class_{c}':meta[:,3]==c for c in (0,1)})
        groups.update({'no_service_available':meta[:,5]==0,'service_available':meta[:,5]==1})
        for group,select in groups.items():
            x=p[select]; n=len(x)
            if not n: continue
            a=x.argmax(-1)
            row=dict(iteration=iteration,group=group,n=n,**compare(x,logp[select],ref[select],reflog[select]),
                     pass_probability=float(x[:,-1].mean()),
                     serve_probability=float(x[:,:3*env.Z].sum(-1).mean()),
                     relocation_probability=float(x[:,3*env.Z:4*env.Z].sum(-1).mean()),
                     argmax_pass_fraction=float((a==4*env.Z).mean()),
                     argmax_relocation_fraction=float(((a>=3*env.Z)&(a<4*env.Z)).mean()))
            row['no_relocation_probability']=1-row['relocation_probability']
            summaries.append(row)
            for dest in range(env.Z):
                destinations.append(dict(iteration=iteration,group=group,destination=dest,
                    probability=float(x[:,3*env.Z+dest].mean()),
                    argmax_fraction=float((a==3*env.Z+dest).mean())))
    np.savez_compressed(out/'action_probabilities.npz',**full_probs)
    write_csv(out/'fixed_state_summary.csv',summaries)
    write_csv(out/'fixed_state_destinations.csv',destinations)
    write_csv(out/'trajectory_episodes.csv',episodes)
    means=[]
    keys=['profit_total','requests_served','relocation_count','relocation_minutes',
          'relocation_elapsed_minutes','idle_elapsed_minutes','occupied_elapsed_minutes',
          'stay_fraction','no_relocation_fraction','relocation_fraction']
    for i in source_iterations:
        subset=[r for r in episodes if r['iteration']==i]
        means.append(dict(iteration=i,**{k:float(np.mean([r[k] for r in subset])) for k in keys}))
    write_csv(out/'trajectory_means.csv',means)
    result=dict(trajectory_means=means,fixed_state_summary=summaries,
                source_sha256={name:sha256(Path(name)) for name in
                    ('diagnose_policy_drift.py','atomic_dual/environment.py',
                     'atomic_dual/evaluation.py','atomic_dual/ppo.py')},
                checkpoints={str(i):dict(path=str(cp.resolve()),sha256=sha256(cp)) for i,cp in checkpoints.items()},
                scenario_ids=[s['scenario_id'] for s in scenarios],
                sampling=dict(seed_base=20261107,per_source_date=args.samples_per_scenario,
                              sources=source_iterations,metadata_columns=['source_iteration','date_index','t','class','origin','service_available']),
                definitions=dict(entropy='natural logarithm, feasible actions, active states only',
                    kl='mean KL(pi150 || pik) on identical observations AND masks; natural logarithm',
                    empty_driving='relocation only; same-zone pickups have no empty pickup travel',
                    idle='idle Pass intervals inside horizon, excludes forced busy Pass and final closing instant',
                    relocation_minutes='full sampled task durations, may extend beyond horizon',
                    relocation_elapsed_minutes='rounded task steps clipped to horizon, vehicle-minutes',
                    state_weighting='equal reservoir size per source policy and validation date, excludes forced Pass'))
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(out),trajectory_means=means)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--samples-per-scenario',type=int,default=1000)
    run(parser.parse_args())
