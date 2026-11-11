"""Export pilot training curves, paired results and controlled rollout profiles."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from .data import load_environment,sha256
from .evaluation import evaluate_scenario,make_scenario
from .ppo import AverageRewardPPO,PPOConfig
from .stage4 import memory_usage


def load_completed_run(root,require_test=False):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('status')!='complete': raise ValueError('Experiment is not complete')
    if require_test and manifest.get('arguments',{}).get('validation_only') is not False:
        raise ValueError('A test report requires completed non-validation-only runs')
    return manifest


def profile(data,period,output):
    torch.set_num_threads(1)
    env=load_environment(Path(data)/f'{period}.npz',period)
    results=[];trajectories=[]
    for cached in (False,True):
        e=copy.deepcopy(env)
        if cached: e.enable_observation_cache()
        trainer=AverageRewardPPO(e,PPOConfig(hidden_dim=64,actor_epochs=1,critic_epochs=1),seed=7)
        torch.manual_seed(20261106)
        start=time.perf_counter();tr=trainer.rollout(7);elapsed=time.perf_counter()-start
        results.append(dict(cache=cached,seconds=elapsed,samples=len(tr['actions']),
            trajectory_array_bytes=sum(v.nbytes for v in tr.values() if isinstance(v,np.ndarray)),**memory_usage()))
        trajectories.append(tr)
    for name in ('states','next_states','masks','actions','logps','rewards','dones'):
        if not np.array_equal(trajectories[0][name],trajectories[1][name]): raise AssertionError(f'Cache changed {name}')
    result=dict(period=period,seed=7,policy_seed=20261106,hidden_dim=64,threads=1,profiles=results,
        exact_arrays_equal=True,observed_speed_ratio=results[0]['seconds']/results[1]['seconds'],
        note='Single sequential measurements; wallclock can be affected by concurrent experiments; forced-forward skip enabled in both')
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


def _lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def report(roots,output,sensitivity=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    summary=[];diagnostics=[];training=[]
    for root in map(Path,roots):
        manifest=load_completed_run(root,require_test=True)
        period=manifest['arguments']['period']
        seeds=json.loads((root/'training_seed_summary.json').read_text())
        for row in seeds: summary.append(dict(period=period,**row))
        fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
        styles={seed:['-','--',':','-.'][i%4] for i,seed in enumerate(manifest['arguments']['seeds'])}
        for selection in json.loads((root/'selection_frozen.json').read_text()):
            name=selection['name'];folder=root/name
            train=_lines(folder/'train.jsonl');val=_lines(folder/'validation.jsonl')
            training.extend(dict(period=period,run=name,**r) for r in train)
            color='#0072B2' if selection['objective']=='centered' else '#D55E00'
            style=styles[selection['training_seed']]
            x=[r['iteration'] for r in train]
            for axis,key in [(axes[0,0],'gain_estimate_per_period'),(axes[0,2],'actor_entropy'),
                (axes[1,0],'approx_kl'),(axes[1,1],'critic_mse_after'),(axes[1,2],'forced_action_fraction')]:
                axis.plot(x,[r[key] for r in train],color=color,alpha=.65,label=name,marker='.',linewidth=1,linestyle=style)
            axes[0,1].plot([r['iteration'] for r in val],[r['validation_profit_mean'] for r in val],
                color=color,alpha=.65,label=name,marker='.',linewidth=1,linestyle=style)
            diagnostics.append(dict(period=period,run=name,selection=selection,**json.loads((folder/'diagnostics.json').read_text())))
        titles=['Training profit (sampled policy)','October validation (greedy policy)','Actor entropy (choice states)',
                'Approximate KL','Critic MC MSE (own objective)','Forced-action fraction']
        for axis,title in zip(axes.flat,titles):
            axis.set_title(title,fontsize=10);axis.set_xlabel('Iteration');axis.grid(alpha=.2)
        axes[1,1].set_yscale('symlog',linthresh=.01)
        axes[0,0].set_ylabel('ADP model currency');axes[0,1].set_ylabel('ADP model currency')
        axes[0,1].legend(fontsize=6,loc='best')
        fig.suptitle(f'{period.title()}: {manifest["arguments"]["iterations"]}-iteration pilot; each line is a training seed')
        fig.savefig(out/f'{period}_curves.png',dpi=160);fig.savefig(out/f'{period}_curves.svg');plt.close(fig)
    sensitivity_rows=[]
    if sensitivity:
        load_completed_run(sensitivity)
        sensitivity_rows=json.loads((Path(sensitivity)/'summary.json').read_text())
    payload=dict(training_seed_summary=summary,diagnostics=diagnostics,train_rows=training,
        sensitivity=sensitivity_rows,source_sha256=sha256(__file__),
        conclusion='Pilot evidence only. Inspect all seeds against baselines; no convergence guarantee.')
    (out/'results.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# Stage 4 pilot results','',
        'Each objective was trained independently with all reported seeds. Model selection uses October validation only.',
        'Test statistics average each seed over the same November dates; SD below is across training-seed means, not across seed-date pairs.',
        '', '| Period | Objective | Training seeds | Mean test profit | SD across seed means | Mean paired delta vs Greedy |',
        '|---|---|---:|---:|---:|---:|']
    for row in summary:
        sd=f"{row['std_across_seed_means']:.3f}" if row['std_across_seed_means'] is not None else 'NA'
        lines.append(f"| {row['period']} | {row['objective']} | {row['n_training_seeds']} | {row['mean']:.3f} | {sd} | {row['mean_paired_difference_vs_greedy']:.3f} |")
    if sensitivity_rows:
        lines+=['','## Validation-only sensitivity pilot','One training seed, two iterations, one episode per iteration. These are not confirmatory ablations.',
            '', '| Case | Selected iteration | Validation profit |', '|---|---:|---:|']
        for row in sensitivity_rows:
            lines.append(f"| {row['case']} | {row['iteration']} | {row['validation_profit_mean']:.3f} |")
    lines+=['','## Limits',
        '- Six iterations are a pilot and do not establish convergence.',
        '- Ordinary episodic comparator here keeps gamma=1 and TD(0); only gain centering is removed. It is not a separate GAE implementation.',
        '- Keep all seeds, including any selection of the untrained iteration zero.',
        '- Travel/financial/model differences from upstream ADP remain; no direct ADP superiority claim.',
        '- Counterfactual diagnostics compare one-step critic ranking to same-batch heuristic continuation, not true long-horizon optimal actions.']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('profile');a.add_argument('--data',required=True);a.add_argument('--period',required=True);a.add_argument('--output',required=True)
    b=sub.add_parser('report');b.add_argument('--runs',nargs='+',required=True);b.add_argument('--output',required=True);b.add_argument('--sensitivity')
    args=p.parse_args()
    if args.command=='profile':profile(args.data,args.period,args.output)
    else:report(args.runs,args.output,args.sensitivity)


if __name__=='__main__':main()
