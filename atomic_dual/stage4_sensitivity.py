"""Predeclared small validation-only sensitivity matrix; never tune on November."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from .stage4 import run
from .data import sha256


def cases(main,half,double):
    return dict(base=dict(data=main),batch128=dict(data=main,minibatch_size=128),
        actor_lr1e3=dict(data=main,actor_lr=.001),eta16=dict(data=main,eta_buckets=16),
        fixed_order=dict(data=main,fixed_order=True),step05=dict(data=half),step2=dict(data=double))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--data-half',required=True);p.add_argument('--data-double',required=True)
    p.add_argument('--output',required=True);p.add_argument('--period',choices=['morning','evening'],default='morning')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    plan=cases(a.data,a.data_half,a.data_double)
    manifest=dict(status='running',period=a.period,seed=7,iterations=2,episodes=1,
        evaluation_split='validation',source_sha256=sha256(__file__),cases=plan)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    summaries=[]
    for name,changes in plan.items():
        options=dict(data=a.data,period=a.period,seeds=[7],objectives=['centered'],iterations=2,episodes=1,eval_every=2,
            validation_dates=['2025-10-27','2025-10-30'],test_dates=None,validation_only=True,device='cpu',hidden_dim=64,
            actor_lr=.0003,critic_lr=.001,minibatch_size=512,eta_buckets=8,actor_epochs=3,critic_epochs=4,reward_scale=100.,fixed_order=False,
            output=str(out/name))
        options.update(changes)
        run(SimpleNamespace(**options))
        selection=json.loads((out/name/'selection_frozen.json').read_text())[0]
        summaries.append(dict(case=name,**selection))
        (out/'summary.json').write_text(json.dumps(summaries,indent=2))
    manifest['status']='complete'
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__': main()
