"""Versioned ADP distribution/test alignment, retaining corrected training demand."""
from __future__ import annotations
import argparse
import ast
import json
import re
from pathlib import Path
import numpy as np
from .data import (ZONES, PERIODS, sha256, read_source_parameters, financial_arrays,
                   batch_events, load_environment)


def extract_evening_test_matrix(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main_test')
    candidates=[n for n in ast.walk(main) if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='travel_matrix' for t in n.targets)]
    if len(candidates)!=1 or not isinstance(candidates[0].value,ast.Call):
        raise ValueError('Expected one literal upstream test travel matrix')
    matrix=np.asarray(ast.literal_eval(candidates[0].value.args[0]),dtype=float)
    if matrix.shape!=(9,9) or not np.isfinite(matrix).all() or np.any(matrix<=0):
        raise ValueError('Invalid upstream test matrix')
    return matrix


def source_daily_files(upstream,hour):
    folder=Path(upstream)/f'data/filtered_daily_trips_{hour}_{hour+2}_core_zones'
    result={}
    for path in sorted(folder.glob('*.parquet')):
        match=re.fullmatch(rf'trips_(2025-11-\d{{2}})_{hour}_{hour+2}_core',path.stem)
        if not match: raise ValueError(f'Unexpected upstream replay filename: {path.name}')
        result[match[1]]=path
    if not result: raise ValueError('Missing upstream filtered November replays')
    return result


def reconstruct_test_means(frames):
    import pandas as pd
    frame=pd.concat(frames,ignore_index=True)
    duration=(frame.tpep_dropoff_datetime-frame.tpep_pickup_datetime).dt.total_seconds()/60.
    valid=frame.assign(duration=duration).loc[(duration>1)&(duration<120)]
    matrix=valid.groupby(['PULocationID','DOLocationID']).duration.mean().unstack().reindex(index=ZONES,columns=ZONES).to_numpy()
    if matrix.shape!=(9,9) or not np.isfinite(matrix).all() or np.any(matrix<=0):
        raise ValueError('Missing test OD means; no invented fallback')
    return matrix


def prepare(base_data,upstream,output):
    import pandas as pd
    base,root,out=map(Path,(base_data,upstream,output))
    if out.exists(): raise FileExistsError('Use a new dataset directory')
    old=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    if old.get('status')!='complete' or old.get('calibration_cutoff_exclusive')!='2025-10-25':
        raise ValueError('Requires completed stage4 training-only calibration')
    checks={k.replace('\\','/'):v for k,v in old['outputs_sha256'].items()}
    def verified(path):
        if checks.get(path.relative_to(base).as_posix())!=sha256(path):
            raise ValueError(f'Base artifact hash mismatch: {path}')
        return path
    source=root/'adp_nhpp/train_adp_nhpp.py'
    test_source=root/'adp_nhpp/test_adp_nhpp.py'
    params=read_source_parameters(source)
    files={p:source_daily_files(root,h) for p,(h,_,_) in PERIODS.items()}
    if set(files['morning'])!=set(files['evening']):
        raise ValueError('Period test dates differ; explicit protocol review required')
    inputs={str(p.resolve()):sha256(p) for p in [source,test_source,root/'data/get passengers.ipynb',
                                               *[f for daily in files.values() for f in daily.values()]]}
    manifest=dict(schema_version=1,status='running',profile='adp_distribution_alignment_v1',
        seed=old['seed'],step_minutes=old['step_minutes'],calibration_cutoff_exclusive=old['calibration_cutoff_exclusive'],
        validation_dates=old['validation_dates'],test_dates=sorted(files['morning']),periods={},
        base_manifest_sha256=sha256(base/'manifest.json'),input_sha256=inputs,adapter_sha256=sha256(__file__),
        boundaries=['corrected October 1-24 training demand retained','November means are test-only',
                    'morning matrix reconstructed; evening matrix exact upstream literal',
                    'batch timing, active rejection and fleet sizes retained',
                    'event-keyed random streams; not original global RNG calling sequence'])
    out.mkdir(parents=True)
    columns=['tpep_pickup_datetime','tpep_dropoff_datetime','PULocationID','DOLocationID']
    for period,(hour,_,_) in PERIODS.items():
        with np.load(verified(base/f'{period}.npz'),allow_pickle=False) as f:
            arrays={k:f[k].copy() for k in f.files if k!='metadata_json'}
            meta=json.loads(str(f['metadata_json']))
        daily={}
        for label,path in files[period].items():
            frame=pd.read_parquet(path,columns=columns)
            for col in columns[:2]: frame[col]=pd.to_datetime(frame[col])
            dt=frame.tpep_pickup_datetime
            if (frame.empty or not (dt.dt.strftime('%Y-%m-%d')==label).all() or
                    not ((dt.dt.hour>=hour)&(dt.dt.hour<hour+2)&(dt.dt.day_of_week<5)).all() or
                    not frame.PULocationID.isin(ZONES).all() or not frame.DOLocationID.isin(ZONES).all()):
                raise ValueError(f'Invalid filtered source rows: {path}')
            daily[label]=frame.sort_values('tpep_pickup_datetime',kind='stable')
        minutes=(extract_evening_test_matrix(test_source) if period=='evening'
                 else reconstruct_test_means(list(daily.values())))
        fares,costs=financial_arrays(minutes,params)
        arrays.update(test_trip_minutes=minutes,test_relocation_minutes=minutes,test_fares=fares,test_relocation_costs=costs)
        provenance=dict(method='exact_upstream_hardcoded' if period=='evening' else 'reconstructed_by_upstream_preprocessing',
                        dates=sorted(daily),duration_filter_minutes='1 < duration < 120',
                        source_sha256={str(p.resolve()):inputs[str(p.resolve())] for p in files[period].values()})
        meta.update(initialization_mode='adp_random',travel_time_model='adp_exponential',
            travel_model='ADP_exponential_OD_mean_floor_0.01_minutes',financial_parameters=params,
            test_travel_provenance=provenance,profile=manifest['profile'])
        path=out/f'{period}.npz'
        np.savez_compressed(path,**arrays,metadata_json=json.dumps(meta,sort_keys=True))
        replay_dir=out/f'{period}_replays';replay_dir.mkdir()
        for label in manifest['validation_dates']:
            with np.load(verified(base/f'{period}_replays/validation_{label}.npz'),allow_pickle=False) as f:
                rm=json.loads(str(f['metadata_json']));requests=f['request_batches'].copy()
            rm['calibration_sha256']=sha256(path)
            np.savez_compressed(replay_dir/f'validation_{label}.npz',request_batches=requests,metadata_json=json.dumps(rm,sort_keys=True))
        counts={}
        for label,frame in daily.items():
            dt=frame.tpep_pickup_datetime
            seconds=(dt-dt.dt.normalize()).dt.total_seconds().to_numpy()-hour*3600
            type_seed=old['seed']+int(label.replace('-',''))
            types=np.random.default_rng(type_seed).choice(3,len(frame),p=[.5,.2,.3])
            requests=batch_events(seconds,frame.PULocationID.map({z:i for i,z in enumerate(ZONES)}),
                frame.DOLocationID.map({z:i for i,z in enumerate(ZONES)}),types,meta['step_minutes']*60)
            rm=dict(period=period,step_minutes=meta['step_minutes'],zone_ids=ZONES,batch_timing='closing',
                split='test',date=label,calibration_sha256=sha256(path),type_seed=type_seed,
                kind='upstream_filtered_historical_arrivals_with_simulated_types',requests=len(frame),
                source_sha256=inputs[str(files[period][label].resolve())])
            np.savez_compressed(replay_dir/f'test_{label}.npz',request_batches=requests,metadata_json=json.dumps(rm,sort_keys=True))
            counts[label]=int(requests.sum())
        load_environment(path,period,meta['step_minutes'])
        manifest['periods'][period]=dict(training=meta['train_audit'],test_travel_provenance=provenance,
            calibration_sha256=sha256(path),test_daily_counts=counts)
    manifest.update(status='complete',outputs_sha256={p.relative_to(out).as_posix():sha256(p)
                    for p in sorted(out.rglob('*')) if p.is_file()})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-data',required=True);p.add_argument('--upstream',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();m=prepare(a.base_data,a.upstream,a.output)
    print(json.dumps(dict(profile=m['profile'],test_days=len(m['test_dates']),status=m['status']),indent=2))


if __name__=='__main__':main()
