"""Training-only calibration with immutable chronological validation/test replays."""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import (ZONES,PERIODS,UNITS,sha256,read_source_parameters,select_period,calibrate,
                   financial_arrays,initial_fleet,batch_events,load_environment)


def fit_training_split(frame,step_minutes,hour):
    if frame.empty or (frame.tpep_pickup_datetime >= '2025-10-25').any():
        raise ValueError('Only October 1-24 training dates may be fitted')
    return calibrate(frame,step_minutes,hour)


def prepare_training_data(upstream,output,step_minutes=1.,seed=7):
    import pandas as pd
    root,out=Path(upstream),Path(output)
    if out.exists(): raise FileExistsError('Use a new dataset version')
    if step_minutes<=0 or not np.isclose(120/step_minutes,round(120/step_minutes)):
        raise ValueError('step_minutes must divide 120')
    source=root/'adp_nhpp/train_adp_nhpp.py'
    parameters=read_source_parameters(source)
    paths={m:root/f'data/yellow_tripdata_2025-{m:02d}.parquet' for m in (10,11)}
    columns=['tpep_pickup_datetime','tpep_dropoff_datetime','PULocationID','DOLocationID']
    frames={m:pd.read_parquet(path,columns=columns) for m,path in paths.items()}
    for frame in frames.values():
        for col in columns[:2]: frame[col]=pd.to_datetime(frame[col])
    manifest=dict(schema_version=1,status='running',seed=seed,step_minutes=step_minutes,
        input_sha256={str(p.resolve()):sha256(p) for p in [source,*paths.values()]},
        calibration_cutoff_exclusive='2025-10-25',validation_dates=[d.strftime('%Y-%m-%d') for d in pd.bdate_range('2025-10-27','2025-10-31')],
        test_dates=[d.strftime('%Y-%m-%d') for d in pd.bdate_range('2025-11-01','2025-11-30')],periods={},
        source_sha256=sha256(__file__))
    out.mkdir(parents=True)
    for period,(hour,express,premium) in PERIODS.items():
        october=select_period(frames[10],10,hour)
        training=october.loc[october.tpep_pickup_datetime<'2025-10-25']
        rates,minutes,coeff,audit=fit_training_split(training,step_minutes,hour)
        fares,costs=financial_arrays(minutes,parameters)
        meta=dict(schema_version=1,period=period,split='train',step_minutes=step_minutes,zone_ids=ZONES,
            fleet=[express,premium],units=UNITS,batch_timing='closing',financial_boundary='assignment_including_final_boundary',
            calibration='corrected_training_split',travel_model='deterministic_OD_mean',train_audit=audit,
            financial_parameters=parameters,initial_seed=seed,passenger_probabilities_E_P_D=[.5,.2,.3],
            input_sha256={str(paths[10].resolve()):sha256(paths[10]),str(source.resolve()):sha256(source)})
        npz=out/f'{period}.npz'
        np.savez_compressed(npz,arrival_rates=rates,trip_minutes=minutes,relocation_minutes=minutes,fares=fares,
            relocation_costs=costs,initial_regions=initial_fleet(express,premium,seed),polynomial_10min_descending=coeff,
            metadata_json=json.dumps(meta,sort_keys=True))
        replay_dir=out/f'{period}_replays';replay_dir.mkdir()
        daily={}
        for split,month,dates in [('validation',10,manifest['validation_dates']),('test',11,manifest['test_dates'])]:
            selected=select_period(frames[month],month,hour)
            for label in dates:
                dt=pd.Timestamp(label)
                rows=selected.loc[selected.tpep_pickup_datetime.dt.date==dt.date()].sort_values('tpep_pickup_datetime',kind='stable')
                pick=rows.tpep_pickup_datetime
                seconds=(pick-pick.dt.normalize()).dt.total_seconds().to_numpy()-hour*3600
                types=np.random.default_rng(seed+int(dt.strftime('%Y%m%d'))).choice(3,len(rows),p=[.5,.2,.3])
                requests=batch_events(seconds,rows.PULocationID.map({z:i for i,z in enumerate(ZONES)}),
                    rows.DOLocationID.map({z:i for i,z in enumerate(ZONES)}),types,step_minutes*60)
                rm=dict(period=period,step_minutes=step_minutes,zone_ids=ZONES,batch_timing='closing',split=split,date=label,
                    calibration_sha256=sha256(npz),kind='historical_arrivals_with_simulated_types',requests=len(rows))
                np.savez_compressed(replay_dir/f'{split}_{label}.npz',request_batches=requests,metadata_json=json.dumps(rm,sort_keys=True))
                daily[f'{split}_{label}']=len(rows)
        load_environment(npz,period,step_minutes)
        manifest['periods'][period]=dict(training=audit,replay_counts=daily,calibration_sha256=sha256(npz))
    manifest['outputs_sha256']={str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file()}
    manifest['status']='complete'
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',required=True);p.add_argument('--output',required=True)
    p.add_argument('--step-minutes',type=float,default=1.);p.add_argument('--seed',type=int,default=7)
    a=p.parse_args();m=prepare_training_data(a.upstream,a.output,a.step_minutes,a.seed)
    print(json.dumps({k:v['training'] for k,v in m['periods'].items()},indent=2))


if __name__=='__main__': main()
