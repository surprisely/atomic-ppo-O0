"""Auditable ADP calibration bridge. Does not import or execute upstream code."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import platform
import random
import warnings
from pathlib import Path
import numpy as np
from .environment import DualServiceEnv, EnvConfig

ZONES = [237, 161, 236, 186, 162, 230, 142, 239, 163]
PERIODS = {'morning': (6, 80, 20), 'evening': (17, 240, 60)}
KEYS = ['arrival_rates', 'trip_minutes', 'relocation_minutes', 'fares', 'relocation_costs']
UNITS = {'arrival_rates': 'requests/window', 'trip_minutes': 'minute',
         'relocation_minutes': 'minute', 'fares': 'ADP_yuan', 'relocation_costs': 'ADP_yuan'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def integrate_positive_polynomial(coeff_descending, edges_seconds):
    """Integrate max(f(t/600),0)/600 dt exactly between real roots.

    f is fitted to requests per 10-minute bin. Working in x=t/600 avoids
    the erroneous ascending/descending conversion in the upstream notebook.
    """
    coeff = np.asarray(coeff_descending, dtype=float)
    edges = np.asarray(edges_seconds, dtype=float) / 600.
    if not np.isfinite(coeff).all() or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError('Polynomial and increasing edges must be finite')
    roots = np.roots(np.trim_zeros(coeff, 'f')) if np.any(coeff) else []
    real = sorted(r.real for r in roots if abs(r.imag) < 1e-9)
    primitive = np.polyint(coeff)
    values = []
    for left, right in zip(edges[:-1], edges[1:]):
        cuts = [left] + [r for r in real if left < r < right] + [right]
        value = sum(np.polyval(primitive, b) - np.polyval(primitive, a)
                    for a, b in zip(cuts[:-1], cuts[1:]) if np.polyval(coeff, (a+b)/2) > 0)
        values.append(max(0., value))
    return np.asarray(values)


def batch_events(seconds, origins, destinations, types, step_seconds, zones=9):
    if not np.isfinite(step_seconds) or step_seconds <= 0 or not np.isclose(7200/step_seconds, round(7200/step_seconds)):
        raise ValueError('step_seconds must divide 7200')
    seconds = np.asarray(seconds, dtype=float)
    arrays = [np.asarray(a) for a in (origins, destinations, types)]
    if (seconds.ndim != 1 or not np.isfinite(seconds).all()
            or any(a.shape != seconds.shape or not np.isfinite(a).all() or np.any(a != np.floor(a)) for a in arrays)):
        raise ValueError('Event arrays must be aligned, finite and indices integer')
    for a, limit in zip(arrays, (zones, zones, 3)):
        if np.any(a < 0) or np.any(a >= limit):
            raise ValueError('Event index outside domain')
    valid = (seconds >= 0) & (seconds < 7200)
    batch = np.zeros((round(7200/step_seconds), zones, zones, 3), dtype=np.int64)
    np.add.at(batch, (np.floor(seconds[valid]/step_seconds).astype(int),
                     *(a[valid].astype(int) for a in arrays)), 1)
    return batch


def financial_arrays(minutes, parameters):
    distance = np.asarray(minutes) * parameters['average_speed'] / 60.
    fares = np.stack([parameters[k+'_base_fare'] + parameters[k+'_rate_per_km']*distance
                      for k in ('express', 'premium')])
    costs = fares * parameters['relocation_cost_factor']
    for c in costs:
        np.fill_diagonal(c, 0.)
    return fares, costs


def read_source_parameters(source):
    """Literal extraction only: no upstream imports (wandb/training side effects)."""
    tree = ast.parse(Path(source).read_text(encoding='utf-8-sig'))
    revenue = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RevenueCalculator')
    result = {}
    for n in ast.walk(revenue):
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Attribute) and isinstance(n.value, ast.Constant):
            result[n.targets[0].attr] = ast.literal_eval(n.value)
    required = ('express_base_fare', 'premium_base_fare', 'express_rate_per_km',
                'premium_rate_per_km', 'average_speed', 'relocation_cost_factor')
    result = {k: result[k] for k in required}
    if any(not np.isfinite(v) or v < 0 for v in result.values()):
        raise ValueError('Invalid financial source parameters')
    zone_literals = [ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign)
                     and isinstance(n.targets[0], ast.Attribute) and n.targets[0].attr == 'original_zone_ids']
    if zone_literals != [ZONES]:
        raise ValueError('Upstream region mapping changed; review required')
    return result


def initial_fleet(express, premium, seed):
    rng = random.Random(seed)
    regions = []
    for n in (express, premium):
        counts = np.full(9, n // 9)
        for _ in range(n % 9):
            counts[rng.randint(0, 8)] += 1
        regions.extend(np.repeat(np.arange(9), counts).tolist())
    return np.asarray(regions, dtype=np.int64)


def select_period(frame, month, hour):
    dt = frame.tpep_pickup_datetime
    mask = ((dt.dt.year == 2025) & (dt.dt.month == month) & (dt.dt.weekday < 5)
            & (dt.dt.hour >= hour) & (dt.dt.hour < hour+2)
            & frame.PULocationID.isin(ZONES) & frame.DOLocationID.isin(ZONES))
    return frame.loc[mask].copy()


def calibrate(frame, step_minutes, hour):
    """Caller supplies October-only period rows; reject other months explicitly."""
    dt = frame.tpep_pickup_datetime
    if frame.empty or not ((dt.dt.year == 2025) & (dt.dt.month == 10) & (dt.dt.weekday < 5)
                          & (dt.dt.hour >= hour) & (dt.dt.hour < hour+2)).all():
        raise ValueError('Calibration requires nonempty October 2025 weekdays only')
    dates = sorted(str(d) for d in dt.dt.date.unique())
    seconds = (dt - dt.dt.normalize()).dt.total_seconds() - hour*3600
    origins = frame.PULocationID.map({z:i for i,z in enumerate(ZONES)}).to_numpy()
    destinations = frame.DOLocationID.map({z:i for i,z in enumerate(ZONES)}).to_numpy()
    counts = np.zeros((12, 9, 9))
    np.add.at(counts, ((seconds.to_numpy()//600).astype(int), origins, destinations), 1)
    counts /= len(dates)
    coeff = np.polyfit(np.arange(12)+.5, counts.reshape(12, -1), 3).T.reshape(9,9,4)
    edges = np.linspace(0, 7200, round(120/step_minutes)+1)
    od_rates = np.stack([integrate_positive_polynomial(c, edges) for c in coeff.reshape(-1,4)], axis=1).reshape(-1,9,9)
    rates = od_rates[..., None] * np.array([.5,.2,.3])
    durations = (frame.tpep_dropoff_datetime - dt).dt.total_seconds()/60
    filtered = frame.assign(duration=durations).loc[(durations > 1) & (durations < 120)]
    means = filtered.groupby(['PULocationID','DOLocationID']).duration.mean().unstack().reindex(index=ZONES, columns=ZONES).to_numpy()
    if not np.isfinite(means).all() or np.any(means <= 0):
        raise ValueError('Missing OD travel time; no fabricated fallback allowed')
    return rates, means, coeff, {'dates':dates, 'rows':len(frame), 'travel_rows':len(filtered),
                               'empirical_daily_requests':len(frame)/len(dates), 'integrated_daily_requests':float(rates.sum())}


def load_environment(path, period, step_minutes=1., replay_path=None):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata_json'])) if 'metadata_json' in data else None
        if meta is not None:
            if meta.get('period') != period:
                raise ValueError('NPZ period mismatch')
            if (meta.get('schema_version') != 1 or meta.get('units') != UNITS or meta.get('zone_ids') != ZONES
                    or meta.get('batch_timing') != 'closing' or meta.get('split') != 'train'
                    or meta.get('financial_boundary') != 'assignment_including_final_boundary'
                    or not np.isclose(meta.get('step_minutes', -1), step_minutes)):
                raise ValueError('NPZ schema/units/time/split mismatch')
        else:
            warnings.warn('Legacy NPZ: provenance and timing unverified', stacklevel=2)
        if not np.isfinite(step_minutes) or step_minutes <= 0 or not np.isclose(120/step_minutes, round(120/step_minutes)):
            raise ValueError('step_minutes must divide 120')
        _, express, premium = PERIODS[period]
        if meta and meta.get('fleet') != [express, premium]:
            raise ValueError('Fleet metadata mismatch')
        cfg = EnvConfig(round(120/step_minutes), step_minutes, express, premium,
                        batch_timing='closing' if meta else 'opening')
        batches = None
        if replay_path is not None:
            if meta is None:
                raise ValueError('Replay requires versioned calibration')
            with np.load(replay_path, allow_pickle=False) as replay:
                rm = json.loads(str(replay['metadata_json']))
                if any(rm.get(k) != meta.get(k) for k in ('period','step_minutes','zone_ids','batch_timing')):
                    raise ValueError('Replay metadata mismatch')
                if rm.get('calibration_sha256') != sha256(path):
                    raise ValueError('Replay calibration fingerprint mismatch')
                batches = replay['request_batches'].copy()
        env = DualServiceEnv(cfg, *(data[k].copy() for k in KEYS),
                             initial_regions=data['initial_regions'].copy() if 'initial_regions' in data else None,
                             request_batches=batches,
                             initialization_mode=(meta or {}).get('initialization_mode','fixed'),
                             travel_time_model=(meta or {}).get('travel_time_model','deterministic'),
                             financial_parameters=(meta or {}).get('financial_parameters'))
        if meta and env.Z != 9:
            raise ValueError('Expected nine regions')
        if replay_path is not None:
            configure_evaluation_environment(env,path,rm.get('split'))
        return env


def configure_evaluation_environment(env, path, split):
    """Apply test-only OD means after model selection; never modify training inputs."""
    if split != 'test':
        if getattr(env,'evaluation_override_sha256',None):
            raise ValueError('Test environment cannot be reused for training/validation')
        return
    with np.load(path,allow_pickle=False) as data:
        if 'test_trip_minutes' not in data: return  # Legacy data retain their old semantics.
        arrays = {k:data['test_'+k].copy() for k in KEYS[1:]}
    for k,arr in arrays.items():
        expected = (env.Z,env.Z) if k.endswith('minutes') else (2,env.Z,env.Z)
        if arr.shape != expected or not np.isfinite(arr).all() or np.any(arr < 0):
            raise ValueError('Invalid test-only environment arrays')
    for k,arr in arrays.items(): setattr(env,k,arr)
    env.trip_steps=np.maximum(1,np.ceil(env.trip_minutes/env.cfg.step_minutes).astype(int))
    env.relocation_steps=np.maximum(1,np.ceil(env.relocation_minutes/env.cfg.step_minutes).astype(int))
    digest=hashlib.sha256()
    for k,arr in arrays.items():
        digest.update(k.encode());digest.update(np.ascontiguousarray(arr,dtype=np.float64).tobytes())
    env.evaluation_override_sha256=digest.hexdigest()


def prepare(upstream, output, step_minutes=1., seed=7):
    import pandas as pd
    upstream, output = Path(upstream), Path(output)
    if not np.isfinite(step_minutes) or step_minutes <= 0 or not np.isclose(120/step_minutes, round(120/step_minutes)):
        raise ValueError('step_minutes must divide 120')
    if output.exists():
        raise FileExistsError('Use a new output directory; calibration is never overwritten')
    source = upstream/'adp_nhpp/train_adp_nhpp.py'
    params = read_source_parameters(source)
    train_path = upstream/'data/yellow_tripdata_2025-10.parquet'
    test_path = upstream/'data/yellow_tripdata_2025-11.parquet'
    columns = ['tpep_pickup_datetime','tpep_dropoff_datetime','PULocationID','DOLocationID']
    train = pd.read_parquet(train_path, columns=columns)
    test = pd.read_parquet(test_path, columns=columns)
    for frame in (train, test):
        for key in columns[:2]:
            frame[key] = pd.to_datetime(frame[key])
    inputs = {str(p.resolve()): sha256(p) for p in [source, train_path, test_path,
              upstream/'data/nonhomogeneous demand data.ipynb', upstream/'data/homogeneous demand data.ipynb',
              upstream/'data/get passengers.ipynb']}
    manifest = {'schema_version':1, 'inputs_sha256':inputs, 'seed':seed,
                'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},
                'adapter_sha256':sha256(__file__), 'periods':{}, 'status':'complete'}
    output.mkdir(parents=True)
    (output/'zone_id_map.json').write_text(json.dumps({str(z):i for i,z in enumerate(ZONES)}, indent=2))
    for period, (hour, express, premium) in PERIODS.items():
        selected = select_period(train, 10, hour)
        rates, minutes, coeff, audit = calibrate(selected, step_minutes, hour)
        fares, costs = financial_arrays(minutes, params)
        meta = dict(schema_version=1, period=period, split='train', step_minutes=step_minutes,
                    zone_ids=ZONES, fleet=[express,premium], units=UNITS, batch_timing='closing',
                    financial_boundary='assignment_including_final_boundary',
                    calibration='corrected_calibration', travel_model='deterministic_OD_mean',
                    financial_parameters=params, train_audit=audit, inputs_sha256=inputs,
                    initial_seed=seed, passenger_probabilities_E_P_D=[.5,.2,.3])
        path = output/f'{period}.npz'
        np.savez_compressed(path, arrival_rates=rates, trip_minutes=minutes, relocation_minutes=minutes,
                            fares=fares, relocation_costs=costs, initial_regions=initial_fleet(express,premium,seed),
                            polynomial_10min_descending=coeff, metadata_json=json.dumps(meta, sort_keys=True))
        replay_dir = output/f'{period}_replays'
        replay_dir.mkdir()
        base = {k:meta[k] for k in ('period','step_minutes','zone_ids','batch_timing')}
        base['calibration_sha256'] = sha256(path)
        rng = np.random.default_rng(seed)
        simulation = rng.poisson(rates).astype(np.int64)
        np.savez_compressed(replay_dir/'simulated_train.npz', request_batches=simulation,
                            metadata_json=json.dumps(dict(base, split='train', kind='simulated_from_calibration', seed=seed)))
        selected_test = select_period(test, 11, hour)
        daily = {}
        # Include all calendar weekdays, including any zero-demand day.
        for day in pd.bdate_range('2025-11-01','2025-11-30'):
            rows = selected_test.loc[selected_test.tpep_pickup_datetime.dt.date == day.date()].sort_values('tpep_pickup_datetime', kind='stable')
            dt = rows.tpep_pickup_datetime
            seconds = (dt - dt.dt.normalize()).dt.total_seconds().to_numpy() - hour*3600
            event_seed = seed + int(day.strftime('%Y%m%d'))
            types = np.random.default_rng(event_seed).choice(3, len(rows), p=[.5,.2,.3])
            batches = batch_events(seconds, rows.PULocationID.map({z:i for i,z in enumerate(ZONES)}),
                                   rows.DOLocationID.map({z:i for i,z in enumerate(ZONES)}), types, step_minutes*60)
            label = day.strftime('%Y-%m-%d')
            np.savez_compressed(replay_dir/f'test_{label}.npz', request_batches=batches,
                                metadata_json=json.dumps(dict(base, split='test', date=label, type_seed=event_seed,
                                                              kind='historical_arrivals_with_simulated_types', requests=len(rows))))
            daily[label] = len(rows)
        manifest['periods'][period] = dict(calibration_sha256=sha256(path), **audit, test_daily_counts=daily,
                                           arrays={k:{'shape':list(a.shape),'min':float(a.min()),'max':float(a.max())}
                                                   for k,a in zip(KEYS,[rates,minutes,minutes,fares,costs])})
        load_environment(path, period, step_minutes)
    manifest['outputs_sha256'] = {str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--step-minutes', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    result = prepare(args.upstream, args.output, args.step_minutes, args.seed)
    print(json.dumps({k:{'rows':v['rows'],'expected_daily_requests':v['integrated_daily_requests'],
                         'test_days':len(v['test_daily_counts'])} for k,v in result['periods'].items()}, indent=2))


if __name__ == '__main__':
    main()
