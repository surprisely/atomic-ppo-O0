"""Deterministic, same-environment dispatch baselines. No future replay access."""
import numpy as np

BASELINES = ('all_pass','greedy_fare','compatible_stay','demand_relocate')
VERSION = 'stage3-v1'


def _service(env, mask, dedicated_first):
    car = env.current_car()
    actions = np.flatnonzero(mask[:3*env.Z])
    if not len(actions): return None
    if dedicated_first:
        dedicated = actions[actions//env.Z == int(car.service_class)]
        if len(dedicated): actions = dedicated
    # Stable action-index tie break is shared across all runs.
    return int(max(actions,key=lambda a:(env.fares[int(car.service_class),car.region,a%env.Z],-int(a))))


def _opportunity(env, zone, delay, lookahead):
    """Single-car fare opportunity, using calibrated rates and current fleet only."""
    car = env.current_car()
    start = env.t+max(1,delay)
    end = min(env.T,start+lookahead)
    if start >= end: return 0.
    k = int(car.service_class)
    # Known forecasts, not realized future request batches.
    od = env.arrival_rates[start:end,zone,:,k] + env.arrival_rates[start:end,zone,:,2]
    totals = od.sum(axis=0)
    demand = float(totals.sum())
    if demand <= 0: return 0.
    supply = sum(c is not car and int(c.service_class)==k and
        ((int(c.status)==0 and c.region==zone) or (int(c.status)!=0 and c.destination==zone and c.eta<=delay))
        for c in env.cars)
    fare = float(np.dot(totals,env.fares[k,zone])/demand)
    return fare*min(1.,demand/(supply+1))


def choose_action(name, env, lookahead_steps=10):
    if name not in BASELINES: raise ValueError(f'Unknown baseline {name}')
    if lookahead_steps < 1: raise ValueError('Positive lookahead required')
    mask = env.action_mask()
    if name == 'all_pass' or mask.sum()==1: return env.pass_action
    serve = _service(env,mask,name!='greedy_fare')
    if serve is not None: return serve
    if name != 'demand_relocate': return env.pass_action
    car = env.current_car()
    best = _opportunity(env,car.region,1,lookahead_steps)
    action = env.pass_action
    for destination in range(env.Z):
        candidate = env.relocate_action(destination)
        if not mask[candidate]: continue
        delay = int(env.relocation_steps[car.region,destination])
        score = (_opportunity(env,destination,delay,lookahead_steps)
                 - env.relocation_costs[int(car.service_class),car.region,destination])
        if score > best+1e-12:
            action,best = candidate,score
    return int(action)
