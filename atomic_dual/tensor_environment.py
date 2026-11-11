"""Batched Tensor simulator with sequential atomic decisions within each episode.

Exogenous requests/orders retain the NumPy environment's seeds. Stochastic travel
uses an explicit independent atomic-step noise contract, not NumPy event-keyed PCG.
No CPU reads of device state are needed during a timestamp/episode.
"""
from __future__ import annotations

import copy
import math
import types
import numpy as np
import torch
from torch.nn.functional import one_hot

from .batched_sampling import episode_uniforms


def travel_uniforms(seed,horizon):
    return np.random.default_rng(np.random.SeedSequence([int(seed),0x54454E53])).random(horizon)


def _reference_task_minutes(self,mean,key):
    if getattr(self,'_tensor_noise_seed',None)!=self.episode_seed:
        self._tensor_noise_values=travel_uniforms(self.episode_seed,self.T*self.N)
        self._tensor_noise_seed=self.episode_seed
    u=self._tensor_noise_values[self.atomic_count]
    return max(-math.log1p(-u)*(mean if mean>0 else .1),.01)


def noise_reference(template):
    """NumPy oracle with only its travel RNG replaced by the shared test tape."""
    env=copy.deepcopy(template)
    env._sample_task_minutes=types.MethodType(_reference_task_minutes,env)
    return env


def prepare_inputs(template,seeds):
    """Prepare policy-independent tapes once; never called inside the GPU loop."""
    env=copy.deepcopy(template);env._cache_enabled=False
    batches=[];orders=[];initial=[]
    for seed in seeds:
        env.reset(seed)
        initial.append(env.episode_initial_regions.copy())
        rb=[];ro=[]
        for t in range(env.T):
            if t:
                env.t=t;env._begin_timestamp()
            rb.append(env.requests.copy());ro.append(env._car_order.copy())
        batches.append(rb);orders.append(ro)
    return dict(batches=np.asarray(batches,dtype=np.int64),orders=np.asarray(orders,dtype=np.int64),
        initial=np.asarray(initial,dtype=np.int64),
        policy=np.stack([episode_uniforms(s,env.T*env.N) for s in seeds]),
        travel=np.stack([travel_uniforms(s,env.T*env.N) for s in seeds]))


class TensorFleetEnv:
    def __init__(self,template,seeds,device='cpu',travel_rng='step'):
        self.cfg=template.cfg;self.T=template.T;self.N=template.N;self.Z=template.Z
        self.K=len(seeds);self.H=self.T*self.N;self.A=template.action_dim
        self.B=self.cfg.max_eta_feature+1;self.F=2*3*self.Z*self.B
        self.D=self.F+self.Z*self.Z*3+2*self.Z+self.Z+8
        self.device=torch.device(device)
        self.travel_rng=travel_rng;self.event_tables=None
        def tensor(a,dtype=None):return torch.as_tensor(a,dtype=dtype,device=self.device)
        self.trip=tensor(template.trip_minutes,torch.float64)
        self.relocation=tensor(template.relocation_minutes,torch.float64)
        self.fares=tensor(template.fares,torch.float64)
        self.costs=tensor(template.relocation_costs,torch.float64)
        self.stochastic=template.travel_time_model=='adp_exponential'
        p=template.financial_parameters or {}
        self.base=tensor([p.get('express_base_fare',0),p.get('premium_base_fare',0)],torch.float64)
        self.rate=tensor([p.get('express_rate_per_km',0),p.get('premium_rate_per_km',0)],torch.float64)
        self.speed=p.get('average_speed',0);self.factor=p.get('relocation_cost_factor',0)
        self.rows=torch.arange(self.K,device=self.device)
        self.classes=(torch.arange(self.N,device=self.device)>=self.cfg.n_express).long()
        self.zones=torch.arange(self.Z,device=self.device)
        self.q=torch.arange(3,device=self.device)
        shape=(self.K,self.N)
        self.region=torch.zeros(shape,dtype=torch.long,device=self.device)
        self.status=torch.zeros_like(self.region);self.destination=torch.full_like(self.region,-1)
        self.eta=torch.zeros_like(self.region);self.relocation_counts=torch.zeros_like(self.region)
        self.profit=torch.zeros(self.K,dtype=torch.float64,device=self.device)
        self.served=torch.zeros((self.K,6),dtype=torch.long,device=self.device)
        self.rejected=torch.zeros(self.K,dtype=torch.long,device=self.device)
        self.arrived=torch.zeros_like(self.rejected)
        self.t=torch.zeros((),dtype=torch.long,device=self.device)
        self.requests=torch.zeros((self.K,self.Z*self.Z*3),dtype=torch.long,device=self.device)
        self.batch_arrivals=torch.zeros_like(self.requests)
        self.event_offsets=torch.zeros_like(self.requests)
        self.order=torch.zeros(shape,dtype=torch.long,device=self.device)
        self.hist=torch.zeros((self.K,self.F),dtype=torch.float32,device=self.device)
        self.remaining=torch.zeros((self.K,2*self.Z),dtype=torch.float32,device=self.device)
        self.ones=torch.ones(shape,dtype=torch.float32,device=self.device)
        self.load_inputs(template,seeds,initial=True)
        self.reset_state()

    def load_inputs(self,template,seeds,initial=False):
        arrays=prepare_inputs(template,seeds)
        for key,value in arrays.items():
            value=torch.as_tensor(value,device=self.device)
            if initial:setattr(self,key+'_tape',value)
            else:getattr(self,key+'_tape').copy_(value)
        if self.stochastic and self.travel_rng=='event':
            from .event_tables import EventTables
            if self.event_tables is None:self.event_tables=EventTables(self,template,arrays,seeds)
            else:self.event_tables.load(arrays,seeds)

    def prepare_event_timestamp(self):
        if self.event_tables is not None:self.event_tables.prepare_timestamp()

    def reset_state(self):
        self.region.copy_(self.initial_tape)
        self.status.zero_();self.destination.fill_(-1);self.eta.zero_();self.relocation_counts.zero_()
        self.profit.zero_();self.served.zero_();self.rejected.zero_();self.arrived.zero_();self.t.zero_()

    def _hist_index(self,classes,status,region,destination,eta):
        zone=torch.where(status==0,region,destination)
        return ((classes*3+status)*self.Z+zone)*self.B+eta.clamp_max(self.B-1)

    def begin_timestamp(self):
        index=self.t.reshape(1)
        self.requests.copy_(self.batches_tape.index_select(1,index).reshape(self.K,-1))
        if self.event_tables is not None:
            self.batch_arrivals.copy_(self.requests)
            self.event_offsets.copy_(self.event_tables.offsets.index_select(1,index).reshape(self.K,-1))
        self.order.copy_(self.orders_tape.index_select(1,index).reshape(self.K,self.N))
        self.arrived.add_(self.requests.sum(-1))
        indices=self._hist_index(self.classes,self.status,self.region,self.destination,self.eta)
        self.hist.zero_();self.hist.scatter_add_(1,indices,self.ones)
        self.remaining.zero_()
        self.remaining.scatter_add_(1,self.classes*self.Z+self.region,(self.status==0).float())

    def _current(self,j):
        cid=self.order[:,j]
        return (cid,self.classes[cid],self.region[self.rows,cid],self.status[self.rows,cid],
                self.destination[self.rows,cid],self.eta[self.rows,cid])

    def observe(self,j):
        cid,classes,region,status,destination,eta=self._current(j)
        zone=torch.where(status==0,region,destination)
        # Divide integer counts in float64 then cast: match NumPy float32 features
        # without reciprocal-multiplication rounding differences on CUDA.
        meta=torch.stack(((self.t.double()+(self.cfg.batch_timing=='closing')).expand(self.K)/self.T,
                          torch.full((self.K,),j/self.N,dtype=torch.float64,device=self.device),
                          eta.clamp_max(1000).double()/1000.),dim=-1).float()
        return torch.cat(((self.hist.double()/self.N).float(),(self.requests.double()/self.N).float(),
            (self.remaining.double()/self.N).float(),one_hot(zone,self.Z).float(),
            one_hot(classes,2).float(),one_hot(status,3).float(),meta),dim=-1)

    def action_mask(self,j):
        _,classes,origin,status,_,_=self._current(j)
        request=self.requests.reshape(self.K,self.Z,self.Z*3).gather(1,
            origin[:,None,None].expand(self.K,1,self.Z*3)).reshape(self.K,self.Z,3)
        compatible=(self.q[None,:]==2)|(self.q[None,:]==classes[:,None])
        serve=(request>0)&compatible[:,None,:]&(status==0)[:,None,None]
        relocate=(self.zones[None,:]!=origin[:,None])&(status==0)[:,None]
        return torch.cat((serve.transpose(1,2).reshape(self.K,3*self.Z),relocate,
                          torch.ones((self.K,1),dtype=torch.bool,device=self.device)),dim=-1)

    def step(self,actions,j):
        cid,classes,origin,status,destination,eta=self._current(j)
        serve=actions<3*self.Z;relocate=(actions>=3*self.Z)&(actions<4*self.Z)
        task=serve|relocate
        dest=actions.remainder(self.Z);q=(actions//self.Z).clamp_max(2)
        mean=torch.where(serve,self.trip[origin,dest],self.relocation[origin,dest])
        if self.event_tables is not None:
            request_index=((origin*self.Z+dest)*3+q)[:,None]
            ordinal=(self.batch_arrivals-self.requests).gather(1,request_index).squeeze(1)
            event_index=self.event_offsets.gather(1,request_index).squeeze(1)+ordinal
            event_index=event_index.clamp_max(self.event_tables.capacity-1)
            service=self.event_tables.service[event_index]
            relocation=self.event_tables.relocation[self.rows,cid,dest]
            minutes=torch.where(serve,service[:,0],relocation[:,0])
            steps=torch.where(serve,service[:,1],relocation[:,1]).long()
            reward=torch.where(serve,service[self.rows,classes+2],torch.where(relocate,relocation[:,2],0.))
        elif self.stochastic:
            index=(self.t*self.N+j).reshape(1)
            u=self.travel_tape.index_select(1,index).reshape(self.K)
            minutes=(-torch.log1p(-u)*torch.where(mean>0,mean,.1)).clamp_min(.01)
            fare=self.base[classes]+self.rate[classes]*self.speed*minutes/60.
            reward=torch.where(serve,fare,torch.where(relocate,-fare*self.factor,0.))
        else:
            minutes=mean
            reward=torch.where(serve,self.fares[classes,origin,dest],
                               torch.where(relocate,-self.costs[classes,origin,dest],0.))
        if self.event_tables is None:
            steps=torch.ceil(minutes/self.cfg.step_minutes).long().clamp_min(1)
        new_status=torch.where(serve,1,torch.where(relocate,2,status))
        new_dest=torch.where(task,dest,destination);new_eta=torch.where(task,steps,eta)
        old_index=self._hist_index(classes,status,origin,destination,eta)
        new_index=self._hist_index(classes,new_status,origin,new_dest,new_eta)
        self.hist.scatter_add_(1,old_index[:,None],-torch.ones((self.K,1),device=self.device))
        self.hist.scatter_add_(1,new_index[:,None],torch.ones((self.K,1),device=self.device))
        self.remaining.scatter_add_(1,(classes*self.Z+origin)[:,None],-(status==0).float()[:,None])
        request_index=((origin*self.Z+dest)*3+q)[:,None]
        self.requests.scatter_add_(1,request_index,-serve.long()[:,None])
        self.served.scatter_add_(1,(classes*3+q)[:,None],serve.long()[:,None])
        self.status[self.rows,cid]=new_status;self.destination[self.rows,cid]=new_dest
        self.eta[self.rows,cid]=new_eta
        self.relocation_counts[self.rows,cid]+=relocate.long()
        self.profit.add_(reward)
        self.last_task_minutes=torch.where(task,minutes,0.)
        return reward

    def finish_timestamp(self):
        self.rejected.add_(self.requests.sum(-1));self.requests.zero_()
        advance=(self.t<self.T-1)|(self.cfg.batch_timing=='opening')
        new_eta=self.eta-((self.status!=0)&advance).long()
        arrived=(self.status!=0)&(new_eta==0)
        self.region.copy_(torch.where(arrived,self.destination,self.region))
        self.destination.masked_fill_(arrived,-1);self.status.masked_fill_(arrived,0)
        self.eta.copy_(new_eta);self.t.add_(1)
