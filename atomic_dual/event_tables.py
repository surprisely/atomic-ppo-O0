"""Original event-key RNG with timestamp prefetch and exact CPU financial math.

PCG64 bits are generated in batches on the selected device. Python math.log1p
is retained at timestamp boundaries to match the original float64 task minutes,
ceil and fare/cost bit patterns, rather than just their distributions.
"""
import math
import weakref
import numpy as np
import torch

from .event_rng import pcg64_first_uniform


class EventCapacityError(RuntimeError):
    pass


def _task_minutes(uniforms,mean):
    phi=np.fromiter((-math.log1p(-float(u)) for u in uniforms.ravel()),
                    dtype=np.float64,count=uniforms.size).reshape(uniforms.shape)
    return np.maximum(phi*np.where(mean>0,mean,.1),.01)


def _steps(minutes,delta):
    steps=np.maximum(1,np.ceil(minutes/delta))
    if np.any(steps>2**53):raise ValueError('Event table task steps exceed exact float64 integer range')
    return steps


class EventTables:
    def __init__(self,env,template,arrays,seeds):
        if type(np.random.default_rng(0).bit_generator).__name__!='PCG64':
            raise ValueError('Event compatibility requires NumPy default_rng PCG64')
        self.env=weakref.proxy(env);self.template=template;self.device=env.device
        count=int(arrays['batches'].sum())
        self.capacity=max(1,math.ceil(count*1.25)+1)
        self.service=torch.zeros((self.capacity,4),dtype=torch.float64,device=self.device)
        self.offsets=torch.zeros((env.K,env.T,env.Z*env.Z*3),dtype=torch.long,device=self.device)
        self.relocation=torch.zeros((env.K,env.N,env.Z,3),dtype=torch.float64,device=self.device)
        self.classes=np.broadcast_to((np.arange(env.N)>=env.cfg.n_express)[None,:,None],(env.K,env.N,env.Z))
        self.load(arrays,seeds)

    def load(self,arrays,seeds):
        e=self.env
        if any(int(s)<0 or int(s)>2**32-1 for s in seeds):
            raise ValueError('Event-compatible Tensor seeds must fit uint32')
        counts=arrays['batches'].ravel()
        if np.any(counts>2**32-1):raise ValueError('Event request counts must fit uint32')
        total=int(counts.sum())
        if total>=self.capacity:raise EventCapacityError('Service event table capacity exceeded; recapture required')
        offsets=np.cumsum(counts,dtype=np.int64)-counts
        self.offsets.copy_(torch.as_tensor(offsets.reshape(e.K,e.T,-1),device=self.device))
        self.seeds=torch.as_tensor(seeds,dtype=torch.long,device=self.device)
        self.service.zero_()
        if not total:return
        cells=np.repeat(np.arange(counts.size,dtype=np.int64),counts)
        ordinal=np.arange(total,dtype=np.int64)-np.repeat(offsets,counts)
        q=cells%3;dest=(cells//3)%e.Z;origin=(cells//(3*e.Z))%e.Z
        time=(cells//(3*e.Z*e.Z))%e.T;episode=cells//(e.T*3*e.Z*e.Z)
        keys=np.stack((np.asarray(seeds,dtype=np.int64)[episode],np.zeros(total,dtype=np.int64),
                       time,origin,dest,q,ordinal),axis=-1)
        uniform=pcg64_first_uniform(torch.as_tensor(keys,device=self.device),validate=False).cpu().numpy()
        minutes=_task_minutes(uniform,self.template.trip_minutes[origin,dest])
        p=self.template.financial_parameters
        fares=[p[c+'_base_fare']+(p[c+'_rate_per_km']*p['average_speed'])*minutes/60.
               for c in ('express','premium')]
        packed=np.stack((minutes,_steps(minutes,e.cfg.step_minutes),*fares),axis=-1)
        self.service[:total].copy_(torch.as_tensor(packed,device=self.device))

    def prepare_timestamp(self):
        e=self.env;k,n,z=e.K,e.N,e.Z
        shape=(k,n,z)
        keys=torch.stack((self.seeds[:,None,None].expand(shape),torch.ones(shape,dtype=torch.long,device=self.device),
            torch.arange(n,device=self.device)[None,:,None].expand(shape),
            e.relocation_counts[:,:,None].expand(shape),e.region[:,:,None].expand(shape),
            torch.arange(z,device=self.device)[None,None,:].expand(shape)),dim=-1).reshape(-1,6)
        uniform=pcg64_first_uniform(keys,validate=False).reshape(shape)
        mean=e.relocation[e.region[:,:,None],e.zones[None,None,:]]
        # One batch readback (uniform and original mean), not one per atomic action.
        values=torch.stack((uniform,mean),dim=-1).cpu().numpy()
        minutes=_task_minutes(values[...,0],values[...,1])
        p=self.template.financial_parameters
        base=np.where(self.classes,p['premium_base_fare'],p['express_base_fare'])
        rate=np.where(self.classes,p['premium_rate_per_km'],p['express_rate_per_km'])
        fare=base+(rate*p['average_speed'])*minutes/60.
        reward=-fare*p['relocation_cost_factor']
        packed=np.stack((minutes,_steps(minutes,e.cfg.step_minutes),reward),axis=-1)
        self.relocation.copy_(torch.as_tensor(packed,device=self.device))
