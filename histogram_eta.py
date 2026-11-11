"""Original histogram -> MLP, with sparse storage for exact integer-ETA bins.

The first Linear is evaluated algebraically on sparse histogram counts. There
is no learned vehicle encoder, candidate feature augmentation or pooling layer.
"""
import copy
import hashlib
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict,dataclass

import numpy as np
import torch
from torch.nn import functional as F
from atomic_dual.batched_sampling import BatchedEpisodeSampler
from atomic_dual.ppo import AverageRewardPPO,MLP
from atomic_dual.sampling import rollout


@dataclass(frozen=True)
class HistogramSpec:
    eta: str
    max_eta: int
    actor_od: str = 'full'
    critic_od: str = 'full'

    def __post_init__(self):
        if self.eta not in ('bucket','exact') or self.max_eta<8: raise ValueError('Invalid histogram specification')
        if self.eta=='bucket' and self.max_eta!=8: raise ValueError('L0 requires original 8+ buckets')
        if any(od not in ('full','marginal') for od in (self.actor_od,self.critic_od)):
            raise ValueError('Invalid OD representation')
        if self.eta=='exact' and (self.actor_od,self.critic_od)!=('full','full'):
            raise ValueError('OD aggregation requires the L0 histogram')


def exact_eta_bound(env,snapshots=()):
    """Finite bound of THIS implementation's 53-bit uniform inverse-CDF draw.

    Mathematical exponential support is unbounded. But _sample_task_minutes
    uses U=k/2**53<1, so -log1p(-U)<=53*log(2). Include existing snapshot ETAs.
    One conservative extra bin absorbs numerical boundary rounding, not a tail.
    """
    if env.travel_time_model=='adp_exponential':
        mean=max(float(env.trip_minutes.max()),float(env.relocation_minutes.max()),.1)
        bound=math.ceil(max(mean*(-math.log1p(-np.nextafter(1.,0.))),.01)/env.cfg.step_minutes)+1
    else: bound=max(int(env.trip_steps.max()),int(env.relocation_steps.max()))
    existing=max((int(c.eta) for s in snapshots for c in s.cars),default=0)
    return max(8,bound,existing)


class HistogramEnv:
    """Old observation tail, lossless sparse integer histogram (index,count).

    N slots suffice: at most N nonzero histogram bins. Sparse storage avoids a
    K*H*64k dense replay matrix; the network still implements the ORIGINAL MLP.
    """
    def __init__(self,base,representation):
        object.__setattr__(self,'base',base); object.__setattr__(self,'representation',representation)
        object.__setattr__(self,'_counts',{})
        if self.hist_dim>=2**24: raise ValueError('Histogram indices exceed exact float32 sparse-storage range')
        base.enable_observation_cache()
        if representation.eta=='exact' and not base.done: self._rebuild()

    def __getattr__(self,name): return getattr(object.__getattribute__(self,'base'),name)
    def __setattr__(self,name,value):
        if name in ('base','representation','_counts'): object.__setattr__(self,name,value)
        else: setattr(self.base,name,value)
    @property
    def hist_dim(self): return 2*3*self.Z*(self.representation.max_eta+1)
    @property
    def tail_dim(self): return 3*self.Z**2+2*self.Z+self.Z+8
    def research_settings(self):
        return dict(self.base.research_settings(),representation=asdict(self.representation),encoder='original_histogram_mlp_v2')
    def _index(self,c):
        eta=int(c.eta)
        if eta!=c.eta or eta<0: raise ValueError('Nonnegative integer ETA required')
        if eta>self.representation.max_eta: raise ValueError('ETA exceeds declared exact histogram bound; never silently clipped')
        zone=c.region if c.status==0 else c.destination
        return ((int(c.service_class)*3+int(c.status))*self.Z+zone)*(self.representation.max_eta+1)+eta
    def _rebuild(self):
        if self.representation.eta=='bucket': return
        counts={}
        for c in self.cars:
            idx=self._index(c); counts[idx]=counts.get(idx,0)+1
        self._counts=counts
    def reset(self,*a,**kw):
        obs=self.base.reset(*a,**kw)
        if self.representation.eta=='bucket': return obs
        self._rebuild()
        return self.observe()
    def observe(self):
        old=self.base.observe()
        if self.representation.eta=='bucket': return old
        result=np.zeros(2*self.N+self.tail_dim,dtype=np.float32)
        for j,idx in enumerate(sorted(self._counts)):
            result[j]=idx; result[self.N+j]=self._counts[idx]/self.N
        result[2*self.N:]=old[-self.tail_dim:]
        return result
    def dense_observe(self):
        if self.representation.eta=='bucket': return self.base.observe()
        packed=self.observe(); hist=np.zeros(self.hist_dim,dtype=np.float32)
        np.add.at(hist,packed[:self.N].astype(np.int64),packed[self.N:2*self.N])
        return np.concatenate((hist,packed[2*self.N:]))
    def step(self,action):
        if self.representation.eta=='bucket': return self.base.step(action)
        c=self.current_car(); old=self._index(c); t=self.t
        _,r,done,info=self.base.step(action)
        if done: return None,r,done,info
        if self.t!=t: self._rebuild()
        else:
            new=self._index(c)
            if old!=new:
                self._counts[old]-=1
                if not self._counts[old]: del self._counts[old]
                self._counts[new]=self._counts.get(new,0)+1
        return self.observe(),r,done,info


class HistogramMLP(MLP):
    """Same Linear/Tanh/Linear/Tanh/Linear; sparse first Linear evaluation."""
    def __init__(self,env,output,width):
        self.N=env.N; self.hist_dim=env.hist_dim
        super().__init__(env.hist_dim+env.tail_dim,output,width)
    def forward(self,x):
        single=x.ndim==1
        if single: x=x.unsqueeze(0)
        if len(x)>512 and not torch.is_grad_enabled():
            y=torch.cat([self.forward(p) for p in x.split(512)])
        else:
            # Sum w[:,bin]*count/N, exactly the original first Linear's formula.
            linear=self.net[0]
            pooled=F.embedding_bag(x[:,:self.N].long(),linear.weight[:,:self.hist_dim].T,
                per_sample_weights=x[:,self.N:2*self.N],mode='sum')
            y=pooled+F.linear(x[:,2*self.N:],linear.weight[:,self.hist_dim:],linear.bias)
            for layer in list(self.net)[1:]: y=layer(y)
        return y[0] if single else y


class MarginalODMLP(MLP):
    """Project a shared L0 observation before the first Linear."""
    def __init__(self,env,output,width,reference=None):
        self.start=env.hist_dim
        self.end=self.start+3*env.Z**2
        self.Z=env.Z
        margin_dim=6*env.Z
        super().__init__(env.hist_dim+env.tail_dim-3*env.Z**2+margin_dim,output,width)
        if reference is not None:
            state=reference.state_dict()
            weight=state['net.0.weight'].cpu()
            shared_end=min(self.start+margin_dim,self.end)
            margin_extra=self.net[0].weight.detach()[:,shared_end:self.start+margin_dim]
            state['net.0.weight']=torch.cat((weight[:,:shared_end],margin_extra,weight[:,self.end:]),dim=1)
            self.load_state_dict(state)

    def forward(self,x):
        if x.ndim==2 and len(x)>8192 and not torch.is_grad_enabled():
            return torch.cat([self.forward(batch) for batch in x.split(8192)])
        orders=x[...,self.start:self.end].reshape(*x.shape[:-1],self.Z,self.Z,3)
        origins=orders.sum(dim=-2).reshape(*x.shape[:-1],3*self.Z)
        destinations=orders.sum(dim=-3).reshape_as(origins)
        return super().forward(torch.cat((x[...,:self.start],origins,destinations,x[...,self.end:]),dim=-1))


def initialize_from_bucket(target,reference,env):
    """Split L0's >=8 coefficient into independent, initially equal coefficients."""
    state=reference.state_dict(); expanded={}
    for k,v in state.items():
        if k=='net.0.weight':
            old_hist=2*3*env.Z*9
            h=v[:,:old_hist].reshape(v.shape[0],2*3*env.Z,9)
            mapping=torch.arange(env.representation.max_eta+1).clamp(max=8)
            hist=h[:,:,mapping].reshape(v.shape[0],-1)
            expanded[k]=torch.cat((hist,v[:,old_hist:]),dim=1)
        else: expanded[k]=v.clone()
    target.load_state_dict(expanded)


def parameter_hash(*networks):
    h=hashlib.sha256()
    for net in networks:
        for k,v in net.state_dict().items(): h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def bucket_projection_hash(env,*networks):
    """Canonical initialization check: identical old bins, tail, later layers."""
    h=hashlib.sha256()
    for net in networks:
        for k,v in net.state_dict().items():
            if k=='net.0.weight':
                v=torch.cat((v[:,:env.hist_dim].reshape(v.shape[0],-1,env.representation.max_eta+1)[:,:,:9].flatten(1),
                    v[:,env.hist_dim:]),dim=1)
            h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _initialize(env,hidden):
    global _env,_actor
    torch.set_num_threads(1); _env=env; env.reset(0)
    if env.representation.eta=='bucket':
        if env.representation.actor_od=='marginal':
            _actor=MarginalODMLP(env,env.action_dim,hidden).eval()
        else:
            _actor=MLP(env.hist_dim+env.tail_dim,env.action_dim,hidden).eval()
    else:
        _actor=HistogramMLP(env,env.action_dim,hidden).eval()
def _sample(task):
    weights,seed,uniform=task
    _actor.load_state_dict({k:torch.from_numpy(v) for k,v in weights.items()}); torch.manual_seed(seed)
    tr=rollout(_env,_actor,'cpu',seed,uniform=uniform); obs=tr.pop('states').base; tr.pop('next_states'); return obs,tr


class HistogramSampler:
    def __init__(self,env,cfg,workers): self.env=copy.deepcopy(env); self.cfg=cfg; self.workers=workers; self.pool=None
    def __enter__(self):
        if self.workers>1:
            self.pool=ProcessPoolExecutor(max_workers=self.workers,mp_context=mp.get_context('spawn'),initializer=_initialize,
                initargs=(self.env,self.cfg.hidden_dim))
        return self
    def collect(self,actor,seeds):
        uniform=self.cfg.rollout_seed_mode=='uniform'
        if self.pool is None:
            actor=copy.deepcopy(actor).cpu().eval(); result=[]
            with torch.random.fork_rng(devices=[]):
                for s in seeds:
                    torch.random.default_generator.manual_seed(s); result.append(rollout(self.env,actor,'cpu',s,uniform=uniform))
            return result
        weights={k:v.detach().cpu().numpy().copy() for k,v in actor.state_dict().items()}; result=[]
        for obs,tr in self.pool.map(_sample,((weights,s,uniform) for s in seeds)):
            tr.update(states=obs[:-1],next_states=obs[1:]); result.append(tr)
        return result
    def __exit__(self,*a):
        if self.pool: self.pool.shutdown(wait=True,cancel_futures=True)


class HistogramPPO(AverageRewardPPO):
    def __init__(self,env,cfg,seed=0):
        sampling=(cfg.rollout_backend,cfg.rollout_seed_mode)
        if cfg.rollout_device!='cpu' or sampling not in (('episode','episode'),('batched','uniform')):
            raise ValueError('Requires CPU episode or CPU uniform batched sampling')
        if cfg.rollout_backend=='batched' and env.representation.eta!='bucket':
            raise ValueError('Batched sampling requires the L0 histogram')
        if cfg.device!='cpu' and env.representation.eta!='bucket':
            raise ValueError('CUDA updates require the L0 histogram')
        super().__init__(env,cfg,seed)
        if env.representation.eta=='bucket':
            with torch.random.fork_rng(devices=[]):
                if env.representation.actor_od=='marginal':
                    self.actor=MarginalODMLP(env,env.action_dim,cfg.hidden_dim,self.actor).to(self.device)
                    self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=cfg.actor_lr)
                if env.representation.critic_od=='marginal':
                    self.critic=MarginalODMLP(env,1,cfg.hidden_dim,self.critic).to(self.device)
                    self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=cfg.critic_lr)
            self.fit_actor=copy.deepcopy(self.actor).cpu().eval()
            return
        torch.manual_seed(seed)
        reference_actor=MLP(2*3*env.Z*9+env.tail_dim,env.action_dim,cfg.hidden_dim)
        reference_critic=MLP(2*3*env.Z*9+env.tail_dim,1,cfg.hidden_dim)
        rng=torch.get_rng_state()
        self.actor=HistogramMLP(env,env.action_dim,cfg.hidden_dim)
        self.critic=HistogramMLP(env,1,cfg.hidden_dim)
        initialize_from_bucket(self.actor,reference_actor,env); initialize_from_bucket(self.critic,reference_critic,env)
        torch.set_rng_state(rng)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=cfg.actor_lr)
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=cfg.critic_lr)
        self.fit_actor=copy.deepcopy(self.actor).eval()
    def update(self,trajs):
        self.fit_actor.load_state_dict(self.actor.state_dict())
        result=super().update(trajs)
        if self.device.type=='cuda': torch.cuda.synchronize(self.device)
        return result
    def train(self,iterations,episodes_per_iteration,seed=0,on_iteration=None):
        schedule=dict(seed=seed,episodes_per_iteration=episodes_per_iteration,sampling_contract='histogram_episode_v2')
        batched=self.cfg.rollout_backend=='batched'
        if batched:
            schedule.update(sampling_contract='episode_uniform_v1',rollout_backend='batched',
                rollout_batch_size=self.cfg.rollout_batch_size)
        if self.training_schedule is not None and self.training_schedule!=schedule: raise ValueError('Schedule mismatch')
        self.training_schedule=schedule; reports=[]
        sampler=(BatchedEpisodeSampler(self.env,self.cfg) if batched else
            HistogramSampler(self.env,self.cfg,min(self.cfg.rollout_workers,episodes_per_iteration)))
        with sampler:
            for _ in range(iterations):
                started=time.perf_counter()
                self.progress('sampling',iteration=self.iteration+1)
                trajs=sampler.collect(self.actor,[seed+self.iteration*episodes_per_iteration+i for i in range(episodes_per_iteration)])
                sampled=time.perf_counter(); self.progress('update',iteration=self.iteration+1,sampling_seconds=sampled-started)
                report=self.update(trajs); del trajs
                self.iteration+=1; report.update(iteration=self.iteration,sampling_seconds=sampled-started,update_seconds=time.perf_counter()-sampled)
                if batched: report.update(sampler.last_stats)
                reports.append(report)
                if on_iteration and on_iteration(report) is False: break
        return reports
    def progress(self,stage,**kwargs):
        callback=getattr(self,'progress_callback',None)
        if callback: callback(dict(stage=stage,**kwargs))
