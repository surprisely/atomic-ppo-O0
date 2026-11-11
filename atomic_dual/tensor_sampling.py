"""Device-resident environment + policy, with optional timestamp CUDA graphs."""
import copy
import gc
import time
import numpy as np
import torch

from .tensor_environment import TensorFleetEnv
from .event_tables import EventCapacityError


class _ResidentRollouts:
    def __init__(self,template,seeds,actor,device,travel_rng='step'):
        self.env=TensorFleetEnv(template,seeds,device,travel_rng=travel_rng)
        e=self.env;d=e.device
        self.actor=copy.deepcopy(actor).to(d).eval()
        self.observations=torch.empty((e.K,e.H+1,e.D),dtype=torch.float32,device=d)
        self.masks=torch.empty((e.K,e.H,e.A),dtype=torch.bool,device=d)
        self.actions=torch.empty((e.K,e.H),dtype=torch.long,device=d)
        self.logps=torch.empty((e.K,e.H),dtype=torch.float32,device=d)
        self.rewards=torch.empty((e.K,e.H),dtype=torch.float32,device=d)
        self.graph=None

    def reset(self):
        self.env.reset_state()
        self.observations[:,-1].zero_()

    @torch.no_grad()
    def timestamp(self):
        e=self.env;e.begin_timestamp()
        for j in range(e.N):
            index=(e.t*e.N+j).reshape(1)
            states=e.observe(j);masks=e.action_mask(j)
            logits=self.actor(states).masked_fill(~masks,-torch.inf)
            logps=logits-torch.logsumexp(logits,dim=-1,keepdim=True)
            cdf=logps.exp().cumsum(-1);cdf=cdf/cdf[:,-1:]
            uniforms=e.policy_tape.index_select(1,index)
            actions=(uniforms>=cdf).sum(-1)
            chosen_logps=logps.gather(1,actions[:,None]).squeeze(1)
            reward=e.step(actions,j)
            self.observations.index_copy_(1,index,states[:,None,:])
            self.masks.index_copy_(1,index,masks[:,None,:])
            self.actions.index_copy_(1,index,actions[:,None])
            self.logps.index_copy_(1,index,chosen_logps[:,None])
            self.rewards.index_copy_(1,index,reward.float()[:,None])
        e.finish_timestamp()

    def capture(self):
        # Warm-up must be on a side stream; each warm-up resets mutable state.
        self.reset();self.env.prepare_event_timestamp()
        stream=torch.cuda.Stream(device=self.env.device)
        stream.wait_stream(torch.cuda.current_stream(self.env.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.reset();self.timestamp()
        torch.cuda.current_stream(self.env.device).wait_stream(stream)
        self.reset()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):self.timestamp()
        self.graph=graph
        self.reset()

    def export(self):
        def array(t):
            a=t.detach().cpu().numpy()
            # CPU tensors would otherwise alias storage reused next iteration.
            return a.copy() if self.env.device.type=='cpu' else a
        observations=array(self.observations);masks=array(self.masks)
        actions=array(self.actions);logps=array(self.logps);rewards=array(self.rewards)
        profit=array(self.env.profit);served=array(self.env.served.sum(-1));rejected=array(self.env.rejected)
        dones=np.zeros(self.env.H,dtype=np.bool_);dones[-1]=True
        return [dict(states=observations[k,:-1],next_states=observations[k,1:],
            masks=masks[k],actions=actions[k],logps=logps[k],rewards=rewards[k],dones=dones.copy(),
            total_profit=float(profit[k]),served=int(served[k]),rejected=int(rejected[k]))
            for k in range(self.env.K)]


class TensorEpisodeSampler:
    def __init__(self,env,config):
        self.template=env;self.config=config;self.device=torch.device(config.rollout_device)
        if self.device.type=='cuda' and not torch.cuda.is_available():
            raise ValueError('Tensor CUDA environment requested but CUDA is unavailable')
        self.cache={}

    def __enter__(self):return self

    @torch.no_grad()
    def collect(self,actor,seeds):
        stats=dict(tensor_prepare_seconds=0.,tensor_capture_seconds=0.,tensor_replay_seconds=0.,
                   tensor_export_seconds=0.,tensor_atomic_host_readbacks=0,
                   tensor_event_prefetch_seconds=0.,tensor_event_boundary_syncs=0,tensor_event_capacity_recaptures=0,
                   tensor_cuda_graph=self.device.type=='cuda' and self.config.rollout_tensor_graph,
                   max_inference_batch=0,actor_forward_calls=0)
        trajectories=[]
        for start in range(0,len(seeds),self.config.rollout_batch_size):
            group=seeds[start:start+self.config.rollout_batch_size];k=len(group)
            now=time.perf_counter()
            first=k not in self.cache
            if first:
                buffer=_ResidentRollouts(self.template,group,actor,self.device,self.config.rollout_travel_rng)
                self.cache[k]=buffer
            else:
                buffer=self.cache[k]
                try:
                    buffer.env.load_inputs(self.template,group)
                except EventCapacityError:
                    del self.cache[k];del buffer;gc.collect()
                    buffer=_ResidentRollouts(self.template,group,actor,self.device,self.config.rollout_travel_rng)
                    self.cache[k]=buffer;first=True
                    stats['tensor_event_capacity_recaptures']+=1
                buffer.actor.load_state_dict(actor.state_dict())
            if self.device.type=='cuda':torch.cuda.synchronize(self.device)
            stats['tensor_prepare_seconds']+=time.perf_counter()-now
            if first and stats['tensor_cuda_graph']:
                now=time.perf_counter();buffer.capture();torch.cuda.synchronize(self.device)
                stats['tensor_capture_seconds']+=time.perf_counter()-now
            buffer.reset()
            if buffer.env.event_tables is not None:
                # Boundary prefetch retains exact NumPy/libm math. No atomic-step
                # synchronization; the next table is ready before all N decisions.
                for _ in range(buffer.env.T):
                    now=time.perf_counter();buffer.env.prepare_event_timestamp()
                    if self.device.type=='cuda':torch.cuda.synchronize(self.device)
                    stats['tensor_event_prefetch_seconds']+=time.perf_counter()-now
                    stats['tensor_event_boundary_syncs']+=1
                    now=time.perf_counter()
                    if buffer.graph is not None:buffer.graph.replay()
                    else:buffer.timestamp()
                    if self.device.type=='cuda':torch.cuda.synchronize(self.device)
                    stats['tensor_replay_seconds']+=time.perf_counter()-now
            else:
                now=time.perf_counter()
                for _ in range(buffer.env.T):
                    if buffer.graph is not None:buffer.graph.replay()
                    else:buffer.timestamp()
                if self.device.type=='cuda':torch.cuda.synchronize(self.device)
                stats['tensor_replay_seconds']+=time.perf_counter()-now
            now=time.perf_counter();trajectories.extend(buffer.export())
            stats['tensor_export_seconds']+=time.perf_counter()-now
            stats['max_inference_batch']=max(stats['max_inference_batch'],k)
            stats['actor_forward_calls']+=buffer.env.H
        self.last_stats=stats
        return trajectories

    def __exit__(self,*args):
        self.cache.clear()
