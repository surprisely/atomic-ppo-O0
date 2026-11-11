import copy
import json
import math
from dataclasses import asdict
import numpy as np
import pytest
import torch
from atomic_dual.demo import make_synthetic_env
from atomic_dual.environment import EnvConfig,CarStatus
from atomic_dual.ppo import AverageRewardPPO,PPOConfig,MLP
from atomic_dual.evaluation import make_scenario,evaluate_scenario
from physical_time_gae import physical_advantage
from histogram_eta import HistogramEnv,HistogramSpec,HistogramPPO,HistogramSampler,exact_eta_bound,parameter_hash,bucket_projection_hash
from experiment_l_histogram_eta import adapt_scenario,paired_probe


def base(): return make_synthetic_env(EnvConfig(3,1.,2,1),zones=2,seed=7)
def config(workers=1): return PPOConfig(hidden_dim=8,actor_epochs=1,critic_epochs=1,minibatch_size=4,rollout_workers=workers,rollout_seed_mode='episode')
def trainer(mode='bucket',workers=1): return HistogramPPO(HistogramEnv(base(),HistogramSpec(mode,8 if mode=='bucket' else 40)),config(workers),7)


def test_bucket_histogram_equals_original_observation_and_dynamics():
    tr=trainer(); env=tr.env
    assert type(tr.actor) is type(tr.critic) is MLP
    env.reset(51); reference=copy.deepcopy(env.base); reference._refresh_observation_cache()
    while not env.done:
        np.testing.assert_array_equal(env.observe(),reference.observe())
        a=int(np.flatnonzero(env.action_mask())[0])
        _,r,d,info=env.step(a); _,rr,dd,ii=reference.step(a)
        assert (r,d,info)==(rr,dd,ii)
        if not d:
            old=env.observe(); env._rebuild(); np.testing.assert_array_equal(old,env.observe())


def test_exact_bins_8_12_25_remain_separate_and_count_conserved():
    for mode in ('bucket','exact'):
        env=trainer(mode).env
        for c,eta in zip(env.cars,(8,12,25)):
            c.status=CarStatus.OCCUPIED; c.destination=1; c.eta=eta
        env._refresh_observation_cache(); env._rebuild()
        hist=env.dense_observe()[:env.hist_dim].reshape(2,3,2,-1)
        assert np.isclose(hist.sum(),1)
        if mode=='bucket': assert np.isclose(hist[0,1,1,8],2/3)
        else:
            assert hist[0,1,1,8]>0 and hist[0,1,1,12]>0 and hist[1,1,1,25]>0
            env.cars[0].eta=41
            with pytest.raises(ValueError,match='never silently clipped'): env._rebuild()


def test_old_weights_and_common_initial_function_even_with_large_eta():
    torch.set_num_threads(1)
    original=AverageRewardPPO(base(),config(),7); low=trainer(); exact=trainer('exact')
    assert parameter_hash(original.actor,original.critic)==parameter_hash(low.actor,low.critic)
    assert bucket_projection_hash(exact.env,exact.actor,exact.critic)==parameter_hash(original.actor,original.critic)
    for tr in (low,exact):
        c=tr.env.cars[0]; c.status=CarStatus.OCCUPIED; c.destination=1; c.eta=25
        tr.env._refresh_observation_cache(); tr.env._rebuild()
    with torch.no_grad():
        x=torch.tensor(low.env.observe()); y=torch.tensor(exact.env.observe())
        assert torch.allclose(low.actor(x),exact.actor(y),atol=2e-7,rtol=1e-6)
        assert torch.allclose(low.critic(x),exact.critic(y),atol=2e-7,rtol=1e-6)


def test_sparse_linear_forward_and_gradients_match_dense_old_mlp():
    torch.set_num_threads(1); tr=trainer('exact'); env=tr.env
    sparse=tr.actor; dense=MLP(env.hist_dim+env.tail_dim,env.action_dim,8); dense.load_state_dict(sparse.state_dict())
    xs=[]; ds=[]
    for seed in (6,7,8):
        env.reset(seed); xs.append(env.observe()); ds.append(env.dense_observe())
    a=sparse(torch.tensor(np.stack(xs))); b=dense(torch.tensor(np.stack(ds)))
    assert torch.allclose(a,b,atol=2e-7,rtol=1e-6)
    a.square().sum().backward(); b.square().sum().backward()
    for p,q in zip(sparse.parameters(),dense.parameters()): assert torch.allclose(p.grad,q.grad,atol=2e-7,rtol=1e-5)


def test_eta_bound_covers_largest_implemented_uniform_draw():
    env=base(); env.reset(7); env.travel_time_model='adp_exponential'
    m=float(max(env.trip_minutes.max(),env.relocation_minutes.max(),.1))
    bound=exact_eta_bound(env)
    assert bound>=math.ceil((-math.log1p(-np.nextafter(1.,0.)))*m)
    snapshot=copy.deepcopy(env); snapshot.cars[0].eta=bound+5
    assert exact_eta_bound(env,[snapshot])==bound+5


@pytest.mark.parametrize('mode', ['bucket', 'exact', 'asymmetric'])
def test_parallel_frozen_sampler_and_exact_resume(tmp_path,mode):
    torch.set_num_threads(1)
    spec=HistogramSpec('bucket',8,'marginal','full') if mode=='asymmetric' else trainer(mode).env.representation
    a=HistogramPPO(HistogramEnv(base(),spec),config(2),7)
    with HistogramSampler(a.env,a.cfg,1) as sampler: serial=sampler.collect(a.actor,[9,10])
    with HistogramSampler(a.env,a.cfg,2) as sampler: parallel=sampler.collect(a.actor,[9,10])
    for x,y in zip(serial,parallel):
        for k in x: np.testing.assert_array_equal(x[k],y[k])
    with physical_advantage('minute_gae',a.env.N): a.train(1,2,23)
    cp=tmp_path/'a.pt'; a.save_checkpoint(cp)
    with physical_advantage('minute_gae',a.env.N): first=a.train(1,2,23)
    b=HistogramPPO.from_checkpoint(HistogramEnv(base(),spec),cp)
    with physical_advantage('minute_gae',b.env.N): second=b.train(1,2,23)
    assert first[0]['critic_loss']==second[0]['critic_loss']
    assert parameter_hash(a.actor,a.critic)==parameter_hash(b.actor,b.critic)


def test_validation_and_paired_mc_snapshot_are_compatible():
    torch.set_num_threads(1); tr=trainer('exact'); env=tr.env
    s=make_scenario(env.base,19,'evening','dummy')
    row=evaluate_scenario(env,adapt_scenario(env,s),'ppo_greedy',tr.actor)
    assert row['profit_total']==env.profit
    env.reset(22); c=env.current_car(); env.requests[c.region,1,2]=2
    env._refresh_observation_cache(); env._rebuild(); tr.audit_gain=10.
    r=paired_probe(env,2*env.Z+1,tr,dict(exogenous=10,travel=11,policy=12))
    assert np.isfinite(r['error_td_minus_mc']) and abs(r['float32_residual'])<1e-3


def test_marginal_od_matches_explicit_sums_and_preserves_full_critic():
    env=make_synthetic_env(EnvConfig(3,1.,2,1),zones=9,seed=7)
    full=HistogramPPO(HistogramEnv(copy.deepcopy(env),HistogramSpec('bucket',8)),config(),7)
    asymmetric=HistogramPPO(HistogramEnv(env,HistogramSpec('bucket',8,'marginal','full')),config(),7)
    x=torch.tensor(asymmetric.env.observe())
    start=asymmetric.env.hist_dim; end=start+243
    orders=torch.arange(243,dtype=x.dtype).reshape(9,9,3)/243
    x[start:end]=orders.flatten()
    expected=torch.cat((x[:start],orders.sum(1).flatten(),orders.sum(0).flatten(),x[end:]))
    assert asymmetric.actor.net[0].in_features==575
    assert asymmetric.critic.net[0].in_features==764
    torch.testing.assert_close(asymmetric.actor(x),asymmetric.actor.net(expected))
    torch.testing.assert_close(asymmetric.actor(x.repeat(2,1)),asymmetric.actor.net(expected.repeat(2,1)))
    assert parameter_hash(asymmetric.critic)==parameter_hash(full.critic)


@pytest.mark.parametrize('experiment,names,device,backend',[
    ('eta',('L0','L1'),'cpu','episode'),
    ('od',('O0','O1','O2'),'cpu','episode'),
    ('od',('O2',),'cpu','episode'),
    pytest.param('od',('O0','O1','O2'),'cuda','episode',
                 marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')),
    pytest.param('od',('O0','O1','O2'),'cuda','batched',
                 marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')),
])
def test_histogram_runner_complete_and_resume(tmp_path,monkeypatch,experiment,names,device,backend):
    import experiment_l_histogram_eta as runner
    from atomic_dual.data import sha256
    from atomic_dual.evaluation import save_scenario
    env=base()
    root=tmp_path/'reference'; (root/'validation_scenarios').mkdir(parents=True)
    data=tmp_path/'data'; data.mkdir(); (data/'evening.npz').write_bytes(b'test fixture')
    scenarios=[make_scenario(env,s,'evening','dummy') for s in (11,12)]
    for n,s in enumerate(scenarios): save_scenario(s,root/'validation_scenarios'/f'{n}.npz')
    (root/'manifest.json').write_text(json.dumps(dict(arguments=dict(data=str(data)),environment=asdict(env.cfg),
        ppo=asdict(config()),validation_scenario_ids=[s['scenario_id'] for s in scenarios])))
    bank=tmp_path/'bank'; bank.mkdir(); snapshot=copy.deepcopy(env); snapshot.reset(6)
    c=snapshot.current_car(); snapshot.requests[c.region,1,2]=2
    action=2*snapshot.Z+1; probs=np.zeros((1,env.action_dim)); probs[0,action]=1
    np.savez(bank/'action_probabilities.npz',**{'150':probs})
    snaps=tmp_path/'snapshots'; snaps.mkdir(); file=snaps/'trusted_local_snapshots.pt'
    torch.save(dict(snapshots={0:snapshot}),file)
    (snaps/'manifest.json').write_text(json.dumps(dict(outputs_sha256={file.name:sha256(file)})))
    monkeypatch.setattr(runner,'load_environment',lambda *a,**kw:copy.deepcopy(env))
    seed=9 if experiment=='od' else 42
    validation_algorithm='ppo_sampled' if experiment=='od' else 'ppo_greedy'
    args=runner.parser().parse_args(['--experiment',experiment,'--device',device,'--rollout-backend',backend,
        '--run',str(root),'--bank',str(bank),'--snapshots',str(snaps),
        '--output',str(tmp_path/'first'),'--iterations','1','--episodes','2' if backend=='batched' else '1','--workers','1',
        '--q-replicates','2','--eval-every','1','--hidden-dim','8','--seeds',str(seed),
        '--validation-algorithm',validation_algorithm])
    if len(names)==1:
        args.od_group=names[0]
    runner.run(args)
    assert json.loads((tmp_path/'first'/'manifest.json').read_text())['status']=='complete'
    assert {folder.name for folder in (tmp_path/'first').iterdir() if folder.is_dir()}=={f'{name}_seed{seed}' for name in names}
    expected=(tmp_path/'first'/'comparison.csv').read_text()
    expected_policy_seeds={s['scenario_id']:int(np.random.SeedSequence([seed,s['policy_seed']]).generate_state(1)[0])
        if validation_algorithm=='ppo_sampled' else s['policy_seed'] for s in scenarios}
    for name in names:
        folder=tmp_path/'first'/f'{name}_seed{seed}'
        stages=[json.loads(s)['stage'] for s in (folder/'progress.jsonl').read_text().splitlines()]
        assert stages==['sampling','update']
        validation_rows=[json.loads(s) for s in (folder/'validation_episodes.jsonl').read_text().splitlines()]
        assert len(validation_rows)==2*len(scenarios)
        for row in validation_rows:
            assert row['algorithm']==validation_algorithm
            assert row['policy_seed']==expected_policy_seeds[row['exogenous_scenario_id']]
        if backend=='batched':
            assert json.loads((folder/'train.jsonl').read_text())['max_inference_batch']==2
    args.resume_from=str(tmp_path/'first'); args.output=str(tmp_path/'resumed'); runner.run(args)
    assert (tmp_path/'resumed'/'comparison.csv').read_text()==expected
    if experiment=='od': return

    evaluate=runner.evaluate_scenario
    monkeypatch.setattr(runner,'evaluate_scenario',lambda *a,**kw:dict(evaluate(*a,**kw),profit_total=0.))
    args.resume_from=None; args.output=str(tmp_path/'early'); args.iterations=2
    args.l0_only=True; args.early_stop_patience=1
    runner.run(args)
    assert not (tmp_path/'early'/'L1_seed42').exists()
    selection=json.loads((tmp_path/'early'/'L0_seed42'/'selection.json').read_text())
    assert selection['early_stopped'] and selection['final_iteration']==1 and selection['iteration']==0
    args.resume_from=args.output; args.output=str(tmp_path/'early_resumed')
    runner.run(args)
    assert (tmp_path/'early_resumed'/'comparison.csv').read_text()==(tmp_path/'early'/'comparison.csv').read_text()
