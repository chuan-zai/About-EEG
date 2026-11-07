import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from eeg_transfer.data import CHANNELS, EIGHT, Standardizer, synthetic, split_sessions, load_npz, windows
from eeg_transfer.bnci import load_bnci_mat, sha256
from eeg_transfer.plans import training_plans, evaluation_plan, digest, budget, trial_keys
from eeg_transfer.reconstruction import ReconstructionAccumulator, planned_batch, evaluate
from eeg_transfer.model import EEGModel
from eeg_transfer.masking import masked_mse
from eeg_transfer.experiment import prepare, run

ROOT=Path(__file__).resolve().parents[1]
REAL=ROOT/'data/processed/S01.npz'


def test_trial_windows_and_fit_isolation():
    train,val,test=split_sessions(synthetic(time=128))
    norm=Standardizer().fit(train)
    before=(norm.mean.copy(),norm.std.copy())
    val.x[:]=1e20; test.x[:]=-1e20
    other=Standardizer().fit(train)
    np.testing.assert_array_equal(before[0],other.mean)
    np.testing.assert_array_equal(before[1],other.std)
    key_sets=[]
    for d in (train,val,test):
        keys={(str(d.subject[i]),str(d.session[i]),str(d.trial[i])) for i,_,_ in windows(d,32,16)}
        key_sets.append(keys)
    assert not key_sets[0]&key_sets[1] and not key_sets[0]&key_sets[2] and not key_sets[1]&key_sets[2]


def test_paired_training_and_fixed_evaluation():
    train,val,test=split_sessions(synthetic())
    plan=training_plans(train,8,2,32,32,42)
    assert plan==training_plans(train,8,2,32,32,42)
    a0=plan['groups']['A0']
    for name in ('A1','A2','A3'):
        mixed=plan['groups'][name]
        high=[e for e in mixed if e['montage']==22]
        assert high==a0[:len(high)]==plan['groups'][name+'_exposure']
        extra_high=[e for e in plan['groups'][name+'_extra'] if e['montage']==22]
        assert extra_high==a0
        assert len(mixed)==8
        assert budget(mixed,32)['target_elements']==sum(len(s['targets'])*32 for e in mixed for s in e['samples'])
        for event in mixed:
            for sample in event['samples']:
                if event['montage']==8:
                    assert all(CHANNELS[c] in EIGHT for c in sample['targets'])
    plans=[evaluation_plan(test,32,32) for _ in plan['groups']]
    assert len({digest(p) for p in plans})==1
    assert len({e['seed'] for e in plans[0]['events']})==3


def test_metrics_exact_pooled_and_pearson():
    p=torch.zeros(1,22,4); t=p.clone(); m=torch.zeros_like(p,dtype=torch.bool)
    p[0,0]=torch.tensor([1.,2.,3.,4.]); t[0,0]=2*p[0,0]; m[0,0]=True
    p[0,1,:2]=torch.tensor([1.,3.]); t[0,1,:2]=torch.tensor([3.,1.]); m[0,1,:2]=True
    acc=ReconstructionAccumulator(); acc.update(p,t,m,np.ones(22)*2)
    r=acc.result()
    assert r['target_elements']==6
    assert r['rmse_standardized']==pytest.approx(np.sqrt(38/6))
    assert r['rmse_V']==pytest.approx(2*np.sqrt(38/6))
    assert r['electrodes']['Fz']['pearson']==pytest.approx(1)
    assert r['electrodes']['FC3']['pearson']==pytest.approx(-1)
    assert r['electrodes']['FC1']['pearson'] is None


def test_outside14_evaluation_no_gradient_and_no_mutation():
    train,_,test=split_sessions(synthetic(n=12,time=32))
    norm=Standardizer().fit(train); model=EEGModel(time=32,dim=32)
    plan=evaluation_plan(test,32,32,[0],'outside14')
    batch=planned_batch(test,norm,plan['events'][:2],22,32)
    assert batch['reconstruction_mask'].any(-1).sum(-1).tolist()==[14,14]
    visible=batch['availability_mask'] & ~batch['reconstruction_mask'].any(-1)
    assert {CHANNELS[c] for c in visible[0].nonzero().flatten()}==set(EIGHT)
    before={k:v.clone() for k,v in model.state_dict().items()}
    for param in model.parameters():
        param.grad=torch.ones_like(param)
    metrics=evaluate(model,test,norm,plan,batch_size=2)
    assert metrics['overall']['target_elements']==len(test.x)*14*32
    assert model.training
    for name,v in model.state_dict().items():
        torch.testing.assert_close(v,before[name],rtol=0,atol=0)
    assert all(torch.equal(p.grad,torch.ones_like(p)) for p in model.parameters())
    # A training montage cannot accept this independent evaluation supervision.
    with pytest.raises(ValueError,match='Unavailable'):
        planned_batch(test,norm,plan['events'][:2],8,32)


@pytest.mark.skipif(not REAL.exists(),reason='Run BNCI preparation to enable real-data integration test')
def test_real_mapping_units_npz_and_source_metadata():
    from scipy.io import loadmat
    data,report=load_bnci_mat(ROOT/'data/raw/A01T.mat',1,'T')
    raw=loadmat(ROOT/'data/raw/A01T.mat',simplify_cells=True)['data']
    first=data.metadata[0]; r=raw[first['run']-1]; start=first['cue_sample_0based']
    assert data.channels==CHANNELS and data.sfreq==250 and data.x.shape[1:]==(22,1000)
    np.testing.assert_allclose(data.x[0],r['X'][start:start+1000,:22].T*1e-6,rtol=2e-7,atol=1e-12)
    assert start==int(r['trial'][first['trial_in_run']-1])-1+500
    assert report['nonfinite_epoch_values']==0
    assert all(m['artifact']==0 for m in data.metadata)
    combined=load_npz(REAL)
    assert combined.metadata[0]==first
    assert set(combined.session)=={'T','E'}
    parts=split_sessions(combined)
    sets=[{(str(d.session[i]),str(d.trial[i])) for i,_,_ in windows(d,250,250)} for d in parts]
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
    np.testing.assert_array_equal(combined.x[:len(data.x)],data.x)
    audit=json.loads((ROOT/'reports/data-audit-S01.json').read_text())
    assert audit['files'][0]['sha256']==sha256(ROOT/'data/raw/A01T.mat')


@pytest.mark.skipif(not REAL.exists(),reason='Real data not prepared')
def test_real_8ch_input_and_loss_invariance():
    train,_,_=split_sessions(load_npz(REAL)); norm=Standardizer().fit(train)
    event=next(e for e in training_plans(train,4,2,32,32,42)['groups']['A2'] if e['montage']==8)
    batch=planned_batch(train,norm,event['samples'],8,32)
    changed=train.take(np.arange(len(train.x)))
    hidden=[i for i,c in enumerate(CHANNELS) if c not in EIGHT]
    changed.x[:,hidden]=np.nan
    other=planned_batch(changed,norm,event['samples'],8,32)
    for k in batch:
        torch.testing.assert_close(batch[k],other[k],rtol=0,atol=0)
    model=EEGModel(time=32,dim=32)
    a=model(**{k:v for k,v in batch.items() if k!='target'})
    # Force forbidden target values into input; model itself must clear them.
    leak=copy.deepcopy(batch)
    leak['signal'][batch['reconstruction_mask']]=float('nan')
    b=model(**{k:v for k,v in leak.items() if k!='target'})
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(masked_mse(a,batch['target'],batch['availability_mask'],batch['reconstruction_mask']),masked_mse(b,other['target'],other['availability_mask'],other['reconstruction_mask']))


@pytest.mark.skipif(not REAL.exists(),reason='Real data not prepared')
def test_real_run_manifest_replay_and_evaluation_masks(tmp_path):
    cfg=dict(data_path=str(REAL),seed=42,split_seed=42,device='cpu',time=32,stride=500,dim=32,layers=2,lr=.001,weight_decay=.01,batch_size=2,steps=4,evaluation_seeds=[11,22],evaluation_trial_limit=1,evaluation_batch_size=2)
    package=prepare(cfg,tmp_path/'plans')
    manifests=[]; logs=[]
    for name in ('one','two'):
        manifest,metrics=run(tmp_path/'plans/plans.json','A2',tmp_path/name,True)
        manifests.append(manifest)
        rows=[json.loads(line) for line in (tmp_path/name/'train.jsonl').read_text().splitlines()]
        assert all(row['gradient_norm_l2']>0 for row in rows)
        assert sum(r['target_elements'] for r in rows)==manifest['budget']['target_elements']
        logs.append([{k:v for k,v in row.items() if k!='seconds'} for row in rows])
        assert metrics['test_random25']['plan_sha256']==digest(package['evaluation']['test_random25'])
        assert json.loads((tmp_path/name/'plans.json').read_text())==package
    assert logs[0]==logs[1]
    assert manifests[0]==manifests[1]
