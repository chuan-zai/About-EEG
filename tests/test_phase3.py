import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from eeg_transfer.model_v2 import EEGModelV2
from eeg_transfer.models import build_model
from eeg_transfer.masking import make_batch, availability, masked_mse
from eeg_transfer.data import synthetic, split_sessions, Standardizer, CHANNELS, EIGHT, load_npz
from eeg_transfer.reconstruction import ReconstructionAccumulator, evaluate
from eeg_transfer.plans import evaluation_plan, evaluation_coverage, digest
from eeg_transfer.regions import compare_to_a0
from eeg_transfer.validation import train_only_baselines
from eeg_transfer.statistics import holm, subject_inference
from eeg_transfer.bnci import sha256
from eeg_transfer.experiment import prepare, run

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)

@pytest.mark.parametrize('kind',[8,22])
@pytest.mark.parametrize('time',[32,250])
def test_v2_safety_shapes_and_gradients(kind,time):
    torch.manual_seed(2)
    model=EEGModelV2(time=time,dim=32)
    x=torch.randn(2,22,time,requires_grad=True)
    batch=make_batch(x,availability(kind,2))
    args={k:v for k,v in batch.items() if k!='target'}
    a=model(**args)
    assert a.shape==x.shape and torch.isfinite(a).all()
    forbidden=~batch['availability_mask'][...,None].expand_as(x)|batch['reconstruction_mask']
    altered=x.detach().clone(); altered[forbidden]=float('nan')
    b=model(**{**args,'signal':altered})
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    loss=masked_mse(a,batch['target'],batch['availability_mask'],batch['reconstruction_mask']); loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert torch.count_nonzero(x.grad[~batch['availability_mask'][...,None].expand_as(x)])==0
    # Target signal gradient must have NO input path (target used only for loss).
    model.zero_grad(); z=x.detach().clone().requires_grad_()
    output=model(**{**args,'signal':z}); output.square().sum().backward()
    assert torch.count_nonzero(z.grad[forbidden])==0


def test_v1_unchanged_and_v2_temporal_information():
    expected=(ROOT/'reports/phase3/v1-original.sha256').read_text().split()[0]
    assert sha256(ROOT/'eeg_transfer/model.py')==expected
    torch.manual_seed(3); model=EEGModelV2(time=50,dim=32)
    batch=make_batch(torch.randn(1,22,50),availability(22,1))
    args={k:v for k,v in batch.items() if k!='target'}
    original=model(**args)
    changed=model(**{**args,'signal':batch['signal'].flip(-1)})
    assert not torch.allclose(original,changed,atol=1e-7)
    with pytest.raises(ValueError):
        build_model(dict(model_version='unknown',time=50,dim=32,layers=2))


def test_regions_pool_correctly_not_average_rmse():
    p=torch.zeros(1,22,10); t=p.clone(); mask=torch.zeros_like(p,dtype=torch.bool)
    i=CHANNELS.index('Fz'); j=CHANNELS.index('FC1')
    p[:,i,:]=2; mask[:,i,:]=True; p[:,j,:2]=4; mask[:,j,:2]=True
    acc=ReconstructionAccumulator(); acc.update(p,t,mask,np.ones(22))
    r=acc.result()
    assert r['regions']['inside8']['rmse_V']==2
    assert r['regions']['outside14']['rmse_V']==4
    assert r['rmse_V']==pytest.approx(np.sqrt(72/12))
    assert sum(v['target_elements'] for v in r['regions'].values())==r['target_elements']
    base={'plan_sha256':'same','protocol':'random25','overall':r}
    damaged=copy.deepcopy(base)
    damaged['overall']['regions']['outside14']['rmse_V']=4.4
    c=compare_to_a0(damaged,base)
    assert c['outside_minus_inside_relative_change']==pytest.approx(.1)
    damaged['plan_sha256']='different'
    with pytest.raises(ValueError):
        compare_to_a0(damaged,base)


def test_coverage_and_baselines_train_only():
    train,_,test=split_sessions(synthetic(n=40,time=32))
    plan=evaluation_plan(test,32,32)
    coverage=evaluation_coverage(plan,minimum=0)
    assert sum(coverage['target_windows_including_mask_seeds'].values())==len(plan['events'])*6
    broken=copy.deepcopy(plan); broken['events']=[broken['events'][0]]
    with pytest.raises(ValueError):
        evaluation_coverage(broken)
    norm=Standardizer().fit(train)
    baselines=train_only_baselines(train,norm,32,32)
    physical=baselines['zero_volts'].waveform.numpy()*norm.std+norm.mean
    np.testing.assert_allclose(physical,0,atol=1e-10)
    expected=norm.transform(train.x.mean(0,keepdims=True),np.ones(22,dtype=bool))
    np.testing.assert_allclose(baselines['train_mean_waveform'].waveform.numpy(),expected,atol=1e-6)


def test_statistics_subject_unit_and_multiplicity():
    r=subject_inference(np.ones(9),repeats=1000)
    assert r['subjects']==9 and r['mean_delta']==1
    assert r['ci95_percentile']==[1.,1.]
    assert r['p_signflip_two_sided']==pytest.approx(2/512)
    assert holm([.01,.03,.04])==pytest.approx([.03,.06,.06])
    assert subject_inference(np.zeros(9),repeats=1000)['p_signflip_two_sided']==1


@pytest.mark.parametrize('subject',list(range(1,10)))
def test_all_subject_audits_units_splits_and_train_norm(subject):
    path=ROOT/f'reports/phase3/data/S{subject:02d}.json'
    if not path.exists() or not (ROOT/f'data/processed/S{subject:02d}.npz').exists():
        pytest.skip(f'S{subject:02d} acquisition incomplete')
    report=json.loads(path.read_text()); data=load_npz(ROOT/f'data/processed/S{subject:02d}.npz')
    assert data.channels==CHANNELS and data.sfreq==250 and data.x.shape[1:]==(22,1000)
    assert sha256(ROOT/f'data/processed/S{subject:02d}.npz')==report['processed']['sha256']
    for s in report['sessions']:
        assert sum(r['trials'] for r in s['runs'])==288
        assert s['source_unit']=='uV' and s['output_unit']=='V' and s['scale']==1e-6
        assert s['nonfinite_epoch_values']==0
        assert sum(r['accepted']+len(r['dropped']) for r in s['runs'])==288
    train,val,test=split_sessions(data)
    assert set(train.session)=={'T'} and set(val.session)=={'T'} and set(test.session)=={'E'}
    assert not set(train.trial)&set(val.trial)
    norm=Standardizer().fit(train)
    with np.load(ROOT/f'data/processed/S{subject:02d}-train-normalization.npz') as z:
        np.testing.assert_array_equal(norm.mean,z['mean']); np.testing.assert_array_equal(norm.std,z['std'])
    # Cross-check actual first epoch against authoritative MAT values, not just metadata.
    from scipy.io import loadmat
    raw=loadmat(ROOT/f'data/raw/A{subject:02d}T.mat',simplify_cells=True)['data']
    meta=data.metadata[0]; start=meta['cue_sample_0based']
    expected=raw[meta['run']-1]['X'][start:start+1000,:22].T*1e-6
    np.testing.assert_allclose(data.x[0],expected,rtol=2e-7,atol=1e-12)


@pytest.mark.skipif(not (ROOT/'data/processed/S01.npz').exists(),reason='S01 integration data not prepared')
def test_compressed_v2_runner_optimizer_and_no_test(tmp_path):
    real=ROOT/'data/processed/S01.npz'
    cfg=dict(data_path=str(real),seed=42,split_seed=42,device='cpu',time=32,stride=500,dim=32,layers=2,model_version='v2',patch_size=5,lr=.001,weight_decay=.01,batch_size=2,steps=4,evaluation_seeds=[11,22],evaluation_trial_limit=1,evaluation_batch_size=2,compress_plans=True)
    package=prepare(cfg,tmp_path/'plans')
    m,metrics=run(tmp_path/'plans/plans.json.gz','A2',tmp_path/'run')
    assert all(not k.startswith('test') for k in metrics)
    status=json.loads((tmp_path/'run/status.json').read_text())
    assert status['state']=='complete' and set(status['optimizer_steps'])=={4}
    rows=[json.loads(line) for line in (tmp_path/'run/train.jsonl').read_text().splitlines()]
    assert [r['optimizer_step'] for r in rows]==[1,2,3,4]
    assert all(r['learning_rate']==.001 for r in rows)
    assert m['model_version']=='v2'
    with pytest.raises(FileExistsError):
        run(tmp_path/'plans/plans.json.gz','A2',tmp_path/'run')


@pytest.mark.skipif(not (ROOT/'data/processed/S01.npz').exists(),reason='S01 integration data not prepared')
def test_formal_protocol_lock_rejects_changes_before_training(tmp_path):
    cfg=dict(data_path=str(ROOT/'data/processed/S01.npz'),seed=42,split_seed=42,device='cpu',time=32,stride=500,dim=32,layers=2,model_version='v2',patch_size=5,lr=.001,weight_decay=.01,batch_size=2,steps=4,evaluation_seeds=[11,22],evaluation_trial_limit=1,evaluation_batch_size=2,protocol_lock=str(tmp_path/'lock.json'))
    lock={'ready_for_formal_run':False,'architecture':{'model_version':'v2'},'fixed_hyperparameters':{'lr':.001}}
    Path(cfg['protocol_lock']).write_text(json.dumps(lock))
    prepare(cfg,tmp_path/'plans')
    with pytest.raises(ValueError,match='not locked'):
        run(tmp_path/'plans/plans.json','A2',tmp_path/'run')
    lock['ready_for_formal_run']=True; lock['fixed_hyperparameters']['lr']=.002
    Path(cfg['protocol_lock']).write_text(json.dumps(lock))
    with pytest.raises(ValueError,match='not locked'):
        run(tmp_path/'plans/plans.json','A2',tmp_path/'run')
    assert not (tmp_path/'run').exists()


def test_summary_complete_pairing_and_no_incomplete_inference(tmp_path):
    from eeg_transfer.summarize import summarize
    subjects=list(range(1,10)); seeds=[42,43,44]; groups=['A0','A1','A2','A3']
    runs=[]
    for subject in subjects:
        for seed in seeds:
            for group in groups:
                folder=tmp_path/f'{subject}-{seed}-{group}'; folder.mkdir()
                descriptor={'id':folder.name,'subject':subject,'seed':seed,'group':group,'output':str(folder),'data_sha256':f'data{subject}','plans_sha256':'package'}
                runs.append(descriptor)
                manifest={'group':group,'config':{'seed':seed},'data_sha256':f'data{subject}','package_sha256':'package','initial_weights_sha256':f'initial{seed}'}
                (folder/'manifest.json').write_text(json.dumps(manifest))
                (folder/'status.json').write_text(json.dumps({'state':'complete'}))
                factor={'A0':1.,'A1':1.1,'A2':1.2,'A3':1.3}[group]
                score={'plan_sha256':f'plan{subject}','protocol':'random25','overall':{'rmse_V':factor*subject*1e-6,'regions':{'inside8':{'target_elements':80,'rmse_V':subject*1e-6},'outside14':{'target_elements':140,'rmse_V':factor*subject*1e-6}}}}
                outside=copy.deepcopy(score); outside['protocol']='outside14'
                (folder/'evaluation.json').write_text(json.dumps({'test_random25':score,'test_outside14':outside}))
    suite={'subjects':subjects,'seeds':seeds,'groups':groups,'runs':runs,'evaluation_hashes':{str(s):{'test_random25':f'plan{s}'} for s in subjects}}
    path=tmp_path/'manifest.json'; path.write_text(json.dumps(suite))
    result=summarize(path,tmp_path/'summary.json')
    assert result['complete'] and result['statistics']['primary_3_comparisons']['A1']['subjects']==9
    assert result['statistics']['primary_3_comparisons']['A1']['mean_delta']==pytest.approx(.5e-6)
    # Fixtures only: these invented metrics verify arithmetic and are not EEG results.
    Path(runs[-1]['output'],'status.json').unlink()
    incomplete=summarize(path,tmp_path/'incomplete.json')
    assert not incomplete['complete'] and incomplete['statistics'] is None


@pytest.mark.skipif(not (ROOT/'data/plans/rq1-v2/S01/42/plans.json.gz').exists(),reason='Formal integration plans not prepared')
def test_full_formal_suite_plans_integrity_and_pairing():
    import gzip
    path=ROOT/'reports/phase3/rq1-manifest.json'
    assert path.exists(), 'Generate suite plans before final phase-three validation'
    suite=json.loads(path.read_text())
    assert len(suite['runs'])==270 and len({r['id'] for r in suite['runs']})==270
    assert suite['subjects']==list(range(1,10)) and suite['seeds']==[42,43,44]
    splits={}
    for audit in suite['plan_audits']:
        file=ROOT/f"data/plans/rq1-v2/S{audit['subject']:02d}/{audit['seed']}/plans.json.gz"
        assert sha256(file)==audit['plans_sha256']
        with gzip.open(file,'rt') as f:
            package=json.load(f)
        config=package['config']
        assert config['model_version']=='v2' and config['dim']==64
        current=digest(package['splits'])
        assert splits.setdefault(audit['subject'],current)==current
        for name,plan in package['evaluation'].items():
            assert digest(plan)==suite['evaluation_hashes'][str(audit['subject'])][name]
            if plan['protocol']=='random25':
                coverage=evaluation_coverage(plan,minimum=100 if name.startswith('test') else 20)
                assert min(coverage['target_windows_including_mask_seeds'].values())>=coverage['minimum_required']
        g=package['training']['groups']; a0=g['A0']
        for group in ('A1','A2','A3'):
            high=[e for e in g[group] if e['montage']==22]
            assert high==g[group+'_exposure']==a0[:len(high)]
            assert [e for e in g[group+'_extra'] if e['montage']==22]==a0


@pytest.mark.skipif(not (ROOT/'results/v2-validation/gate.json').exists(),reason='Learning gate artifacts not present')
def test_learning_gate_evidence_is_T_only_and_predeclared():
    root=ROOT/'results/v2-validation'
    gate=json.loads((root/'gate.json').read_text())
    assert gate['config_sha256']==sha256(ROOT/'configs/v2-validation.json')
    assert len(gate['records'])==6
    plan=json.loads((root/'validation-plan.json').read_text())
    assert all(e['session']=='T' for e in plan['events'])
    assert {r['seed'] for r in gate['records']}=={42,43,44}
    for r in gate['records']:
        rows=[json.loads(v) for v in (root/f"{r['version']}-{r['seed']}"/'train.jsonl').read_text().splitlines()]
        assert len(rows)==120 and all(np.isfinite(v['loss']) and np.isfinite(v['gradient_norm_l2']) for v in rows)
