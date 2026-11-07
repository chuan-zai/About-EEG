import numpy as np
import pytest
import torch
from eeg_transfer.data import CHANNELS, EIGHT, synthetic, split_sessions, Standardizer, load_npz
from eeg_transfer.masking import availability, make_batch, masked_mse
from eeg_transfer.model import EEGModel
from eeg_transfer.train import run, seed_all, update_schedule
from eeg_transfer.evaluate import classification_probe

torch.set_num_threads(1)

def test_split_and_normalization(tmp_path):
    d = synthetic()
    a,b,c = split_sessions(d)
    keys = lambda z: set(zip(z.subject,z.session,z.trial))
    assert not keys(a)&keys(b) and not keys(a)&keys(c) and not keys(b)&keys(c)
    norm = Standardizer().fit(a)
    assert np.allclose(norm.mean,a.x.mean((0,2),keepdims=True))
    assert set(a.session)=={'T'} and set(c.session)=={'E'}
    path = tmp_path/'data.npz'
    np.savez(path,x=d.x,y=d.y,subject=d.subject,session=d.session,trial=d.trial,sfreq=d.sfreq,channels=d.channels)
    assert np.array_equal(load_npz(path).x,d.x)

@pytest.mark.parametrize('kind',[8,22])
def test_target_and_missing_invariance(kind):
    seed_all(1)
    model = EEGModel(time=32,dim=32).eval()
    x = torch.randn(2,22,32)
    av = availability(kind,2)
    batch = make_batch(x,av)
    args = {k:v for k,v in batch.items() if k!='target'}
    pred = model(**args)
    altered = x.clone()
    forbidden = ~av[...,None].expand_as(x) | batch['reconstruction_mask']
    altered[forbidden] = float('nan')
    other = model(**{**args,'signal':altered})
    torch.testing.assert_close(pred,other,rtol=0,atol=0)
    assert torch.isfinite(pred).all()
    changed_target = batch['target'].clone()
    changed_target[~av[...,None].expand_as(x)] = float('nan')
    loss = masked_mse(pred,batch['target'],av,batch['reconstruction_mask'])
    torch.testing.assert_close(loss,masked_mse(other,changed_target,av,batch['reconstruction_mask']))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_hidden_preprocessing_and_supervision():
    d = synthetic()
    norm = Standardizer().fit(d)
    av = availability(8,2)
    a = d.x[:2].copy(); b = a.copy()
    b[:,~av[0].numpy()] = np.nan
    np.testing.assert_array_equal(norm.transform(a,av[0].numpy()),norm.transform(b,av[0].numpy()))
    x = torch.tensor(a,requires_grad=True)
    batch = make_batch(x,av)
    assert not (batch['reconstruction_mask'] & ~av[...,None]).any()
    model = EEGModel(time=64,dim=32)
    pred = model(**{k:v for k,v in batch.items() if k!='target'})
    masked_mse(pred,batch['target'],av,batch['reconstruction_mask']).backward()
    assert torch.count_nonzero(x.grad[~av[...,None].expand_as(x)])==0
    assert tuple(CHANNELS[i] for i in av[0].nonzero().flatten())==EIGHT


def test_loss_exact_and_empty():
    pred = torch.zeros(1,22,4); target = torch.ones_like(pred)
    av = availability(8,1); mask = av[...,None].expand_as(pred)
    assert masked_mse(pred,target,av,mask)==1
    with pytest.raises(ValueError):
        masked_mse(pred,target,av,torch.zeros_like(mask))


def test_controls():
    for p in (1,.75,.5,.25):
        mixed = update_schedule(p,100,42)
        assert mixed.count(22)==round(p*100)
        assert len(update_schedule(p,100,42,'exposure_matched'))==mixed.count(22)
        assert update_schedule(p,100,42,'extra_8').count(22)==100


def test_smoke_reproducible(tmp_path):
    cfg = dict(data='synthetic',seed=42,device='cpu',time=32,dim=32,layers=2,lr=.001,batch_size=2,steps=2,p22=.5)
    a = run(cfg,tmp_path/'a'); b = run(cfg,tmp_path/'b')
    assert a==b and len(a)==2
    assert (tmp_path/'a/checkpoint.pt').exists()


def test_probe():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40,8)); y = np.arange(40)%4
    scores = classification_probe(x,y,x[:8],y[:8],x[8:16],y[8:16])
    assert 0 <= scores['test']['macro_f1'] <= 1


def test_provenance_and_real_loaders(tmp_path):
    from eeg_transfer.masking import trial_batch
    from eeg_transfer.data import load_mat, load_gdf
    from scipy.io import savemat
    import mne
    d = synthetic()
    b = trial_batch(d,[0,1],Standardizer().fit(d),8)
    assert b['trial_id']==['0','1'] and b['session_id']==['T','T']
    path = tmp_path/'canonical.mat'
    savemat(path,dict(x=d.x,y=d.y,subject=d.subject.astype(object),session=d.session.astype(object),trial=d.trial.astype(object),sfreq=d.sfreq,channels=np.array(CHANNELS,dtype=object)))
    np.testing.assert_array_equal(load_mat(path).x,d.x)
    assert callable(load_gdf) and mne.__version__
