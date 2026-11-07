"""Downstream labels enter only this module. Never tune against test metrics."""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from .masking import masked_mse


def reconstruction_metrics(pred, target, available, recon):
    rmse = masked_mse(pred,target,available,recon).sqrt().item()
    correlations = []
    for c in range(pred.shape[1]):
        m = recon[:,c] & available[:,c,None]
        a,b = pred[:,c][m].detach().cpu().numpy(),target[:,c][m].detach().cpu().numpy()
        correlations.append(float(np.corrcoef(a,b)[0,1]) if len(a)>1 and a.std()>0 and b.std()>0 else None)
    return {'rmse':rmse,'electrode_correlation':correlations}


def frozen_features(model, x, available):
    model.eval()
    with torch.no_grad():
        ids = torch.arange(22,device=x.device).expand(len(x),-1)
        return model.representation(x,ids,available).cpu().numpy()


def classification_probe(train_features, train_y, val_features, val_y, test_features, test_y):
    """Fixed L2 linear probe: C=1, no search. Validation reported, never fitted."""
    if set(np.unique(train_y)) != {0,1,2,3}:
        raise ValueError('Probe training requires all four classes')
    probe = LogisticRegression(C=1.,max_iter=1000,random_state=42).fit(train_features,train_y)
    def score(x,y):
        p = probe.predict(x)
        return {'accuracy':float(accuracy_score(y,p)), 'macro_f1':float(f1_score(y,p,labels=[0,1,2,3],average='macro',zero_division=0))}
    return {'validation':score(val_features,val_y),'test':score(test_features,test_y)}


def unseen_channel_probe(*args, **kwargs):
    raise NotImplementedError('Independent paired probe requires train-only probe fitting; missing-channel SSL targets are prohibited')


def gradient_analysis_hook(*args, **kwargs):
    raise NotImplementedError('RQ2 deferred until RQ1 protocol is validated')
