"""Fixed-mask reconstruction evaluation; no fitting and no autograd."""
import numpy as np
import torch
from .data import CHANNELS, EIGHT
from .masking import availability
from .plans import digest, evaluation_coverage


def planned_batch(data, normalizer, samples, montage, length, device='cpu'):
    lookup={(str(s),str(e),str(t)):i for i,(s,e,t) in enumerate(zip(data.subject,data.session,data.trial))}
    arrays=[]
    av=availability(montage,len(samples))
    recon=torch.zeros(len(samples),22,length,dtype=torch.bool)
    for b,sample in enumerate(samples):
        i=lookup[(sample['subject'],sample['session'],sample['trial'])]
        start=sample['start']
        segment=data.x[i,:,start:start+length]
        if segment.shape!=(22,length):
            raise ValueError('Window outside trial')
        targets=sample['targets']
        if not targets or len(targets)!=len(set(targets)) or any(c<0 or c>=22 for c in targets):
            raise ValueError('Invalid targets')
        if not av[b,targets].all():
            raise ValueError('Unavailable channels cannot supervise SSL training')
        recon[b,targets]=True
        arrays.append(segment)
    clean=normalizer.transform(np.stack(arrays),av[0].numpy())
    target=torch.from_numpy(clean)
    return {k:v.to(device) for k,v in dict(signal=torch.where(av[...,None]&~recon,target,0),target=target,channel_ids=torch.arange(22).expand(len(samples),-1),availability_mask=av,reconstruction_mask=recon).items()}


class ReconstructionAccumulator:
    """Sufficient statistics per electrode; pooled target-element RMSE, not mean RMSE.
    Pearson pools target elements per electrode, not a mean over trial correlations.
    """
    def __init__(self):
        self.s=np.zeros((8,22),dtype=np.float64)

    def update(self,pred,target,mask,scale):
        p=pred.detach().cpu().double().numpy(); t=target.detach().cpu().double().numpy()
        m=mask.cpu().numpy(); scale=np.asarray(scale).reshape(22)
        for c in range(22):
            a,b=p[:,c][m[:,c]],t[:,c][m[:,c]]
            error=((a-b)**2).sum()
            self.s[:,c]+=np.array([len(a),error,error*scale[c]**2,a.sum(),b.sum(),(a*a).sum(),(b*b).sum(),(a*b).sum()])

    def result(self):
        n,se,sev,sp,st,spp,stt,spt=self.s
        def rmse(errors,counts):
            return float(np.sqrt(errors/counts)) if counts else None
        corr=[]
        for c in range(22):
            cov=spt[c]-sp[c]*st[c]/n[c] if n[c] else 0
            vp=spp[c]-sp[c]**2/n[c] if n[c] else 0
            vt=stt[c]-st[c]**2/n[c] if n[c] else 0
            corr.append(float(np.clip(cov/np.sqrt(vp*vt),-1,1)) if n[c]>1 and vp>0 and vt>0 else None)
        regions={}
        for name,ids in [('inside8',[i for i,c in enumerate(CHANNELS) if c in EIGHT]),('outside14',[i for i,c in enumerate(CHANNELS) if c not in EIGHT])]:
            regions[name]={'target_elements':int(n[ids].sum()),'squared_error_sum_V2':float(sev[ids].sum()),'rmse_V':rmse(sev[ids].sum(),n[ids].sum()),'rmse_standardized':rmse(se[ids].sum(),n[ids].sum())}
        return {'regions':regions,'squared_error_sum_V2':float(sev.sum()),'target_elements':int(n.sum()),'rmse_V':rmse(sev.sum(),n.sum()),'rmse_standardized':rmse(se.sum(),n.sum()),'electrodes':{c:{'target_elements':int(n[i]),'rmse_V':rmse(sev[i],n[i]),'rmse_standardized':rmse(se[i],n[i]),'pearson':corr[i]} for i,c in enumerate(CHANNELS)}}


@torch.no_grad()
def evaluate(model,data,normalizer,plan,batch_size=8,device='cpu'):
    was_training=model.training
    model.eval()
    total=ReconstructionAccumulator()
    per_seed={s:ReconstructionAccumulator() for s in plan['seeds']}
    per_subject={s:ReconstructionAccumulator() for s in np.unique(data.subject)}
    try:
        for start in range(0,len(plan['events']),batch_size):
            samples=plan['events'][start:start+batch_size]
            # Full availability means target14 can be scored, but all14 are masked.
            # The encoder sees precisely the eight complement electrodes.
            batch=planned_batch(data,normalizer,samples,22,plan['length'],device)
            pred=model(**{k:v for k,v in batch.items() if k!='target'})
            mask=batch['reconstruction_mask']
            total.update(pred,batch['target'],mask,normalizer.std)
            for i,s in enumerate(samples):
                for acc in (per_seed[s['seed']],per_subject[s['subject']]):
                    acc.update(pred[i:i+1],batch['target'][i:i+1],mask[i:i+1],normalizer.std)
    finally:
        model.train(was_training)
    return {'coverage':evaluation_coverage(plan,minimum=0),'plan_sha256':digest(plan),'protocol':plan['protocol'],'aggregation':'sqrt(sum target squared error / total target elements); pooled over seeds/windows/trials; Pearson pooled per electrode','overall':total.result(),'by_seed':{str(k):v.result() for k,v in per_seed.items()},'by_subject':{str(k):v.result() for k,v in per_subject.items()}}
