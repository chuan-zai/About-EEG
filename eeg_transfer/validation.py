"""Bounded V1/V2 learning gate. Loads Session T MAT only; never reads Session E."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .bnci import load_bnci_mat, sha256
from .data import Standardizer
from .models import build_model
from .plans import training_plans, evaluation_plan, digest, trial_keys
from .reconstruction import evaluate, planned_batch
from .masking import masked_mse
from .train import seed_all


class ConstantPrediction(nn.Module):
    def __init__(self,waveform):
        super().__init__(); self.register_buffer('waveform',torch.tensor(waveform,dtype=torch.float32))
    def forward(self,signal,**kwargs):
        return self.waveform.expand(signal.shape[0],-1,-1)


def train_only_baselines(train,norm,length,stride):
    # Average ordered windows from TRAIN only, never validation/test or target input.
    arrays=[train.x[i,:,start:start+length] for i in range(len(train.x)) for start in range(0,train.x.shape[-1]-length+1,stride)]
    waveform=np.stack(arrays).mean(0,keepdims=True)
    standardized=norm.transform(waveform,np.ones(22,dtype=bool))
    physical_zero=-norm.mean/norm.std
    return {'zero_volts':ConstantPrediction(np.broadcast_to(physical_zero,(1,22,length)).copy()),'train_mean_waveform':ConstantPrediction(standardized)}


def forward_flops(model,batch):
    # PyTorch reports only supported matmul/conv ops; NOT a complete hardware FLOP count.
    with torch.no_grad(), torch.profiler.profile(with_flops=True) as prof:
        model(**{k:v for k,v in batch.items() if k!='target'})
    return {'supported_forward_flops_per_batch':sum(int(e.flops or 0) for e in prof.key_averages()),'batch_size':len(batch['signal']),'method':'torch.profiler with_flops; incomplete estimate; excludes backward and unsupported/fused ops'}


def learning_gate(config_path,out):
    config=json.loads(Path(config_path).read_text())
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    (out/'predeclared-config.json').write_text(json.dumps(config,indent=2))
    torch.set_num_threads(1)
    data,audit=load_bnci_mat('data/raw/A01T.mat',1,'T')
    rng=np.random.default_rng(config['split_seed']); ids=rng.permutation(len(data.x)); n=round(len(ids)*.2)
    train=data.take(ids[n:]); val=data.take(ids[:n][:config['validation_trial_limit']])
    norm=Standardizer().fit(train)
    np.savez(out/'normalization.npz',mean=norm.mean,std=norm.std)
    plan=evaluation_plan(val,config['time'],config['stride'],config['evaluation_seeds'])
    (out/'validation-plan.json').write_text(json.dumps(plan,indent=2))
    (out/'splits.json').write_text(json.dumps({'train':trial_keys(train),'validation':trial_keys(val)},indent=2))
    baselines={k:evaluate(m,val,norm,plan,batch_size=8) for k,m in train_only_baselines(train,norm,config['time'],config['stride']).items()}
    (out/'baselines.json').write_text(json.dumps(baselines,indent=2))
    rows=[]
    for seed in config['seeds']:
        events=training_plans(train,config['steps'],config['batch_size'],config['time'],config['stride'],seed)['groups']['A0']
        for version in ('v1','v2'):
            seed_all(seed)
            cfg=dict(config,model_version=version,**config[version])
            model=build_model(cfg)
            optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
            where=out/f'{version}-{seed}'; where.mkdir()
            (where/'events.json').write_text(json.dumps(events,indent=2))
            initial=evaluate(model,val,norm,plan,batch_size=8)
            first=planned_batch(train,norm,events[0]['samples'],22,config['time'])
            flops=forward_flops(model,first)
            started=time.perf_counter(); model.train()
            with (where/'train.jsonl').open('w') as log:
                for step,event in enumerate(events):
                    batch=planned_batch(train,norm,event['samples'],22,config['time'])
                    pred=model(**{k:v for k,v in batch.items() if k!='target'})
                    loss=masked_mse(pred,batch['target'],batch['availability_mask'],batch['reconstruction_mask'])
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    grad=torch.sqrt(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None))
                    if not torch.isfinite(loss) or not torch.isfinite(grad):
                        raise FloatingPointError('Nonfinite learning gate update')
                    optimizer.step(); log.write(json.dumps({'step':step,'loss':loss.item(),'gradient_norm_l2':grad.item(),'event_sha256':digest(event)})+'\n')
            seconds=time.perf_counter()-started
            final=evaluate(model,val,norm,plan,batch_size=8)
            torch.save({'model':model.state_dict(),'config':cfg},where/'checkpoint.pt')
            record={'version':version,'seed':seed,'parameters':sum(p.numel() for p in model.parameters()),'training_seconds':seconds,'flops':flops,'initial':initial,'final':final}
            rows.append(record); (where/'report.json').write_text(json.dumps(record,indent=2))
            print(version,seed,'initial',initial['overall']['rmse_V'],'final',final['overall']['rmse_V'],'seconds',seconds,flush=True)
    threshold=config['gate']; best=min(r['overall']['rmse_V'] for r in baselines.values())
    v1={r['seed']:r for r in rows if r['version']=='v1'}; v2=[r for r in rows if r['version']=='v2']
    baseline_wins=sum(1-r['final']['overall']['rmse_V']/best>=threshold['min_baseline_relative_improvement'] for r in v2)
    learning_wins=sum(1-r['final']['overall']['rmse_V']/r['initial']['overall']['rmse_V']>=threshold['min_untrained_relative_improvement'] for r in v2)
    paired=[1-r['final']['overall']['rmse_V']/v1[r['seed']]['final']['overall']['rmse_V'] for r in v2]
    performance_pass=bool(baseline_wins>=threshold['min_seeds_passing'] and learning_wins>=threshold['min_seeds_passing'] and np.median(paired)>=threshold['min_v1_median_relative_improvement'] and max(r['parameters'] for r in v2)<=threshold['max_parameters'])
    result={'selection_data':'S01 Session T only; E file not opened','config_sha256':sha256(config_path),'training_raw_sha256':audit['sha256'],'baseline_best_rmse_V':best,'baseline_wins':baseline_wins,'learning_wins':learning_wins,'paired_v1_relative_improvements':paired,'performance_gate_passed':performance_pass,'safety_tests_required_separately':True,'records':rows,'scientific_conclusion':None}
    (out/'gate.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/v2-validation.json'); p.add_argument('--out',default='results/v2-validation')
    a=p.parse_args(); learning_gate(a.config,a.out)
