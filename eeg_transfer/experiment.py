"""Single-subject RQ1 runner. Formal runs must be launched explicitly."""
import argparse
import gzip
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from .bnci import sha256
from .data import load_npz, split_sessions, Standardizer
from .train import seed_all
from .models import build_model
from .masking import masked_mse
from .plans import training_plans, evaluation_plan, trial_keys, digest, budget, evaluation_coverage
from .reconstruction import planned_batch, evaluate


def dump(path,value):
    target=Path(path)
    temporary=target.with_suffix(target.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False))
    temporary.replace(target)


def prepare(config,out):
    data=load_npz(config['data_path'])
    if len(np.unique(data.subject))!=1:
        raise ValueError('Run each subject independently')
    train,val,test=split_sessions(data,seed=config['split_seed'])
    splits={'train':trial_keys(train),'validation':trial_keys(val),'test':trial_keys(test)}
    plans=training_plans(train,config['steps'],config['batch_size'],config['time'],config['stride'],config['seed'])
    # Smoke limit is declared in config, never selected by test performance.
    limit=config.get('evaluation_trial_limit')
    eval_data={'validation':val,'test':test}
    evaluations={}
    for split,d in eval_data.items():
        if limit is not None:
            d=d.take(np.arange(min(limit,len(d.x))))
        for protocol in ('random25','outside14'):
            seeds=config['evaluation_seeds'] if protocol=='random25' else [0]
            evaluations[f'{split}_{protocol}']=evaluation_plan(d,config['time'],config['stride'],seeds,protocol)
            if protocol=='random25' and config.get('minimum_evaluation_coverage'):
                evaluation_coverage(evaluations[f'{split}_{protocol}'],config['minimum_evaluation_coverage'][split])
    package={'config':config,'data_sha256':sha256(config['data_path']),'splits':splits,'training':plans,'evaluation':evaluations,'budgets':{k:budget(v,config['time']) for k,v in plans['groups'].items()}}
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    if config.get('compress_plans',False):
        with gzip.open(out/'plans.json.gz','wt') as f:
            json.dump(package,f,separators=(',',':'),allow_nan=False)
    else:
        dump(out/'plans.json',package)
    dump(out/'config.json',config)
    dump(out/'budgets.json',package['budgets'])
    return package


def run(package_path,group,out,evaluate_test=False):
    if str(package_path).endswith('.gz'):
        with gzip.open(package_path,'rt') as f:
            package=json.load(f)
    else:
        package=json.loads(Path(package_path).read_text())
    config=package['config']
    if sha256(config['data_path'])!=package['data_sha256']:
        raise ValueError('Data changed since planning')
    if config.get('protocol_lock'):
        lock=json.loads(Path(config['protocol_lock']).read_text())
        if not lock['ready_for_formal_run'] or lock['architecture'] != {k:config[k] for k in lock['architecture']} or lock['fixed_hyperparameters'] != {k:config[k] for k in lock['fixed_hyperparameters']}:
            raise ValueError('Formal protocol is not locked for this architecture')
    data=load_npz(config['data_path'])
    train,val,test=split_sessions(data,seed=config['split_seed'])
    splits={'train':trial_keys(train),'validation':trial_keys(val),'test':trial_keys(test)}
    if splits!=package['splits']:
        raise ValueError('Split mismatch')
    events=package['training']['groups'][group]
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    seed_all(config['seed']); torch.set_num_threads(config.get('threads',1))
    device=torch.device('cuda' if config['device']=='auto' and torch.cuda.is_available() else 'cpu')
    norm=Standardizer().fit(train)
    np.savez(out/'normalization.npz',mean=norm.mean,std=norm.std)
    model=build_model(config).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'],betas=(0.9,0.999),eps=1e-8)
    initial_sha=digest({k:v.detach().cpu().tolist() for k,v in model.state_dict().items()})
    manifest={'config':config,'group':group,'data_sha256':package['data_sha256'],'package_sha256':sha256(package_path),'training_plan_sha256':digest(events),'initial_weights_sha256':initial_sha,'budget':package['budgets'][group],'splits':splits,'device':str(device),'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'parameters':sum(p.numel() for p in model.parameters()),'model_version':config.get('model_version','v1'),'optimizer_audit':{'name':'AdamW','lr':config['lr'],'weight_decay':config['weight_decay'],'betas':[0.9,0.999],'eps':1e-8,'scheduler':'none; constant LR per optimizer step','state':'independent fresh optimizer per group, no moment sharing; additional low-channel steps alter moments and step index','resume':'not supported; restart unchanged plan in a new directory; no automatic retry or seed replacement'},'test_evaluated':evaluate_test,'scientific_conclusion':None,'code_sha256':{p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},'protocol_lock_sha256':sha256(config['protocol_lock']) if config.get('protocol_lock') else None}
    dump(out/'manifest.json',manifest)
    dump(out/'config.json',config)
    # Local copy freezes concrete trial, window and mask plans for this run.
    if config.get('compress_plans',False):
        with gzip.open(out/'plans.json.gz','wt') as f:
            json.dump(package,f,separators=(',',':'),allow_nan=False)
    else:
        dump(out/'plans.json',package)
    dump(out/'status.json',{'state':'started','completed_updates':0})
    (out/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    start=time.perf_counter()
    model.train()
    with (out/'train.jsonl').open('w') as f:
        for step,event in enumerate(events):
            before=time.perf_counter()
            batch=planned_batch(train,norm,event['samples'],event['montage'],config['time'],device)
            pred=model(**{k:v for k,v in batch.items() if k!='target'})
            loss=masked_mse(pred,batch['target'],batch['availability_mask'],batch['reconstruction_mask'])
            optimizer.zero_grad(set_to_none=True); loss.backward()
            grad_norm=torch.sqrt(sum(p.grad.detach().square().sum() for p in model.parameters() if p.grad is not None))
            if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
                raise FloatingPointError('Nonfinite loss or gradient')
            optimizer.step()
            if device.type=='cuda':
                torch.cuda.synchronize()
            row={'step':step,'event_id':event['id'],'event_sha256':digest(event),'seed':event['seed'],'montage':event['montage'],'optimizer_step':step+1,'learning_rate':optimizer.param_groups[0]['lr'],'loss':loss.item(),'gradient_norm_l2':grad_norm.item(),'visible_channels_per_sample':(batch['availability_mask']&~batch['reconstruction_mask'].any(-1)).sum(-1).tolist(),'target_channels_per_sample':batch['reconstruction_mask'].any(-1).sum(-1).tolist(),'target_elements':int(batch['reconstruction_mask'].sum()),'seconds':time.perf_counter()-before}
            f.write(json.dumps(row)+'\n'); f.flush()
    training_seconds=time.perf_counter()-start
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':config,'group':group,'package_sha256':sha256(package_path)},out/'checkpoint.pt.tmp')
    (out/'checkpoint.pt.tmp').replace(out/'checkpoint.pt')
    metrics={}
    for name,plan in package['evaluation'].items():
        if name.startswith('test') and not evaluate_test:
            continue
        d=test if name.startswith('test') else val
        metrics[name]=evaluate(model,d,norm,plan,batch_size=config.get('evaluation_batch_size',8),device=device)
    dump(out/'evaluation.json',metrics)
    dump(out/'runtime.json',{'training_seconds':training_seconds,'budget':budget(events,config['time']),'peak_cuda_memory_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None})
    dump(out/'status.json',{'state':'complete','completed_updates':len(events),'optimizer_steps':[int(state['step']) for state in optimizer.state.values()]})
    return manifest,metrics

if __name__=='__main__':
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare'); prep.add_argument('--config',required=True); prep.add_argument('--out',required=True)
    fit=sub.add_parser('run'); fit.add_argument('--plans',required=True); fit.add_argument('--group',required=True); fit.add_argument('--out',required=True); fit.add_argument('--evaluate-test',action='store_true')
    a=p.parse_args()
    if a.command=='prepare':
        prepare(json.loads(Path(a.config).read_text()),a.out)
    else:
        run(a.plans,a.group,a.out,a.evaluate_test)
