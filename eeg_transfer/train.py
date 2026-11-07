"""Bounded synthetic smoke runner, never a scientific experiment."""
import argparse
import os
import json
import random
import platform
import subprocess
from pathlib import Path
import numpy as np
import torch
from .data import synthetic, split_sessions, Standardizer
from .masking import availability, make_batch, masked_mse
from .model import EEGModel


def seed_all(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def update_schedule(p22, steps, seed, control='mixed'):
    """Ratio is fraction of updates; every sample in an update has same montage.
    Exposure control: retain only mixed schedule's 22ch updates.
    Extra control: identical fixed 22ch count plus extra 8ch updates.
    Schedule rounding is explicit and realized counts are saved.
    """
    if not 0 <= p22 <= 1 or steps < 1:
        raise ValueError('Invalid schedule')
    n22 = round(steps*p22)
    if control == 'mixed':
        schedule = [22]*n22+[8]*(steps-n22)
    elif control == 'exposure_matched':
        schedule = [22]*n22
    elif control == 'extra_8':
        if p22 == 0:
            raise ValueError('Extra control requires p22>0')
        schedule = [22]*steps+[8]*round(steps*(1-p22)/p22)
    else:
        raise ValueError('Unknown control')
    np.random.default_rng(seed).shuffle(schedule)
    return schedule


def run(config, out):
    if config.get('data') != 'synthetic':
        raise ValueError('Runner is smoke-only; real-data orchestration requires protocol review')
    seed_all(config['seed'])
    torch.set_num_threads(1)
    device = torch.device('cuda' if torch.cuda.is_available() and config.get('device','auto') != 'cpu' else 'cpu')
    train,val,test = split_sessions(synthetic(config['seed'],time=config['time']))
    norm = Standardizer().fit(train)
    model = EEGModel(time=config['time'],dim=config['dim'],layers=config['layers']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config['lr'])
    rng = np.random.default_rng(config['seed'])
    schedule = update_schedule(config['p22'],config['steps'],config['seed'],config.get('control','mixed'))
    history = []
    for step,kind in enumerate(schedule):
        ids = rng.integers(len(train.x),size=config['batch_size'])
        available = availability(kind,len(ids))
        # Normalization never mixes channels; unavailable data is removed first.
        x = torch.from_numpy(norm.transform(train.x[ids],available[0].numpy())).to(device)
        batch = make_batch(x,available.to(device))
        pred = model(**{k:v for k,v in batch.items() if k != 'target'})
        loss = masked_mse(pred,batch['target'],batch['availability_mask'],batch['reconstruction_mask'])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        history.append({'step':step,'channels':kind,'loss':loss.item(),'trial_ids':train.trial[ids].tolist()})
    out = Path(out); out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    (out/'train.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in history))
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':config,'torch_rng':torch.get_rng_state()},out/'checkpoint.pt')
    np.savez(out/'normalization.npz',mean=norm.mean,std=norm.std)
    manifest = {'synthetic_only':True,'device':str(device),'torch':torch.__version__,'numpy':np.__version__,'python':platform.python_version(),'parameters':sum(p.numel() for p in model.parameters()),'updates_22':schedule.count(22),'updates_8':schedule.count(8),'splits':{k:list(zip(d.subject.tolist(),d.session.tolist(),d.trial.tolist())) for k,d in [('train',train),('validation',val),('test',test)]}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'environment.txt').write_text(subprocess.check_output([__import__('sys').executable,'-m','pip','freeze'],text=True))
    return history

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/smoke.json'); parser.add_argument('--out',default='results/smoke')
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()),args.out)
