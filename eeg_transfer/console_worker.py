"""Bounded console workers. Never launch a full formal suite."""
import argparse
import json
from pathlib import Path
import numpy as np
from .experiment import prepare, run
from .bnci import download, prepare as prepare_data, sha256
from .data import load_npz, split_sessions, Standardizer


def acquire(subject, output):
    target=Path(f'data/processed/S{subject:02d}.npz')
    if target.exists(): raise FileExistsError('Derived data already exists')
    files=download(subject,'data/raw')
    old=json.loads(Path(f'reports/phase3/data/S{subject:02d}.json').read_text())
    expected={Path(f['path']).name:f['sha256'] for f in old['files']}
    for f in files:
        if f['sha256']!=expected[Path(f['path']).name]:
            raise ValueError('Official source SHA differs from historical audit')
    Path(output).mkdir(parents=True,exist_ok=False)
    prepare_data(subject,'data/raw',target,Path(output)/'audit.json')
    train,_,_=split_sessions(load_npz(target))
    norm=Standardizer().fit(train)
    np.savez(target.with_name(f'S{subject:02d}-train-normalization.npz'),mean=norm.mean,std=norm.std)
    print(f'S{subject:02d} acquired and audited; no training performed',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--subject',type=int)
    for name in ('config','plans','out'): p.add_argument('--'+name,required=name=='out')
    a=p.parse_args()
    if a.subject is not None:
        if a.subject not in range(1,10): raise ValueError('Invalid subject')
        acquire(a.subject,a.out)
    else:
        cfg=json.loads(Path(a.config).read_text())
        prepare(cfg,a.plans)
        run(Path(a.plans)/'plans.json.gz','A2',a.out,evaluate_test=False)
