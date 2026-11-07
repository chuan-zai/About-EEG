"""Download/audit independently; preserve successful work on failure."""
import argparse
import json
from pathlib import Path
from .bnci import prepare, sha256
from .data import load_npz, split_sessions, Standardizer
import numpy as np


def acquire(subjects,root='data',reports='reports/phase3/data'):
    root=Path(root); reports=Path(reports); reports.mkdir(parents=True,exist_ok=True)
    results=[]
    for subject in subjects:
        output=root/'processed'/f'S{subject:02d}.npz'
        audit=reports/f'S{subject:02d}.json'
        try:
            if subject==1 and output.exists() and Path('reports/data-audit-S01.json').exists():
                report=json.loads(Path('reports/data-audit-S01.json').read_text())
                for file in report['files']:
                    if sha256(file['path'])!=file['sha256']:
                        raise ValueError('Existing S01 raw data changed')
                if sha256(output)!=report['processed']['sha256']:
                    raise ValueError('Existing S01 processed data changed')
                audit.write_text(json.dumps(report,indent=2))
            elif output.exists() and audit.exists():
                report=json.loads(audit.read_text())
                if sha256(output)!=report['processed']['sha256']:
                    raise ValueError('Existing processed checksum mismatch')
                for file in report['files']:
                    if sha256(file['path'])!=file['sha256']:
                        raise ValueError('Existing raw checksum mismatch')
            else:
                report=prepare(subject,root/'raw',output,audit,allow_download=True)
            data=load_npz(output)
            train,val,test=split_sessions(data,seed=42)
            norm=Standardizer().fit(train)
            np.savez(root/'processed'/f'S{subject:02d}-train-normalization.npz',mean=norm.mean,std=norm.std)
            results.append({'subject':subject,'status':'ready','audit':str(audit),'data_path':str(output),'data_sha256':sha256(output),'split_counts':report['split_counts']})
            print(f'S{subject:02d} ready: {report["split_counts"]}',flush=True)
        except Exception as exc:
            results.append({'subject':subject,'status':'failed','error':f'{type(exc).__name__}: {exc}'})
            print(f'S{subject:02d} failed: {exc}',flush=True)
        (reports/'status.json').write_text(json.dumps(results,indent=2))
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--subjects',type=int,nargs='+',default=list(range(1,10)))
    a=p.parse_args(); acquire(a.subjects)
