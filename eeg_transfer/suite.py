"""Prepare all fixed runs and commands without launching any training."""
import argparse
import json
from pathlib import Path
from .experiment import prepare
from .plans import digest, evaluation_coverage
from .bnci import sha256


def prepare_suite(config_path,out,reports="reports/phase3",results_root="results/rq1-v2"):
    reports=Path(reports)
    for name in ("rq1-manifest.json", "commands.txt"):
        if (reports/name).exists():
            raise FileExistsError(f"Refusing to overwrite historical suite record: {reports/name}")
    cfg=json.loads(Path(config_path).read_text())
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    manifest={'suite_config':cfg,'suite_config_sha256':sha256(config_path),'subjects':cfg['subjects'],'seeds':cfg['seeds'],'groups':cfg['groups'],'runs':[],'evaluation_hashes':{},'plan_audits':[]}
    commands=[]
    for subject in cfg['subjects']:
        expected_eval=None; expected_splits=None
        for seed in cfg['seeds']:
            c=dict(cfg['template'],data_path=f'data/processed/S{subject:02d}.npz',seed=seed)
            dest=out/f'S{subject:02d}'/str(seed)
            package=prepare(c,dest)
            path=dest/'plans.json.gz'; path_sha=sha256(path)
            ev={k:digest(v) for k,v in package['evaluation'].items()}
            if expected_eval is None:
                expected_eval=ev; expected_splits=package['splits']
            if ev!=expected_eval or package['splits']!=expected_splits:
                raise ValueError('Evaluation or split changed across seeds')
            groups=package['training']['groups']; a0=groups['A0']
            for name in ('A1','A2','A3'):
                high=[e for e in groups[name] if e['montage']==22]
                if high!=a0[:len(high)] or high!=groups[name+'_exposure'] or [e for e in groups[name+'_extra'] if e['montage']==22]!=a0:
                    raise ValueError('Unpaired high-channel events')
            audit={'subject':subject,'seed':seed,'plans_sha256':path_sha,'coverage':{k:evaluation_coverage(p,100 if k.startswith('test') else 20) for k,p in package['evaluation'].items() if p['protocol']=='random25'},'budgets':package['budgets'],'high_events_paired':True}
            manifest['plan_audits'].append(audit)
            for group in cfg['groups']:
                identity=f'S{subject:02d}-seed{seed}-{group}'
                output=str(Path(results_root)/identity)
                manifest['runs'].append({'id':identity,'subject':subject,'seed':seed,'group':group,'plans':str(path),'plans_sha256':path_sha,'output':output,'data_sha256':package['data_sha256']})
                commands.append(f'.venv/bin/python -m eeg_transfer.experiment run --plans {path} --group {group} --out {output} --evaluate-test')
            print(f'Prepared S{subject:02d} seed {seed}',flush=True)
        manifest['evaluation_hashes'][str(subject)]=expected_eval
    reports.mkdir(parents=True,exist_ok=True)
    (reports/'rq1-manifest.json').write_text(json.dumps(manifest,indent=2))
    (reports/'commands.txt').write_text('\n'.join(commands)+'\n')
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/rq1-suite.json'); p.add_argument('--out',default='data/plans/rq1-v2')
    p.add_argument("--reports",default="reports/phase3")
    p.add_argument("--results-root",default="results/rq1-v2")
    a=p.parse_args(); prepare_suite(a.config,a.out,a.reports,a.results_root)
