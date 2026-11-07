"""Official BNCI MAT adapter. Follows MOABB channel order/uV conversion.
MAT trial positions mark trial onset, cue is +2 s; use [2,6) s.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import urllib.request
import numpy as np
from scipy.io import loadmat
from .data import CHANNELS, Trials, split_sessions

BASE = 'https://bnci-horizon-2020.eu/database/data-sets/001-2014/'

def sha256(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def download(subject, root):
    if subject not in range(1,10):
        raise ValueError('Subject must be 1..9')
    root = Path(root); root.mkdir(parents=True,exist_ok=True)
    files = []
    for session in ('T','E'):
        path = root/f'A{subject:02d}{session}.mat'
        url = BASE+path.name
        if not path.exists():
            temporary = path.with_suffix('.part')
            try:
                with urllib.request.urlopen(url,timeout=120) as response, temporary.open('wb') as out:
                    while block := response.read(1024*1024):
                        out.write(block)
                loadmat(temporary,variable_names=['data'])  # fail before atomic rename
                temporary.rename(path)
            finally:
                temporary.unlink(missing_ok=True)
        files.append({'url':url,'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':sha256(path)})
    return files


def load_bnci_mat(path, subject, session, reject_artifacts=True):
    runs = loadmat(path,simplify_cells=True)['data']
    if isinstance(runs,dict):
        runs = [runs]
    signals,labels,ids,metadata,run_reports = [],[],[],[],[]
    for ri,run in enumerate(runs):
        x = np.asarray(run['X'])
        onsets = np.atleast_1d(run['trial']).astype(int)
        y = np.atleast_1d(run['y']).astype(int)
        artifacts = np.atleast_1d(run['artifacts']).astype(int)
        if x.ndim != 2 or x.shape[1] != 25 or float(run['fs']) != 250:
            raise ValueError('Unexpected BNCI channel count or sfreq')
        if len(onsets)!=len(y) or len(y)!=len(artifacts):
            raise ValueError('Trial metadata mismatch')
        rr = {'run':ri+1,'shape':list(x.shape),'trials':len(y),'artifact_flags':int((artifacts!=0).sum()),'nonfinite_values':int((~np.isfinite(x)).sum()),'accepted':0,'dropped':[]}
        for j,(onset,label,artifact) in enumerate(zip(onsets,y,artifacts)):
            if label not in (1,2,3,4):
                raise ValueError('Invalid class metadata')
            start = int(onset)-1+500
            stop = start+1000
            key = f'run{ri+1:02d}_trial{j+1:03d}'
            reason = None
            if reject_artifacts and artifact:
                reason = 'official_artifact'
            elif start<0 or stop>x.shape[0]:
                reason = 'out_of_bounds'
            elif not np.isfinite(x[start:stop,:22]).all():
                reason = 'nonfinite_EEG'
            if reason:
                rr['dropped'].append({'trial':key,'reason':reason}); continue
            signal = x[start:stop,:22].T.astype(np.float32)*1e-6
            signals.append(signal); labels.append(int(label)-1); ids.append(key)
            metadata.append({'run':ri+1,'trial_in_run':j+1,'onset_mat_1based':int(onset),'cue_sample_0based':start,'stop_exclusive':stop,'artifact':int(artifact),'original_label':int(label)})
            rr['accepted'] += 1
        run_reports.append(rr)
    if not signals:
        raise ValueError('No valid EEG trials')
    n = len(signals)
    data = Trials(np.stack(signals),np.array(labels),np.repeat(f'S{subject:02d}',n),np.repeat(session,n),np.array(ids),250.,CHANNELS,metadata)
    report = {'path':str(Path(path).resolve()),'sha256':sha256(path),'subject':f'S{subject:02d}','session':session,'sfreq':250.,'channels':list(CHANNELS),'excluded_channels':['EOG1','EOG2','EOG3'],'source_unit':'uV','output_unit':'V','scale':1e-6,'epoch':'MAT trial onset + [2,6) seconds; 1000 samples','reject_artifacts':reject_artifacts,'runs':run_reports,'accepted_trials':n,'class_counts':np.bincount(data.y,minlength=4).tolist(),'min_V':float(data.x.min()),'max_V':float(data.x.max()),'std_V':float(data.x.std()),'constant_electrode_trials':int((data.x.std(-1)==0).sum()),'nonfinite_epoch_values':int((~np.isfinite(data.x)).sum()),'peak_abs_above_1mV_trials':int((np.abs(data.x).max((1,2))>1e-3).sum())}
    return data,report


def save_trials(path,data):
    np.savez_compressed(path,x=data.x,y=data.y,subject=data.subject,session=data.session,trial=data.trial,sfreq=data.sfreq,channels=data.channels,metadata_json=json.dumps(data.metadata))


def prepare(subject, root, output, report_path, allow_download=False):
    root = Path(root)
    files = download(subject,root) if allow_download else [{'path':str((root/f'A{subject:02d}{s}.mat').resolve()),'sha256':sha256(root/f'A{subject:02d}{s}.mat'),'url':BASE+f'A{subject:02d}{s}.mat','bytes':(root/f'A{subject:02d}{s}.mat').stat().st_size} for s in ('T','E')]
    pairs = [load_bnci_mat(root/f'A{subject:02d}{s}.mat',subject,s) for s in ('T','E')]
    a,b = [p[0] for p in pairs]
    data = Trials(np.concatenate([a.x,b.x]),np.concatenate([a.y,b.y]),np.concatenate([a.subject,b.subject]),np.concatenate([a.session,b.session]),np.concatenate([a.trial,b.trial]),250.,CHANNELS,a.metadata+b.metadata)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    save_trials(output,data)
    train,val,test = split_sessions(data)
    report = {'created_utc':datetime.now(timezone.utc).isoformat(),'source_page':'https://bnci-horizon-2020.eu/database/data-sets','license':'CC BY-ND 4.0; attribution required; do not redistribute derived data without reviewing license','citation':'Tangermann et al. (2012), doi:10.3389/fnins.2012.00055','adapter_version':1,'files':files,'processed':{'path':str(Path(output).resolve()),'sha256':sha256(output)},'sessions':[p[1] for p in pairs],'split_seed':42,'split_counts':{'train':len(train.x),'validation':len(val.x),'test':len(test.x)},'splits':{k:d.trial.tolist() for k,d in [('train',train),('validation',val),('test',test)]}}
    Path(report_path).parent.mkdir(parents=True,exist_ok=True)
    Path(report_path).write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--subject',type=int,default=1); p.add_argument('--raw',default='data/raw'); p.add_argument('--output',default='data/processed/S01.npz'); p.add_argument('--report',default='reports/data-audit-S01.json'); p.add_argument('--download',action='store_true')
    a=p.parse_args(); prepare(a.subject,a.raw,a.output,a.report,a.download)
