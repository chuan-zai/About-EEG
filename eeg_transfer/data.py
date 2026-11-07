"""Trial-level I/O. Signals are volts; arrays [trial, channel, time]. No downloads."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

CHANNELS = ('Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz')
EIGHT = ('Fz','FC3','FC4','C3','Cz','C4','CPz','Pz')

@dataclass
class Trials:
    x: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    session: np.ndarray
    trial: np.ndarray
    sfreq: float
    channels: tuple = CHANNELS
    metadata: list | None = None

    def __post_init__(self):
        self.channels = tuple(self.channels)
        if self.x.ndim != 3 or self.x.shape[1] != 22 or self.channels != CHANNELS:
            raise ValueError('Require canonical 22 EEG channels, explicitly reordered; EOG excluded')
        if not all(len(a) == len(self.x) for a in (self.y,self.subject,self.session,self.trial)):
            raise ValueError('Metadata length mismatch')
        if not np.isfinite(self.x).all() or self.sfreq <= 0:
            raise ValueError('Invalid signal or sampling frequency')
        if self.metadata is not None and len(self.metadata) != len(self.x):
            raise ValueError('Metadata records length mismatch')
        keys = list(zip(self.subject.tolist(),self.session.tolist(),self.trial.tolist()))
        if len(set(keys)) != len(keys):
            raise ValueError('Trial keys must be unique')

    def take(self, ids):
        return Trials(self.x[ids],self.y[ids],self.subject[ids],self.session[ids],self.trial[ids],self.sfreq,self.channels, None if self.metadata is None else [self.metadata[int(i)] for i in ids])


def load_npz(path):
    """Required keys x,y,subject,session,trial,sfreq,channels; no pickle allowed."""
    with np.load(path, allow_pickle=False) as z:
        return Trials(*(z[k] for k in ('x','y','subject','session','trial')),float(z['sfreq']),tuple(z['channels'].tolist()), __import__('json').loads(str(z['metadata_json'])) if 'metadata_json' in z else None)


def load_mat(path):
    """Explicit normalized MAT schema, NOT arbitrary competition MAT variants."""
    from scipy.io import loadmat
    z = loadmat(path, simplify_cells=True)
    return Trials(np.asarray(z['x']),np.atleast_1d(z['y']),np.atleast_1d(z['subject']),np.atleast_1d(z['session']),np.atleast_1d(z['trial']),float(z['sfreq']),tuple(z['channels']))


def load_gdf(path, subject, session, channel_map, tmin=0., tmax=3.996):
    """Explicit raw-name -> canonical-name map avoids guessing unnamed GDF electrodes.
    Cue events 769..772; MNE handles GDF calibration to volts. No filtering across splits.
    Evaluation sessions without class annotations require separately verified labels.
    """
    import mne
    raw = mne.io.read_raw_gdf(str(path), preload=True, verbose=False)
    raw.rename_channels(channel_map)
    if not set(CHANNELS).issubset(raw.ch_names):
        raise ValueError('Incomplete canonical EEG mapping')
    raw.pick(list(CHANNELS)).reorder_channels(list(CHANNELS))
    events, codes = mne.events_from_annotations(raw, verbose=False)
    wanted = {str(769+i):i+1 for i in range(4)}
    if not set(wanted).issubset(codes):
        raise ValueError('Missing four-class cue labels; provide verified labeled NPZ')
    remapped = events.copy()
    keep = np.isin(events[:,2],[codes[k] for k in wanted])
    remapped = remapped[keep]
    for k,v in wanted.items():
        remapped[events[keep,2] == codes[k],2] = v
    epochs = mne.Epochs(raw, remapped, event_id=wanted, tmin=tmin,tmax=tmax,baseline=None,preload=True,reject_by_annotation=True,verbose=False)
    n = len(epochs)
    return Trials(epochs.get_data().astype('float32'),epochs.events[:,2]-1,np.full(n,str(subject)),np.full(n,str(session)),epochs.selection.astype(str),raw.info['sfreq'])


def split_sessions(data, train_session='T', test_session='E', val_fraction=.2, seed=42):
    """Within-subject random trial validation; no test-label use. Split before windows."""
    if train_session == test_session or not 0 < val_fraction < 1:
        raise ValueError('Invalid session split')
    rng = np.random.default_rng(seed)
    train, val = [], []
    for subject in np.unique(data.subject):
        ids = np.flatnonzero((data.subject == subject) & (data.session == train_session))
        if len(ids) < 2:
            raise ValueError('Each subject needs at least two training-session trials')
        ids = rng.permutation(ids)
        n = min(len(ids)-1,max(1,round(len(ids)*val_fraction)))
        val.extend(ids[:n]); train.extend(ids[n:])
    test = np.flatnonzero(data.session == test_session)
    if not len(test):
        raise ValueError('Missing test session')
    return tuple(data.take(np.asarray(i)) for i in (train,val,test))


class Standardizer:
    def fit(self, train):
        self.mean = train.x.mean(axis=(0,2),keepdims=True)
        self.std = train.x.std(axis=(0,2),keepdims=True).clip(1e-8)
        return self

    def transform(self, x, available):
        # Select before arithmetic: unavailable NaN/large values cannot propagate.
        out = np.zeros_like(x,dtype=np.float32)
        out[:,available] = (x[:,available]-self.mean[:,available])/self.std[:,available]
        return out


def windows(data, length, stride):
    if length <= 0 or stride <= 0 or length > data.x.shape[-1]:
        raise ValueError('Invalid window size')
    for i in range(len(data.x)):
        for start in range(0,data.x.shape[-1]-length+1,stride):
            yield i, start, data.x[i,:,start:start+length]


def synthetic(seed=42, n=40, time=64):
    rng = np.random.default_rng(seed)
    return Trials(rng.normal(0,1e-5,(n,22,time)).astype('float32'),np.arange(n)%4,np.repeat('S01',n),np.where(np.arange(n)<n*3//4,'T','E'),np.arange(n).astype(str),250.)
