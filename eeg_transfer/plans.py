"""Serializable, label-free training/evaluation events tied to trial keys."""
import hashlib
import json
import numpy as np
from .data import CHANNELS, EIGHT
from .train import update_schedule


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def trial_keys(data):
    return [dict(subject=str(s),session=str(e),trial=str(t)) for s,e,t in zip(data.subject,data.session,data.trial)]


def window_refs(data, length, stride):
    if not 0 < length <= data.x.shape[-1] or stride < 1:
        raise ValueError('Invalid windows')
    return [dict(**key,start=start) for key in trial_keys(data) for start in range(0,data.x.shape[-1]-length+1,stride)]


def target_channels(kind,rng):
    ids = np.array([i for i,c in enumerate(CHANNELS) if kind==22 or c in EIGHT])
    return sorted(rng.choice(ids,size=max(1,round(len(ids)*.25)),replace=False).tolist())


def training_plans(train, steps, batch_size, length, stride, seed):
    refs = window_refs(train,length,stride)
    # Separate montage streams: high events identical across ratios and controls.
    def event(kind, index):
        event_seed = int(np.random.SeedSequence([seed,kind,index]).generate_state(1)[0])
        rng = np.random.default_rng(event_seed)
        samples = [dict(**refs[int(rng.integers(len(refs)))],targets=target_channels(kind,rng)) for _ in range(batch_size)]
        return {'id':f'{kind}_{index:06d}','seed':event_seed,'montage':kind,'samples':samples}
    high = [event(22,i) for i in range(steps)]
    plans = {}
    for name,p in [('A0',1.),('A1',.75),('A2',.5),('A3',.25)]:
        schedule = update_schedule(p,steps,seed)
        counts={22:0,8:0}; mixed=[]
        for kind in schedule:
            idx=counts[kind]; counts[kind]+=1
            mixed.append(high[idx] if kind==22 else event(8,idx))
        plans[name]=mixed
        if name=='A0':
            continue
        plans[name+'_exposure']=[e for e in mixed if e['montage']==22]
        # Exactly same high prefix as A0, mixed with extra lows; preserve high order.
        extra_schedule=update_schedule(p,steps,seed,'extra_8')
        counts={22:0,8:0}; extra=[]
        for kind in extra_schedule:
            idx=counts[kind]; counts[kind]+=1
            extra.append(high[idx] if kind==22 else event(8,idx))
        plans[name+'_extra']=extra
    return {'version':1,'seed':seed,'length':length,'stride':stride,'ratio_unit':'updates','batch_size':batch_size,'training_trial_keys':trial_keys(train),'groups':plans}


def evaluation_plan(data, length, stride, seeds=(1101,2202,3303), protocol='random25'):
    if protocol not in ('random25','outside14'):
        raise ValueError('Unknown evaluation protocol')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Evaluation seeds must be nonempty and distinct')
    refs = window_refs(data,length,stride)
    events=[]
    for seed in seeds:
        rng=np.random.default_rng(seed)
        for ref in refs:
            targets = target_channels(22,rng) if protocol=='random25' else [i for i,c in enumerate(CHANNELS) if c not in EIGHT]
            events.append(dict(**ref,seed=seed,targets=targets))
    return {'version':1,'length':length,'stride':stride,'seeds':list(seeds),'protocol':protocol,'events':events}


def budget(events,length):
    return {'total_updates':len(events),'updates_22':sum(e['montage']==22 for e in events),'updates_8':sum(e['montage']==8 for e in events),'target_elements':sum(len(s['targets'])*length for e in events for s in e['samples']),'input_slots':sum(len(e['samples'])*22*length for e in events),'note':'Dense 22-slot model processes all slots; wall time is also logged. Extra/exposure controls have different total compute.'}


def evaluation_coverage(plan, minimum=1):
    """Coverage is an engineering count, never an independent statistical N."""
    counts=np.zeros(22,dtype=int); trials=[set() for _ in range(22)]; seen=set()
    for e in plan['events']:
        key=(e['subject'],e['session'],e['trial'],e['start'],e['seed'])
        if key in seen:
            raise ValueError('Duplicate evaluation event')
        seen.add(key)
        for c in e['targets']:
            counts[c]+=1; trials[c].add(key[:3])
    required=range(22) if plan['protocol']=='random25' else [i for i,c in enumerate(CHANNELS) if c not in EIGHT]
    if any(counts[c]<minimum for c in required):
        raise ValueError('Insufficient evaluation electrode coverage')
    return {'minimum_required':minimum,'target_windows_including_mask_seeds':dict(zip(CHANNELS,counts.tolist())),'unique_target_trials':{c:len(trials[i]) for i,c in enumerate(CHANNELS)},'warning':'Window/mask-seed coverage counts are not independent subjects'}
