"""Paired subject-level inference. Seeds averaged WITHIN subject first."""
import itertools
import numpy as np


def holm(pvalues):
    p=np.asarray(pvalues,float)
    order=np.argsort(p); adjusted=np.empty(len(p)); running=0.
    for rank,index in enumerate(order):
        running=max(running,(len(p)-rank)*p[index]); adjusted[index]=min(1.,running)
    return adjusted.tolist()


def subject_inference(subject_deltas,bootstrap_seed=9173,repeats=20000,family_size=3):
    """Two-sided exact sign-flip p is conditional on sign-exchangeability/symmetry.
    Bootstrap unit = subject, not windows or training seeds. Small N=9 limits power.
    """
    d=np.asarray(subject_deltas,dtype=float)
    if len(d)<2 or len(d)>16 or not np.isfinite(d).all():
        raise ValueError('Require 2..16 finite subject effects')
    mean=float(d.mean()); rng=np.random.default_rng(bootstrap_seed)
    means=d[rng.integers(len(d),size=(repeats,len(d)))].mean(-1)
    signs=np.array(list(itertools.product((-1,1),repeat=len(d))))
    null=(signs*d).mean(-1)
    p=float((np.abs(null)>=abs(mean)-1e-15).mean())
    alpha=.05/family_size
    return {'subjects':len(d),'mean_delta':mean,'ci95_percentile':np.quantile(means,[.025,.975]).tolist(),'family_bonferroni_ci':np.quantile(means,[alpha/2,1-alpha/2]).tolist(),'family_size':family_size,'p_signflip_two_sided':p,'bootstrap_seed':bootstrap_seed,'bootstrap_repeats':repeats,'assumptions':'Subjects are independent; sign-flip requires symmetric/exchangeable paired differences. CIs are approximate with N=9.'}
