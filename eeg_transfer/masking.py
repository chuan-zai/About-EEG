import torch
from .data import CHANNELS, EIGHT


def availability(kind, batch):
    if kind not in (8,22):
        raise ValueError('Only 8 or 22 channels')
    return torch.tensor([kind == 22 or c in EIGHT for c in CHANNELS]).expand(batch,-1).clone()


def make_batch(x, available, generator=None, fraction=.25):
    """x [B,22,T]; masks bool. Reconstruction mask is a subset of availability.
    Whole-electrode masking prevents temporal CNN access to any target position.
    Both missing electrodes and reconstruction targets are zeroed before encoder.
    Labels are deliberately absent from SSL interface.
    """
    if not 0 < fraction < 1 or available.shape != x.shape[:2]:
        raise ValueError('Invalid masking parameters')
    recon = torch.zeros_like(x,dtype=torch.bool)
    for b in range(len(x)):
        ids = available[b].nonzero().flatten()
        if len(ids) < 2:
            raise ValueError('Need at least two available electrodes')
        k = min(len(ids)-1,max(1,round(len(ids)*fraction)))
        selected = ids[torch.randperm(len(ids),generator=generator,device=x.device)[:k]]
        recon[b,selected] = True
    target = torch.where(available[...,None],x,0)
    return dict(signal=torch.where(available[...,None] & ~recon,x,0), target=target,
                channel_ids=torch.arange(22,device=x.device).expand(len(x),-1),
                availability_mask=available,reconstruction_mask=recon)


def masked_mse(pred, target, available, recon):
    mask = recon & available[...,None]
    if not mask.any():
        raise ValueError('Empty reconstruction supervision')
    return (pred[mask]-target[mask]).square().mean()


def trial_batch(trials, indices, normalizer, kind, generator=None):
    """CPU batch with provenance; metadata never enters encoder or loss."""
    av = availability(kind,len(indices))
    clean = normalizer.transform(trials.x[indices],av[0].numpy())
    batch = make_batch(torch.from_numpy(clean),av,generator=generator)
    batch.update(subject_id=trials.subject[indices].tolist(),
                 session_id=trials.session[indices].tolist(),
                 trial_id=trials.trial[indices].tolist())
    return batch
