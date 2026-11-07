"""Explicit model-version factory; legacy configs default to V1."""
from .model import EEGModel
from .model_v2 import EEGModelV2


def build_model(config):
    common=dict(time=config['time'],dim=config['dim'],layers=config['layers'])
    version=config.get('model_version','v1')
    if version=='v1':
        return EEGModel(**common)
    if version=='v2':
        return EEGModelV2(**common,patch_size=config.get('patch_size',5))
    raise ValueError('Unknown model_version')
