"""Restore official data using the original adapter; never train or edit old audits.

Run from any directory with the project's .venv Python. Default is inventory only.
--download explicitly enables network retrieval. Every raw file must match the
historical SHA before conversion. --out must be a fresh audit directory.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from eeg_transfer.bnci import download, prepare, sha256
from eeg_transfer.data import load_npz, split_sessions, Standardizer

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    records = []
    for subject in range(1, 10):
        old = json.loads((ROOT / f'reports/phase3/data/S{subject:02d}.json').read_text())
        row = {'subject': subject, 'files': [], 'state': 'missing'}
        for entry in old['files']:
            name = Path(entry['path']).name
            path = ROOT / 'data/raw' / name
            row['files'].append({
                'historical_path': entry['path'], 'current_path': str(path),
                'url': entry['url'], 'expected_sha256': entry['sha256'],
                'present': path.exists(),
            })
        if args.download:
            try:
                download(subject, ROOT / 'data/raw')
                for entry in row['files']:
                    entry['actual_sha256'] = sha256(entry['current_path'])
                    entry['present'] = True
                    if entry['actual_sha256'] != entry['expected_sha256']:
                        raise ValueError('Raw SHA differs; preserve file and stop conversion')
                processed = ROOT / f'data/processed/S{subject:02d}.npz'
                normalization = processed.with_name(f'S{subject:02d}-train-normalization.npz')
                if processed.exists() or normalization.exists():
                    raise FileExistsError('Refusing to overwrite existing derived data')
                current = prepare(subject, ROOT / 'data/raw', processed,
                                  out / f'S{subject:02d}.json', allow_download=False)
                # Compare scientific metadata, ignoring only paths and acquisition time.
                semantic_keys = ('runs', 'accepted_trials', 'class_counts', 'channels',
                                 'sfreq', 'scale', 'epoch', 'reject_artifacts',
                                 'nonfinite_epoch_values')
                row['session_semantics_match'] = all(
                    a[k] == b[k] for a, b in zip(current['sessions'], old['sessions'])
                    for k in semantic_keys)
                row['splits_match'] = current['splits'] == old['splits']
                row['processed_byte_sha_match'] = current['processed']['sha256'] == old['processed']['sha256']
                if not row['session_semantics_match'] or not row['splits_match']:
                    raise ValueError('Scientific metadata/splits changed; do not proceed')
                data = load_npz(processed)
                train, _, _ = split_sessions(data, seed=42)
                norm = Standardizer().fit(train)
                np.savez(normalization, mean=norm.mean, std=norm.std)
                row['state'] = 'restored_pending_integration_tests'
                # A byte mismatch remains explicit; original audits/hashes are untouched.
            except Exception as exc:
                row['state'] = 'blocked'
                row['error'] = f'{type(exc).__name__}: {exc}'
                records.append(row)
                (out / 'status.json').write_text(json.dumps(records, indent=2))
                raise
        records.append(row)
        (out / 'status.json').write_text(json.dumps(records, indent=2))
    print(f'Inventory recorded for {len(records)} subjects; download={args.download}')


if __name__ == '__main__':
    main()
