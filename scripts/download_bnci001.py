#!/usr/bin/env python3
"""Download the 18 original BNCI2014-001 MATLAB files without modification."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen

BASE = 'https://lampx.tugraz.at/~bci/database/001-2014/'
NAMES = [f'A{s:02d}{session}.mat' for s in range(1, 10) for session in 'TE']


def validate(path):
    with path.open('rb') as stream:
        if not stream.read(128).startswith(b'MATLAB 5.0 MAT-file'):
            raise ValueError(f'{path}: not a MATLAB v5 file')
    if path.stat().st_size < 1_000_000:
        raise ValueError(f'{path}: unexpectedly small data file')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'data/BNCI2014-001')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for name in NAMES:
        dest = args.output / name
        if not dest.exists():
            partial = dest.with_suffix('.mat.part')
            try:
                print(f'Downloading {name}', flush=True)
                with urlopen(BASE + name, timeout=120) as response, partial.open('wb') as out:
                    while chunk := response.read(1024 * 1024):
                        out.write(chunk)
                validate(partial)
                partial.replace(dest)
            finally:
                partial.unlink(missing_ok=True)
        validate(dest)
        digest = hashlib.sha256()
        with dest.open('rb') as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        records.append(dict(file=name, url=BASE + name, bytes=dest.stat().st_size, sha256=digest.hexdigest()))
        print(f'Validated {name}', flush=True)
    manifest = args.output / 'manifest.json'
    manifest.write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
    print(f'Complete: {len(records)} files; checksums saved to {manifest}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(f'Download failed: {error}')
