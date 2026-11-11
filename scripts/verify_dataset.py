"""Verify official raw MAT bytes after Git LFS checkout; no third-party deps."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root):
    manifest = json.loads((root/'recovery/2026-11-11/dataset-manifest.json').read_text())
    results=[]
    for item in manifest['files']:
        path=root/item['path']
        state='missing'
        if path.is_file():
            with path.open('rb') as f:
                prefix=f.read(128)
                f.seek(0)
                h=hashlib.sha256()
                while block:=f.read(1024*1024): h.update(block)
            if prefix.startswith(b'version https://git-lfs.github.com/spec/v1'):
                state='LFS pointer: run git lfs pull origin work'
            else:
                state='verified' if path.stat().st_size==item['bytes'] and h.hexdigest()==item['sha256'] else 'checksum/size mismatch'
        results.append({'path':item['path'],'state':state})
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=p.parse_args()
    results=verify(args.root)
    for r in results: print(f"{r['path']}: {r['state']}")
    count=sum(r['state']=='verified' for r in results)
    print(f'{count}/{len(results)} files verified')
    raise SystemExit(0 if count==18 and len(results)==18 else 1)
