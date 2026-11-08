"""Restoration must never overwrite historical provenance."""
import pytest
from eeg_transfer.suite import prepare_suite

@pytest.mark.parametrize('filename', ['rq1-manifest.json', 'commands.txt'])
def test_suite_preserves_historical_records(tmp_path, filename):
    reports=tmp_path/'reports'; reports.mkdir()
    record=reports/filename; record.write_text('historical evidence')
    with pytest.raises(FileExistsError,match='historical suite record'):
        prepare_suite('not-read.json', tmp_path/'plans', reports=reports)
    assert record.read_text()=='historical evidence'
    assert not (tmp_path/'plans').exists()


def test_restored_suite_matches_historical_scientific_plan():
    import gzip
    import json
    from pathlib import Path
    from eeg_transfer.bnci import sha256
    from eeg_transfer.plans import digest
    root=Path(__file__).resolve().parents[1]
    manifest=root/'reports/resumption-2026-11-07/suite/rq1-manifest.json'
    if not manifest.exists():
        pytest.skip('Restored suite not generated')
    current=json.loads(manifest.read_text())
    historical=json.loads((root/'reports/phase3/rq1-manifest.json').read_text())
    assert current['evaluation_hashes']==historical['evaluation_hashes']
    assert len(current['runs'])==270
    for row in current['runs']:
        assert sha256(root/row['plans'])==row['plans_sha256']
        assert sha256(root/f"data/processed/S{row['subject']:02d}.npz")==row['data_sha256']
    for subject in current['subjects']:
        for seed in current['seeds']:
            row=next(r for r in current['runs'] if r['subject']==subject and r['seed']==seed)
            with gzip.open(root/row['plans'],'rt') as f:
                plan=json.load(f)
            assert {k:digest(v) for k,v in plan['evaluation'].items()}==historical['evaluation_hashes'][str(subject)]
            groups=plan['training']['groups']; a0=groups['A0']
            for group in ('A1','A2','A3'):
                high=[e for e in groups[group] if e['montage']==22]
                assert high==a0[:len(high)]==groups[group+'_exposure']
                assert [e for e in groups[group+'_extra'] if e['montage']==22]==a0
