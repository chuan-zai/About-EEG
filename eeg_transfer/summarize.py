"""Summarize fixed complete run lists; never choose seeds by performance."""
import argparse
import json
from pathlib import Path
import numpy as np
from .regions import compare_to_a0
from .statistics import subject_inference, holm


def summarize(manifest_path,out):
    suite=json.loads(Path(manifest_path).read_text()); runs=suite['runs']; loaded={}; missing=[]
    for r in runs:
        root=Path(r['output'])
        if not (root/'evaluation.json').exists() or not (root/'status.json').exists():
            missing.append(r['id']); continue
        if json.loads((root/'status.json').read_text())['state']!='complete':
            missing.append(r['id']); continue
        m=json.loads((root/'manifest.json').read_text())
        if m['group']!=r['group'] or m['config']['seed']!=r['seed'] or m['data_sha256']!=r['data_sha256'] or m['package_sha256']!=r['plans_sha256']:
            raise ValueError('Run manifest does not match preregistered run list')
        expected_config=suite.get('suite_config',{}).get('template',{})
        if any(m['config'].get(k)!=v for k,v in expected_config.items()):
            raise ValueError('Run hyperparameters differ from locked suite')
        scores=json.loads((root/'evaluation.json').read_text())
        if 'test_random25' not in scores:
            missing.append(r['id']); continue
        expected=suite['evaluation_hashes'][str(r['subject'])]['test_random25']
        if scores['test_random25']['plan_sha256']!=expected:
            raise ValueError('Evaluation mask changed')
        loaded[(r['subject'],r['seed'],r['group'])]=(m,scores)
    result={'expected_runs':len(runs),'available_test_runs':len(loaded),'missing_runs':missing,'complete':not missing,'statistics':None,'comparisons':[]}
    # No incomplete-case inferential results: avoid accidental seed/subject selection.
    if not missing:
        effects={g:[] for g in ('A1','A2','A3')}; selection={g:[] for g in effects}
        for subject in suite['subjects']:
            per_group={g:[] for g in effects}
            for seed in suite['seeds']:
                baseline,ref=loaded[(subject,seed,'A0')]
                for g in suite['groups']:
                    m,scores=loaded[(subject,seed,g)]
                    if m['initial_weights_sha256']!=baseline['initial_weights_sha256']:
                        raise ValueError('Paired initialization differs')
                    contrast=compare_to_a0(scores['test_random25'],ref['test_random25'])
                    result['comparisons'].append({'subject':subject,'seed':seed,'group':g,'random25':contrast,'outside14':compare_to_a0(scores['test_outside14'],ref['test_outside14'])})
                    if g in per_group:
                        per_group[g].append(contrast)
            for g in effects:
                effects[g].append(float(np.mean([x['global']['delta_V'] for x in per_group[g]])))
                selection[g].append(float(np.mean([x['outside_minus_inside_relative_change'] for x in per_group[g]])))
        primary={g:subject_inference(d) for g,d in effects.items()}
        regional={g:subject_inference(d) for g,d in selection.items()}
        for family in (primary,regional):
            for g,adj in zip(family,holm([x['p_signflip_two_sided'] for x in family.values()])):
                family[g]['p_holm']=adj
        result['statistics']={'primary_3_comparisons':primary,'exploratory_region_selectivity_3_comparisons':regional,'subject_deltas_V':effects,'unit':'Subject means over all predeclared paired seeds. Windows/masks are not independent subjects.'}
    Path(out).write_text(json.dumps(result,indent=2,allow_nan=False))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--out',required=True)
    a=p.parse_args(); summarize(a.manifest,a.out)
