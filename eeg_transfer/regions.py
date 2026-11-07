"""Matched reconstruction contrasts. Positive changes indicate worse RMSE."""


def compare_to_a0(result,a0):
    if result['plan_sha256']!=a0['plan_sha256'] or result['protocol']!=a0['protocol']:
        raise ValueError('Groups must use identical evaluation protocol and plan')
    def change(value,base):
        if value is None or base is None or base<=0:
            return {'delta_V':None,'relative_delta':None}
        return {'delta_V':value-base,'relative_delta':value/base-1}
    overall=result['overall']; ref=a0['overall']
    contrast={'protocol':result['protocol'],'global':change(overall['rmse_V'],ref['rmse_V'])}
    for region in ('inside8','outside14'):
        a,b=overall['regions'][region],ref['regions'][region]
        if a['target_elements']!=b['target_elements']:
            raise ValueError('Mismatched target counts')
        contrast[region]=change(a['rmse_V'],b['rmse_V'])
    inside,outside=contrast['inside8']['relative_delta'],contrast['outside14']['relative_delta']
    contrast['outside_minus_inside_relative_change']=outside-inside if inside is not None and outside is not None else None
    return contrast
