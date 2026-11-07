"""Analytical MAC estimate of major Conv/Linear/QK/AV operations, not wall time.
Counts dense 22 slots, including masked slots. Excludes bias, GELU, layernorm,
softmax, masking/padding copies and all backward/optimizer operations.
"""
import math


def estimate(config,batch_size=4,channels=22):
    d=config['dim']; t=config['time']; layers=config['layers']; c=channels
    def encoder():
        return 8*c*d*d+2*c*c*d  # QKV/out + two FF linears + QK/AV
    def cross():
        return 4*c*d*d+2*c*c*d  # Q,K,V,out + QK/AV; all C queries/keys
    if config['model_version']=='v1':
        conv=c*t*(32*7+32*d*5)
        attention=layers*encoder()+cross()
        head=c*d*t
    elif config['model_version']=='v2':
        p=config['patch_size']; n=math.ceil(t/p)
        conv=c*n*(d*p+d*d*3)
        attention=n*(layers*encoder()+cross())
        head=n*c*d*p
    else:
        raise ValueError('Unknown architecture')
    parts={k:int(v*batch_size) for k,v in dict(convolution=conv,attention_and_ffn=attention,head=head).items()}
    total=sum(parts.values())
    return {'batch_size':batch_size,'channels':c,'major_operator_MACs':parts,'total_major_operator_MACs':total,'FLOPs_with_2_per_MAC':2*total,'scope':'Analytical dense Conv/Linear/QK/AV forward estimate only; excludes bias, GELU, norm, softmax, copies, backward and optimizer. Not complete hardware FLOPs or runtime.'}
