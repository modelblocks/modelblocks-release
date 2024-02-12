import numpy as np
import pandas as pd
import sys


def interp(x):
    dtype = x.dtype
    vals = x.values
    ref = 0
    for i, v in enumerate(vals):
        if not np.isfinite(v):
            vals[i] = ref
        else:
            ref = v
    return pd.Series(x, dtype=dtype)
        
    
evmeasures_path, itemmeasures_path = sys.argv[1:]

evmeasures = pd.read_csv(evmeasures_path, sep=' ')
itemmeasures = {x: df for x, df in pd.read_csv(itemmeasures_path, sep=' ').groupby('discid')}

duration_cols = [col for col in evmeasures.columns if col.startswith('fdur')]
blink_cols = [col for col in evmeasures.columns if col.startswith('blinkduring')]

out = []
for (subject, discid), _evmeasures in evmeasures.groupby(['subject', 'discid']):
    _evmeasures = _evmeasures.sort_values('time')
    _evmeasures['offset_tmp'] = _evmeasures.time + _evmeasures.fdurGP / 1000  # fdurGP is in ms
    docid = list(_evmeasures.docid.unique())[0]
    _itemmeasures = itemmeasures[discid].copy()
    key_cols = ['word', 'discid', 'discpos']
    extra_cols = ['subject', 'time', 'offset_tmp']
    __evmeasures = _evmeasures[~_evmeasures.fdurFP.isna()]
    __evmeasures = __evmeasures[key_cols + extra_cols + blink_cols + duration_cols]
    _itemmeasures = pd.merge(
        _itemmeasures,
        __evmeasures,
        how='outer',
        on=key_cols
    )
    _itemmeasures['skippedFP'] = _itemmeasures.subject.isna().astype(int)
    _itemmeasures['subject'] = subject
    _itemmeasures['docid'] = docid
    time_interp = interp(_itemmeasures['offset_tmp']).values
    _itemmeasures['time'] = _itemmeasures.time.where(~ _itemmeasures.time.isna(), other=time_interp)
    del _itemmeasures['offset_tmp']
    del _evmeasures['offset_tmp']
    _itemmeasures = _itemmeasures[_itemmeasures.skippedFP == 1]
    for col in blink_cols:
        _itemmeasures[col] = 0
    _itemmeasures['wdelta'] = 0
    _itemmeasures['inregression'] = 0
    _out = pd.concat([_itemmeasures, _evmeasures], axis=0)
    _out.skippedFP = _out.skippedFP.where((_out.fdurFP.isna()) | (_out.skippedFP > 0), other=0)
    for col in _out.columns:
        if col.startswith('startof') or col.startswith('endof'):
            _out[col] = _out[col].fillna(0)  # Cannot infer starts/ends that are not already provided in itemmeasures 
    _out = _out.sort_values(['time', 'discpos'])
    out.append(_out)

out = pd.concat(out, axis=0)
out.to_csv(sys.stdout, sep=' ', index=False, na_rep='NaN')

