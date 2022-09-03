import sys
import numpy as np
import pandas as pd

def filter_regressions(x):
    subject = x.subject.values
    docid = x.docid.values
    trialid = x.trialid.values
    trialpos = x.trialpos.values

    key_prev = (None, None, None)
    frontier = -1
    sel = []
    for s, d, t, ix in zip(subject, docid, trialid, trialpos):
        key = (s, d, t)
        if key == key_prev:
            if ix > frontier:
                sel.append(True)
                frontier = ix
            else:
                sel.append(False)
        else:
            key_prev = key
            sel.append(True)
            frontier = ix

    sel = np.array(sel)

    x = x[sel]

    return x

df = pd.read_csv(sys.stdin, keep_default_na=False)

mapper = {
    'PP_NR': 'subject',
    'PART': 'docid',
    'TRIAL': 'trialid',
    'WORD': 'word',
    'WORD_ID_WITHIN_TRIAL': 'trialpos',
    'WORD_FIRST_FIXATION_TIME': 'time',
    'WORD_GAZE_DURATION': 'fdurFP',
    'WORD_GO_PAST_TIME': 'fdurGP',
}

cols = list(mapper.keys())

df = df[cols]
df = df.rename(mapper, axis=1)
df = df[df.time != '.']
df.trialid = df.trialid.astype(int)
df.trialpos = df.trialpos.astype(int)
df.word = df.word.str.replace('"', "'")
df.time = df.time.astype(int)
df.fdurFP = df.fdurFP.astype(int)
df.fdurGP = df.fdurGP.astype(int)

df.docid = 'part' + df.docid.astype(str)
df.time = (df.time / 1000).round(3)

df = df.sort_values(['subject', 'docid', 'trialid', 'time'])

df = filter_regressions(df)

df['wdelta'] = (df.trialpos - df.groupby(['subject', 'docid', 'trialid']).trialpos.shift(1)).fillna(0).astype(int)
df['prevwasfix'] = (df.wdelta == 1).astype(int)

df.to_csv(sys.stdout, sep=' ', index=False, na_rep='NaN')
