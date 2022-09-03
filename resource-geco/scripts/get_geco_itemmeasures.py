import sys
import numpy as np
import pandas as pd

df = pd.read_csv(sys.stdin, keep_default_na=False)
df2 = pd.read_csv(sys.argv[1], header=None)
sent2trial = dict(zip(df2[0], df2[1]))

sentids = df.SENTENCE_ID.str.split('-', expand=True)
df['trialid'] = df.SENTENCE_ID.map(sent2trial)

targ_sentids = set(df2[0].unique())
src_sentids = set(df.SENTENCE_ID.unique())

df['docid'] = 'part' + sentids[0]
df['sentid'] = sentids[1].astype(int) - 1

sentid_incr = df.groupby('docid').sentid.nunique().to_dict()
_sentid_incr = {}
for i in range(1, 5):
    if i == 1:
        incr = 0
    else:
        incr = sentid_incr['part%d' % (i - 1)]
        incr += _sentid_incr['part%d' % (i - 1)]
    _sentid_incr['part%d' % i] = incr 
sentid_incr = _sentid_incr

sentid_incr = pd.DataFrame(pd.Series(sentid_incr)).reset_index()
sentid_incr = sentid_incr.rename({0: 'incr', 'index': 'docid'}, axis=1)
df = pd.merge(df, sentid_incr, on=['docid'], how='inner')
df.sentid = df.sentid + df.incr

df['word'] = df.SENTENCE.str.strip().str.split()
df = df.explode('word')
df['sentpos'] = df.groupby('sentid').cumcount() + 1
df['docpos'] = df.groupby('docid').cumcount()
df['trialpos'] = df.groupby(['docid', 'trialid']).word.cumcount() + 1

df['startofsentence'] = (df.sentid != df.sentid.shift(1)).fillna(1).astype(int)
df['endofsentence'] = df.startofsentence.shift(-1).fillna(1).astype(int)
df['startoffile'] = (df.docid != df.docid.shift(1)).fillna(1).astype(int)
df['endoffile'] = df.startoffile.shift(-1).fillna(1).astype(int)
df['startofscreen'] = (df.trialid != df.trialid.shift(1)).fillna(1).astype(int)
df['endofscreen'] = df.startofscreen.shift(-1).fillna(1).astype(int)

del df['incr']
del df['SENTENCE']
del df['SENTENCE_ID']
del df['NUMBER_WORDS_SENTENCE']

df.to_csv(sys.stdout, index=False, sep=' ', na_rep='NaN')

