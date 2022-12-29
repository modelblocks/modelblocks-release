import sys
import numpy as np
import pandas as pd

def fix_trial(row):
    docid = row['docid']
    trialid = row['trialid']
    sentid = row['sentid']
    sentpos = row['sentpos']
    if docid == 'part3' and trialid == 99 and sentid == 3820 and sentpos > 6:
        return 100
    return trialid

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
df['trialid'] = df.apply(fix_trial, axis=1)
df['docpos'] = df.groupby('docid').cumcount()
df['trialpos'] = df.groupby(['docid', 'trialid']).word.cumcount() + 1

offset = np.zeros(len(df), dtype=int)

ix = (df.docid == 'part1') & (df.trialid == 41) & (df.trialpos == 56)
ix = ix.argmax()
offset[ix] = 1

ix = (df.docid == 'part1') & (df.trialid == 41) & (df.trialpos == 68)
ix = ix.argmax()
offset[ix] = 1

ix = (df.docid == 'part1') & (df.trialid == 107) & (df.trialpos == 79)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part1') & (df.trialid == 129) & (df.trialpos == 27)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part1') & (df.trialid == 129) & (df.trialpos == 31)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part1') & (df.trialid == 129) & (df.trialpos == 45)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part1') & (df.trialid == 129) & (df.trialpos == 54)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part1') & (df.trialid == 130) & (df.trialpos == 85)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part1') & (df.trialid == 150) & (df.trialpos == 21)
ix = ix.argmax()
offset[ix] = 1

ix = (df.docid == 'part1') & (df.trialid == 163) & (df.trialpos == 37)
ix = ix.argmax()
offset[ix] = 2

ix = (df.docid == 'part2') & (df.trialid == 26) & (df.trialpos == 8)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part2') & (df.trialid == 57) & (df.trialpos == 44)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part2') & (df.trialid == 57) & (df.trialpos == 46)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part2') & (df.trialid == 73) & (df.trialpos == 56)
ix = ix.argmax()
offset[ix] = 1

ix = (df.docid == 'part2') & (df.trialid == 81) & (df.trialpos == 6)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part2') & (df.trialid == 81) & (df.trialpos == 60)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part2') & (df.trialid == 97) & (df.trialpos == 40)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part2') & (df.trialid == 97) & (df.trialpos == 44)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part2') & (df.trialid == 123) & (df.trialpos == 82)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part2') & (df.trialid == 123) & (df.trialpos == 84)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part2') & (df.trialid == 158) & (df.trialpos == 87)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part2') & (df.trialid == 162) & (df.trialpos == 30)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part2') & (df.trialid == 162) & (df.trialpos == 34)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part3') & (df.trialid == 10) & (df.trialpos == 23)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part3') & (df.trialid == 10) & (df.trialpos == 27)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part3') & (df.trialid == 26) & (df.trialpos == 1)
ix = ix.argmax()
offset[ix] = 1

ix = (df.docid == 'part3') & (df.trialid == 60) & (df.trialpos == 50)
ix = ix.argmax()
offset[ix] = 1

ix = (df.docid == 'part3') & (df.trialid == 60) & (df.trialpos == 59)
ix = ix.argmax()
offset[ix] = 1

ix = (df.docid == 'part3') & (df.trialid == 62) & (df.trialpos == 16)
ix = ix.argmax()
offset[ix] = 1

ix = (df.docid == 'part3') & (df.trialid == 63) & (df.trialpos == 37)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part3') & (df.trialid == 63) & (df.trialpos == 45)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part3') & (df.trialid == 63) & (df.trialpos == 95)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part3') & (df.trialid == 72) & (df.trialpos == 44)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part3') & (df.trialid == 72) & (df.trialpos == 48)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part3') & (df.trialid == 74) & (df.trialpos == 24)
ix = ix.argmax()
offset[ix] = 14

ix = (df.docid == 'part3') & (df.trialid == 74) & (df.trialpos == 30)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part3') & (df.trialid == 101) & (df.trialpos == 16)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part3') & (df.trialid == 101) & (df.trialpos == 34)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part3') & (df.trialid == 101) & (df.trialpos == 51)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part3') & (df.trialid == 101) & (df.trialpos == 83)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part3') & (df.trialid == 130) & (df.trialpos == 79)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part3') & (df.trialid == 132) & (df.trialpos == 19)
ix = ix.argmax()
offset[ix] = 1

ix = (df.docid == 'part3') & (df.trialid == 132) & (df.trialpos == 35)
ix = ix.argmax()
offset[ix] = 1

ix = (df.docid == 'part3') & (df.trialid == 136) & (df.trialpos == 45)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part3') & (df.trialid == 143) & (df.trialpos == 87)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part4') & (df.trialid == 13) & (df.trialpos == 67)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part4') & (df.trialid == 13) & (df.trialpos == 69)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part4') & (df.trialid == 52) & (df.trialpos == 20)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part4') & (df.trialid == 60) & (df.trialpos == 45)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part4') & (df.trialid == 82) & (df.trialpos == 32)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part4') & (df.trialid == 82) & (df.trialpos == 37)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part4') & (df.trialid == 82) & (df.trialpos == 38)
ix = ix.argmax()
offset[ix] = -1

ix = (df.docid == 'part4') & (df.trialid == 96) & (df.trialpos == 17)
ix = ix.argmax()
offset[ix] = 1

ix = (df.docid == 'part4') & (df.trialid == 134) & (df.trialpos == 15)
ix = ix.argmax()
offset[ix] = -1

df['offset'] = offset
df['offset'] = df.groupby(['docid', 'trialid']).offset.cumsum()
df['geco_word_id'] = df.docid.str[4:] + '-' + df.trialid.astype(str) + '-' + (df.trialpos + df.offset).astype(str)
del df['offset']

df['startofsentence'] = (df.sentid != df.sentid.shift(1)).fillna(1).astype(int)
df['endofsentence'] = df.startofsentence.shift(-1).fillna(1).astype(int)
df['startoffile'] = (df.docid != df.docid.shift(1)).fillna(1).astype(int)
df['endoffile'] = df.startoffile.shift(-1).fillna(1).astype(int)
df['startofscreen'] = (df.trialid != df.trialid.shift(1)).fillna(1).astype(int)
df['endofscreen'] = df.startofscreen.shift(-1).fillna(1).astype(int)

discid = df.docid + df.trialid.apply(lambda x: '%03d' % x)
name2discid = {x: i for i, x in enumerate(sorted(list(discid.unique())))}
discid = discid.map(name2discid)
df['discid'] = discid
df['discpos'] = df.groupby('discid').cumcount() + 1

del df['incr']
del df['SENTENCE']
del df['SENTENCE_ID']
del df['NUMBER_WORDS_SENTENCE']

df.to_csv(sys.stdout, index=False, sep=' ', na_rep='NaN')

