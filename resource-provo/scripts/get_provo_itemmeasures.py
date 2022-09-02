import sys
import io
import numpy as np
import pandas as pd

stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='latin-1')

df = pd.read_csv(stdin)

sent_info = df[['Text_ID', 'Sentence_Number', 'Word_Number']].drop_duplicates()
gb = sent_info.groupby('Text_ID')
sent_indices = {}
i = 0
for key, _df in gb:
    indices = list(_df.Sentence_Number)
    n = indices[-1]
    indices = np.array([1] + indices) + i - 1
    i += n
    sent_indices[key] = indices

df = df.drop_duplicates(['Text_ID'])
gb = df.groupby('Text_ID')
sents = []
word = []
sentid = []
docid = []
for key, _df in gb:
    _word = _df.Text.iloc[0].split()
    _word = [x.replace('Õ', "'").replace('"', "'",) for x in _word]
    _sentid = sent_indices[key]
    if len(_sentid) == len(_word) - 1:
        if _word[0] == 'Voltaire':
            _sentid = list(_sentid)
            _sentid = [_sentid[0]] + _sentid
            _sentid = np.array(_sentid)
        else:
            _sentid = list(_sentid)
            _sentid.append(_sentid[-1])
            _sentid = np.array(_sentid)
    assert len(_word) == len(_sentid), 'Length mismatch. Got %s words and %d sentids' % (len(_word), len(_sentid))
    word.append(_word)
    sentid.append(_sentid)
    docid += ['d%s' % key] * len(_word)

word = np.concatenate(word, axis=0)
sentid = np.concatenate(sentid, axis=0)

out = pd.DataFrame({'word': word, 'docid': docid, 'sentid': sentid})
out['sentpos'] = out.groupby(sentid).cumcount() + 1
out['trial_ix'] = out.groupby(docid).cumcount()
out['startofsentence'] = (out.sentpos == 1).astype('int')
out['endofsentence'] = out.startofsentence.shift(-1).fillna(1).astype('int')

out.to_csv(sys.stdout, index=False, sep=' ', na_rep='NaN')
    


