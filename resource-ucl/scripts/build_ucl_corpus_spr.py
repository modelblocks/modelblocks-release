import pandas as pd
import sys

names = ['subject','word','sentid','sentpos','fdur','wlen', 'correct', 'resid', 'startofsentence', 'endofsentence']

data = pd.read_csv(sys.stdin,sep='\t',skipinitialspace=True)
data.rename(columns={
    'subj_nr':'subject',
    'sent_nr':'sentid',
    'word_pos':'sentpos',
    'RT':'fdur',
}, inplace=True)
data['word'] = data['word'].astype(str)
data['sentid'] = data['sentid'] -1
data['sentpos'] = data['sentpos']
data['resid'] = data['sentpos']
data['wlen'] = data.apply(lambda x: len(x['word']), axis=1)
data['startofsentence'] = (data.sentpos == 1).astype('int')
data['endofsentence'] = data.startofsentence.shift(-1).fillna(1).astype('int')
data.to_csv(sys.stdout, ' ', columns=names, index=False)
