import pandas as pd
import sys

names = ['subject','word','sentid','sentpos','fdurFP','fdurGP','wdelta','wlen', 'prevwasfix']
cumwdelta = 1
def calccumwdelta(row):
    global cumwdelta
    tmp = cumwdelta
    if row['RTfirstfix'] == 0:
        cumwdelta += 1
    else:
        cumwdelta = 1
    return tmp

data = pd.read_csv(sys.stdin,sep='\t',skipinitialspace=True)
data.rename(columns={
    'subj_nr':'subject',
    'sent_nr':'sentid',
    'word_pos':'sentpos',
    'RTfirstpass':'fdurFP',
    'RTgopast':'fdurGP',
}, inplace=True)
data['sentid'] = data['sentid'] -1
data['sentpos'] = data['sentpos']
data['wdelta'] = data.apply(calccumwdelta, axis=1)
data['wlen'] = data.apply(lambda x: len(x['word']), axis=1)
data['prevwasfix'] = data.apply(lambda x: int(x['wdelta'] == 1), axis=1)
data.to_csv(sys.stdout, ' ', columns=names, index=False)
