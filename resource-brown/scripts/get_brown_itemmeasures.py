import sys
import pandas as pd

sents = []
sentid = 0
for line in sys.stdin:
    line = line.strip().replace(' --', '--').replace('N. Y.', 'N.Y.').replace('N. H.', 'N.H.').split()
    if line:
        for sentpos, word in enumerate(line):
            if sentpos != 0 or word != '--"':
                if word == 'Reverend!':
                    word = 'Reverend!--"'
                sents.append((sentid, sentpos + 1, word))
        if word != 'Reverend!--"':
            sentid += 1

sents = pd.DataFrame(sents, columns=['sentid', 'sentpos', 'word'])

itemmeasures = pd.read_csv(sys.argv[1])
itemmeasures.word = itemmeasures.word.str.replace(' --', '--').replace('N. Y.', 'N.Y.').replace('N. H.', 'N.H.')

sents['code'] = itemmeasures.code

spr = pd.read_csv(sys.argv[2])
spr = spr.drop_duplicates(['text_id', 'text_pos'])
spr = spr.sort_values(['text_id', 'text_pos']).reset_index(drop=True)
spr.word = spr.word.str.replace(' --', '--').replace('N. Y.', 'N.Y.').replace('N. H.', 'N.H.')

sents['docid'] = 'd' + spr.text_id.apply(lambda x: '%02d' % x)
sents = sents.sort_values(['docid', 'sentid', 'sentpos'])
sents['startofsentence'] = (sents.sentpos == 1).astype('int')
sents['endofsentence'] = sents.startofsentence.shift(-1).fillna(1).astype('int')
sents['wlen'] = sents.word.str.len()
sents['resid'] = sents['sentpos']

sents.to_csv(sys.stdout, sep=' ', index=False, na_rep='NaN')

