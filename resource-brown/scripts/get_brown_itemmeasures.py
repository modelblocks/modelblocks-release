import sys
import pandas as pd
import csv

sents = []
sentid = 0
sentpos = 0
for line in sys.stdin:
    line = line.strip().replace(' --', '--').replace('N. Y.', 'N.Y.').replace('N. H.', 'N.H.').replace('"', "'").split()
    if line:
        for word in line:
            if word != "--'":
                if word == 'Reverend!':
                    word = "Reverend!--'"
                sents.append((sentid, sentpos + 1, word))
                sentpos += 1
        if word != "Reverend!--'":
            sentid += 1
            sentpos = 0

sents = pd.DataFrame(sents, columns=['sentid', 'sentpos', 'word'])

itemmeasures = pd.read_csv(sys.argv[1])
itemmeasures.word = itemmeasures.word.str.replace(' --', '--').str.replace('N. Y.', 'N.Y.').str.replace('N. H.', 'N.H.').str.replace('"', "'")

sents['code'] = itemmeasures.code

spr = pd.read_csv(sys.argv[2])
spr = spr.drop_duplicates(['text_id', 'text_pos'])
spr = spr.sort_values(['text_id', 'text_pos']).reset_index(drop=True)
spr.word = spr.word.str.replace(' --', '--').replace('N. Y.', 'N.Y.').replace('N. H.', 'N.H.')

sents['text_id'] = spr.text_id
sents['Word_Number'] = spr.text_pos
sents['docid'] = 'd' + spr.text_id.apply(lambda x: '%02d' % x)
sents = sents.sort_values(['docid', 'sentid', 'sentpos'])
sents['startofsentence'] = (sents.sentpos == 1).astype('int')
sents['endofsentence'] = sents.startofsentence.shift(-1).fillna(1).astype('int')
sents['wlen'] = sents.word.str.len()
sents['resid'] = sents['sentpos']

sents.to_csv(sys.stdout, sep=' ', index=False, na_rep='NaN', quoting=csv.QUOTE_NONE)

