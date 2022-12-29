import sys
import pandas as pd
import csv

spr = pd.read_csv(sys.stdin)
spr = spr.rename(lambda x: 'fdur' if x == 'time' else x, axis=1)
spr = spr[['subject', 'word', 'code', 'fdur']]
itemmeasures = pd.read_csv(sys.argv[1], sep=' ')
del itemmeasures['word']
spr = spr.merge(itemmeasures, on='code')
spr = spr.sort_values(['subject', 'docid', 'sentid', 'sentpos'])
spr['time'] = spr.groupby(['subject', 'docid']).fdur.shift(1).fillna(0.)  # For some reason chaining this line with the one below causes cumsum to spill across groups
spr['time'] = (spr.groupby(['subject', 'docid']).time.cumsum() / 1000).round(3)
spr.word = spr.word.str.replace('"', "'")

spr.to_csv(sys.stdout, sep=' ', index=False, na_rep='NaN')

