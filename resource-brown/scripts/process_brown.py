import sys
import pandas as pd

spr = pd.read_csv(sys.stdin)
spr = spr.rename(lambda x: 'rt' if x == 'time' else x, axis=1)
spr = spr[['subject', 'word', 'code', 'rt']]
itemmeasures = pd.read_csv(sys.argv[1], sep=' ')
del itemmeasures['word']
spr = spr.merge(itemmeasures, on='code')

spr.to_csv(sys.stdout, sep=' ', index=False, na_rep='NaN')

