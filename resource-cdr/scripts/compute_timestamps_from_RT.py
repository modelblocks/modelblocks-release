import sys
import pandas as pd

X = pd.read_csv(sys.stdin, sep=' ', skipinitialspace=True)

if 'fdurGP' in X.columns:
    fdur = 'fdurGP'
else:
    fdur = 'fdur'

X['time'] = X.groupby(['subject', 'docid'])[fdur].shift(1).fillna(value=0)
X.time = X.groupby(['subject', 'docid']).time.cumsum() / 1000 # Convert ms to s

X.to_csv(sys.stdout, ' ', index=False, na_rep='NaN')
