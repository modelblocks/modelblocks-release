import sys
import pandas as pd

df = pd.read_csv(sys.stdin, sep=' ')
df['startofsentence'] = df.sentpos == 1
df['endofsentence'] = df.sentpos.shift(-1).fillna(1) == 1
df.to_csv(sys.stdout, sep=' ', index=False, na_rep='NaN')
