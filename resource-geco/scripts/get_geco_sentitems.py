import sys
import pandas as pd

df = pd.read_csv(sys.stdin, sep=' ', keep_default_na=False)
gb = df.groupby('sentid')
for _, _df in gb:
    print(' '.join(list(_df.word)))
