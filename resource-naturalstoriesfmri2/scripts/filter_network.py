import sys
import pandas as pd

df = pd.read_csv(sys.stdin, sep=' ')
network = int(sys.argv[1])
df = df[df.network == network]
df.to_csv(sys.stdout, sep=' ', index=False, na_rep='NaN')
