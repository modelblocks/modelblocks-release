import sys, pandas as pd

data = pd.read_csv(sys.stdin, sep=' ', skipinitialspace=True)
## Remove non-numeric columns because they can't be reduced
data = data._get_numeric_data()

exclude_cols = ['sentpos']
mean_cols = list(data.columns)
for c in exclude_cols:
    mean_cols.pop(mean_cols.index(c))


data['nwrds'] = 1
sum_cols = ['nwrds']

reduce_map = {}
for c in mean_cols:
   reduce_map[c] = 'mean'
for c in sum_cols:
   reduce_map[c] = 'sum'

data = data.groupby('sentid').agg(reduce_map)

data.to_csv(sys.stdout, ' ', index=False, na_rep='nan')


