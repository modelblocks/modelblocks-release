import sys
import pandas as pd

df = pd.read_csv(sys.stdin, sep=' ')
times = pd.read_csv(sys.argv[1], sep=' ')

time_map = {x: y.offsettime.max() for x, y in times.groupby('docid')}

df['maxstimtime'] = df.docid.map(time_map)

df.to_csv(sys.stdout, sep=' ', index=False, na_rep='NaN')

