import sys
import pandas as pd

groupnames = sys.argv[1:]

df = pd.read_csv(sys.stdin, sep=' ')

groups = df.groupby(groupnames)

# Take means of numeric columns
means = groups.mean()

# Take first entries of all columns
first = groups.first()
nwords = groups.agg('count')['word']
mean_cols = set(means.columns)
keep_cols = [x for x in first.columns if x not in mean_cols]
first = first[keep_cols]
agg = means.join(first)
agg['nwords'] = nwords
agg.reset_index(inplace=True)
agg.to_csv(sys.stdout, sep=' ', na_rep='nan', index=False)

