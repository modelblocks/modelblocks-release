import sys, argparse, pandas as pd
argparser = argparse.ArgumentParser('''
Concatenates two or more space-delimited data tables.
''')
argparser.add_argument('tables', type=str, nargs='+', help='Paths to tables to concatenate')
args, unknown = argparser.parse_known_args()

data = []

for t in args.tables:
    data.append(pd.read_csv(t, sep=' ', skipinitialspace=True))

out = pd.concat(data)
out.to_csv(sys.stdout, ' ', na_rep='NaN', index=False)
