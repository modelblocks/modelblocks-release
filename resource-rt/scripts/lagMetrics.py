import sys, argparse, pandas as pd
argparser = argparse.ArgumentParser('''
Adds lag metrics to user-specified columns in space-delimited data table.
''')
argparser.add_argument('-c', '--columns', dest='c', nargs='+', action='store', help='Columns for which to add metrics.')
argparser.add_argument('-n', '--numlag', dest='n', action='store', default='1', help='Number of lag positions to add.')
args, unknown = argparser.parse_known_args()
args.n = int(args.n[0])

data = pd.read_csv(sys.stdin, sep=' ', skipinitialspace=True)
for col in args.c:
    for i in range(1,args.n+1):
        name = col + 'lag' + str(i)
        if i == 1:
            data[name] = data.groupby(['subject','sentid'])[col].shift().fillna(0)
        else:
            prev_name = col + 'lag' + str(i-1)
            data[name] = data.groupby(['subject','sentid'])[prev_name].shift().fillna(0)

data.to_csv(sys.stdout, sep=' ', index=False, na_rep='NaN')
