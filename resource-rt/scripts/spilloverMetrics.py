import sys, argparse, pandas as pd
argparser = argparse.ArgumentParser('''
Generates spillover metrics from user-supplied columns in tokmeasures input, skipping over punctuation
''')
argparser.add_argument('cols', nargs='*', help='Names of columns for which to add spillover metrics')
argparser.add_argument('-n', '--number', dest='n', action='store', nargs=1, default=[1], help='Number of spillover shifts to calculate.')
argparser.add_argument('-p', '--skippunc', dest='p', action='store_true', help='Skip punctuation when shifting for spillover calculation.')
argparser.add_argument('-a', '--allcols', dest='a', action='store_true', help='Spillover all columns.')
argparser.add_argument('-I', '--ignoresents', dest='ignoresents', action='store_true', help='Spillover across sentence boundaries.')
argparser.add_argument('-s', '--nosubjects', dest='nosubjects', action='store_true', help='Input does not contain by-subject information.')
args, unknown = argparser.parse_known_args()
args.n = int(args.n[0])

punc = ["-LRB-", "-RRB-", ",", "-", ";", ":", "\'", "\'\'", '\"', "`", "``", ".", "!", "?", "*FOOT*"]

data = pd.read_csv(sys.stdin, sep=' ', skipinitialspace=True)
sys.stderr.write('Computing spillover metrics.\n')

if args.a:
    cols = data.columns.values[:]
else:
    cols = args.cols 

if args.p:
    data = data[~data['word'].isin(punc)]

group_cols = []
if not args.nosubjects:
    group_cols.append('subject')
if not args.ignoresents:
    group_cols.append('sentid')
if len(group_cols) > 0:
    grouped = data.groupby(group_cols)
else:
    grouped = data

for col in cols:
    for i in range(1,args.n+1):
        s_name = col + 'S' + str(i)
        data[s_name] = grouped[col].shift(i)
        if data[s_name].dtype == object:
            data[s_name].fillna('null', inplace=True)
        else:
            data[s_name].fillna(0, inplace=True)

data.to_csv(sys.stdout, sep=' ', index=False, na_rep='nan')
            

