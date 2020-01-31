import sys, argparse, pandas as pd
argparser = argparse.ArgumentParser('''
Generates spillover metrics from user-supplied columns in tokmeasures input, skipping over punctuation
''')
argparser.add_argument('cols', nargs='*', help='Names of columns for which to add spillover metrics')
argparser.add_argument('-C', '--colspillpairs', dest='colspillpairs', nargs='*', help='Space-delimited list of "<column>,<spillover>" pairs (e.g. "totsurp,3" for totsurp in spillover position 3)')
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

colspillpairs = False
if args.a:
    cols = data.columns.values[:]
elif args.colspillpairs:
    cols = [x.split(',')[0] for x in args.colspillpairs]
    spills = [int(x.split(',')[1]) for x in args.colspillpairs]
    colspillpairs = True
elif args.cols:
    cols = args.cols
    sys.stderr.write('%s' % cols)
    if len(cols) == 1:
        cols = cols[0].split('+')
else:
    cols = []

if len(cols) == 0:
    sys.stderr.write('Nothing to spillover. Returning input table.\n')
else:
    if args.p:
        data = data[~data['word'].isin(punc)]

    group_cols = ['docid']
    if not args.nosubjects:
        group_cols.append('subject')
    if not args.ignoresents:
        group_cols.append('sentid')
    grouped = data.groupby(group_cols)


for col in cols:
    if colspillpairs:
        for ix, col in enumerate(cols):
            i = spills[ix]
            s_name = col + 'S' + str(i)
            if data[col].dtype == object:
                data[s_name] = grouped[col].shift(i, fill_value='null')
            else:
                data[s_name] = grouped[col].shift(i, fill_value=0)
    else:
        for i in range(1,args.n+1):
            s_name = col + 'S' + str(i)
            if data[col].dtype == object:
                data[s_name] = grouped[col].shift(i, fill_value='null')
            else:
                data[s_name] = grouped[col].shift(i, fill_value=0)

data.to_csv(sys.stdout, sep=' ', index=False, na_rep='NaN')
            

