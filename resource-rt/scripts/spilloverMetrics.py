import sys, argparse, pandas as pd
argparser = argparse.ArgumentParser('''
Generates spillover metrics from user-supplied columns in tokmeasures input, skipping over punctuation
''')
argparser.add_argument('cols', nargs='*', help='Names of columns for which to add spillover metrics')
argparser.add_argument('-n', '--number', dest='n', action='store', nargs=1, default=[1], help='Number of spillover shifts to calculate.')
argparser.add_argument('-p', '--skippunc', dest='p', action='store_true', help='Skip punctuation when shifting for spillover calculation.')
argparser.add_argument('-a', '--allcols', dest='a', action='store_true', help='Spillover all columns.')
args, unknown = argparser.parse_known_args()
args.n = int(args.n[0])

punc = ["-LRB-", "-RRB-", ",", "-", ";", ":", "\'", "\'\'", '\"', "`", "``", ".", "!", "?", "*FOOT*"]

data = pd.read_csv(sys.stdin, sep=' ', skipinitialspace=True)
if args.a:
    cols = data.columns.values
else:
    cols = args.cols 

if args.p:
    data_no_punc = data[~data['word'].isin(punc)]

for col in cols:
    for i in range(1,args.n+1):
        s_name = col + 'S' + str(i)
        if i == 1:
            if args.p:
                data_no_punc[s_name] = data_no_punc.groupby('sentid')[col].shift()
            else:
                data[s_name] = data.groupby('sentid')[col].shift()
        else:
            prev_name = col + 'S' + str(i-1)
            if args.p:
                data_no_punc[s_name] = data_no_punc.groupby('sentid')[prev_name].shift()
            else:
                data[s_name] = data.groupby('sentid')[prev_name].shift()
        if args.p:
            data[s_name] = data_no_punc[s_name]
        if data[s_name].dtype == object:
            data[s_name].fillna('null', inplace=True)
        else:
            data[s_name].fillna(0, inplace=True)

data.to_csv(sys.stdout, sep=' ', index=False, na_rep='nan')
            

