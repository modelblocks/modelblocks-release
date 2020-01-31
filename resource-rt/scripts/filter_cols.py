import sys, re, argparse, pandas as pd

argparser = argparse.ArgumentParser('''
Reads a space-delimited data table and outputs only user-specified columns.
''')
argparser.add_argument('-c', '--cols', dest='c', nargs='+', action='store', help='column names to output')
argparser.add_argument('-x', '--exclude', dest='x', nargs='+', action='store', help='column names to drop')
argparser.add_argument('-d', '--dedup', dest='d', action='store_true', help='drop duplicate column names (keep values from the first one)')
args, unknown = argparser.parse_known_args()

def main():
    data = pd.read_csv(sys.stdin, sep=' ', skipinitialspace=True)
    if args.d:
        dups = [col for col in data.columns.values if re.search('\.[0-9]+$', col)]
        for dup in dups:
            data.drop(dup, axis=1, inplace=True)
    if args.c != None:
        data = data.filter(items=args.c)
    if args.x != None:
        for x in args.x:
            data.drop(x, axis=1, inplace=True)
    data.to_csv(sys.stdout, ' ', na_rep='NaN', index=False)
    
main()
