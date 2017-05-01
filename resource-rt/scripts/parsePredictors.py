import sys, re, argparse

argparser = argparse.ArgumentParser('''Computes accumulation, future, and spillover columns from a space-delimited input column list.''')
argparser.add_argument('-a', '--accumulate', dest='accumulate', action='store_true', help='Compute accumulation columns')
argparser.add_argument('-f', '--future', dest='future', action='store_true', help='Compute future columns')
argparser.add_argument('-s', '--spillover', dest='spillover', action='store_true', help='Compute future columns')
args, unknown = argparser.parse_known_args()

spill = re.compile('(.+)S([0-9]+)$')

preds = sys.stdin.readline().strip().split()

if args.accumulate:
    cols = []
    for col in preds:
        if col.startswith('cum'):
            cols.append(col[3:])
    print(' '.join(cols))
elif args.future:
    cols = []
    for col in preds:
        if col.startswith('fut'):
            cols.append(col[3:])
    print(' '.join(cols))
elif args.spillover:
    cols = []
    for col in preds:
        if spill.match(col):
            cols.append(','.join(spill.match(col).groups()))
    print(' '.join(cols))
else:
    for i, col in enumerate(preds):
        if col.startswith('fut'):
            col = col[3:]
        if col.startswith('cum'):
            col = col[3:]
        if spill.match(col):
            col = spill.match(col).group(1)
        preds[i] = col
    print(' '.join(preds))
    

