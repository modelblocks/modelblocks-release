import sys, re, argparse, itertools
argparser = argparse.ArgumentParser('''
Compares an input linetrees file to a GCG-reannotated "target"
linetrees file, replacing all trees with newlines that failed
to reannotated in the target.
''')
argparser.add_argument('target', type=str, nargs=1, help='GCG-reannotated "target" file')
args, unknown = argparser.parse_known_args()

with open(args.target[0], 'rb') as target:
    for L1, L2 in itertools.izip(target, sys.stdin):
        if L1.strip() != '':
            print(L2.strip())
        else:
            print('')
