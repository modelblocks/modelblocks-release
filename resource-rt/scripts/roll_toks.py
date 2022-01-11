import re
import sys
import codecs
import numpy as np
import argparse
 
parser = argparse.ArgumentParser(description='''
"Rolls" tokens from a space-delimited input table (typically tokens used by a parser) into a target tokenization (typically tokens used in a reading time study, defined by a gold tokenization file) for which all tokens in the input are nested within (i.e. are substrings of) tokens in the target. Tokens in both input and target must be presented in linear order. When rows are rolled, items in the key column are concatenated, items in any user-specified skip columns are skipped (first value is used), and items in all other columns are summed if possible. If no numeric cast is possible in a given column, the value of the first non-null row is used.
''')
parser.add_argument('gold_path', metavar='gold', type=str, nargs=1, \
    help='Path to file containing target (gold) tokenization')
parser.add_argument('skip_cols', metavar='col', type=str, nargs='*', \
    help='Names of columns to skip in the rolling process (the value in the first row will be kept)')
# specifying the endcoding shouldn't be necessary for python 3:
# https://stackoverflow.com/questions/16549332/python-3-how-to-specify-stdin-encoding
#parser.add_argument('-e', '--enc', dest='enc', action='store', default='utf-8', \
#    help='Character encoding to use (defaults to utf-8)')
parser.add_argument('-k', '--key', dest='key', action='store', default='word', \
    help='Name of key column whose tokens will be rolled (defaults to "word")')
args = parser.parse_args()


def roll(r1, r2, skip, key):
    r2.append('1')
    r1[key] = r1[key] + r2[key]
    for i in range(1, len(r1)):
        if i not in skip:
            old = r1[i]
            new = r2[i]
            try:
                old_float = float(r1[i])
                new_float = float(r2[i])
                if np.isfinite(old_float) and np.isfinite(new_float):
                    r1[i] = str(old_float + new_float)
                elif np.isfinite(old_float):
                    r1[i] = str(old_float)
                elif np.isfinite(new_float):
                    r1[i] = str(new_float)
                else:
                    r1[i] = str(old_float + new_float)
            except ValueError:
                if old in ['None', 'null']:
                    r1[i] = r2[i]
    return r1


def main():
    gold = []
    with open(args.gold_path[0],'r') as file:
        for line in file:
            gold.append(line.strip().split(' '))
    g = 1
    header = sys.stdin.readline().rstrip().split(' ')
    print(' '.join(header) + ' rolled')
    rlen = len(header)
    skey = header.index(args.key)
    gkey = gold[0].index(args.key)
    skip = [skey] + [header.index(col) for col in args.skip_cols if col in header]
    row = None
    for i, line in enumerate(sys.stdin):
        line = line.rstrip()
        row_next = line.split(' ')
        assert len(row_next) == rlen, 'Incorrect row length: %d columns expected, %d provided.\n%s' % (rlen, len(row_next), ' '.join(row_next))
        if row == None:
            row = row_next + ['0']
        else:
            row = roll(row, row_next, skip, skey)
        assert len(row[skey]) <= len(gold[g][gkey]), 'Roll failure : %s expected, %s provided. Token line %d, item line %d.' % (gold[g][gkey].encode(args.enc), row[skey].encode(args.enc), i, g)
        if row[skey] == gold[g][gkey]:
            print(' '.join(row))
            row = None
            g += 1
            
main()
