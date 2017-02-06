import sys, argparse
import pandas as pd

argparser = argparse.ArgumentParser('''
Accumulates user-specified fields over saccades in a space-delimited table of reading time data, and appends them as additional fields to the output. Rows must be in order of presentation to subjects.
''')
argparser.add_argument('cum_cols', type=str, nargs='+', help='Names of field(s) to accumulate')
argparser.add_argument('-t', '--tok', dest='tok', action='store', default='word', help='Name of field containing token strings (defaults to "word").')
argparser.add_argument('-f', '--fdur', dest='fdur', action='store', default='fdur', help='Name of field containing fixation duration information (defaults to "fdur").')
argparser.add_argument('-s', '--subject', dest='subj', action='store', default='subject', help='Name of field containing subject ID (defaults to "subject").')
argparser.add_argument('-l', '--sentid', dest='sentid', action='store', default='sentid', help='Name of field containing sentence ID (defaults to "sentid").')
argparser.add_argument('-w', '--sentpos', dest='sentpos', action='store', default='sentpos', help='Name of field containing sentence position (defaults to "sentpos").')
argparser.add_argument('-d', '--debug', dest='DEBUG', action='store_true', help='Only print accumulated fields for debugging purposes.')
args, unknown = argparser.parse_known_args()

val = 0

def accum(row, col):
    global val
    if row['sentpos'] < 2:
        val = row[col]
    else:
        val += row[col]
    out = val
    if row[args.fdur] > 0:
        val = 0
    return out
        
def main():
    global val
    data = pd.read_csv(sys.stdin,sep=' ',skipinitialspace=True)
    data.sort([args.subj, args.sentid, args.sentpos], inplace=True)
    for col in args.cum_cols:
        val = 0
        data['cum' + col] = data.apply(accum, axis=1, args=(col,))
    if args.DEBUG:
        cols = [args.tok,args.fdur]+args.cum_cols+['cum'+col for col in args.cum_cols]
        data.to_csv(sys.stdout, ' ', index=False, columns=cols)
    else:
        data.to_csv(sys.stdout, ' ', index=False, na_rep='nan')
           
main()
