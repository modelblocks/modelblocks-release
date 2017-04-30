import sys, argparse
import pandas as pd

argparser = argparse.ArgumentParser('''
Shifts user-specified fields from one saccade to the previous saccade in a space-delimited table of reading time data, and appends them as additional fields to the output. Rows must be in order of presentation to subjects.
''')
argparser.add_argument('-F', '--fut_cols', type=str, nargs='+', help='Names of field(s) to shift')
argparser.add_argument('-a', '--all', dest='all', action='store_true', help='Compute future versions of all input columns')
argparser.add_argument('-t', '--tok', dest='tok', action='store', default='word', help='Name of field containing token strings (defaults to "word").')
argparser.add_argument('-f', '--fdur', dest='fdur', action='store', default='fdur', help='Name of field containing fixation duration information (defaults to "fdur").')
argparser.add_argument('-S', '--subject', dest='subj', action='store', default='subject', help='Name of field containing subject ID (defaults to "subject").')
argparser.add_argument('-l', '--sentid', dest='sentid', action='store', default='sentid', help='Name of field containing sentence ID (defaults to "sentid").')
argparser.add_argument('-w', '--sentpos', dest='sentpos', action='store', default='sentpos', help='Name of field containing sentence position (defaults to "sentpos").')
argparser.add_argument('-I', '--ignoresents', dest='ignoresents', action='store_true', help='Spillover across sentence boundaries.')
argparser.add_argument('-s', '--nosubjects', dest='nosubjects', action='store_true', help='Input does not contain by-subject information.')
args, unknown = argparser.parse_known_args()
        
def main():
    data = pd.read_csv(sys.stdin,sep=' ',skipinitialspace=True)
    data.sort([args.subj, args.sentid, args.sentpos], inplace=True)
    grouped = data.groupby('subject')
    
    group_cols = []
    if not args.nosubjects:
        group_cols.append('subject')
    if not args.ignoresents:
        group_cols.append('sentid')
    if len(group_cols) > 0:
        grouped = data.groupby(group_cols)
    else:
        grouped = data

    sys.stderr.write('Computing future metrics.\n')
    
    if args.all:
        cols = data.columns.values[:]
    else:
        cols = args.fut_cols
    for col in cols:
        data['fut' + col] = grouped[col].shift(-1)
    data.to_csv(sys.stdout, ' ', index=False, na_rep='nan')
           
main()
