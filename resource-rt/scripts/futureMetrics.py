import sys, argparse
import pandas as pd

argparser = argparse.ArgumentParser('''
Shifts user-specified fields from one saccade to the previous saccade in a space-delimited table of reading time data, and appends them as additional fields to the output. Rows must be in order of presentation to subjects.
''')
argparser.add_argument('-F', '--fut_cols', type=str, nargs='+', help='Names of field(s) to shift')
argparser.add_argument('-a', '--all', dest='all', action='store_true', help='Compute future versions of all input columns')
argparser.add_argument('-t', '--tok', dest='tok', action='store', default='word', help='Name of field containing token strings (defaults to "word").')
argparser.add_argument('-f', '--fdur', dest='fdur', action='store', default='fdur', help='Name of field containing fixation duration information (defaults to "fdur").')
argparser.add_argument('-s', '--subject', dest='subj', action='store', default='subject', help='Name of field containing subject ID (defaults to "subject").')
argparser.add_argument('-l', '--sentid', dest='sentid', action='store', default='sentid', help='Name of field containing sentence ID (defaults to "sentid").')
argparser.add_argument('-w', '--sentpos', dest='sentpos', action='store', default='sentpos', help='Name of field containing sentence position (defaults to "sentpos").')
args, unknown = argparser.parse_known_args()
        
def main():
    data = pd.read_csv(sys.stdin,sep=' ',skipinitialspace=True)
    data.sort([args.subj, args.sentid, args.sentpos], inplace=True)
    
    sys.stderr.write('Computing future metrics.\n')
    
    if args.all:
        cols = data.columns.values[:]
    else:
        cols = args.fut_cols
    for col in cols:
        data['fut' + col] = data.groupby('subject')[col].shift(-1)
    data.to_csv(sys.stdout, ' ', index=False, na_rep='nan')
           
main()
