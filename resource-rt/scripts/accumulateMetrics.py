import sys, argparse, pandas as pd

argparser = argparse.ArgumentParser('''
Accumulates user-specified fields over saccades in a space-delimited table of reading time data, and appends them as additional fields to the output. Rows must be in order of presentation to subjects.
''')
argparser.add_argument('-a', '--all', dest='all', action='store_true', help='Accumulate all columns in input')
argparser.add_argument('-c', '--cols', type=str, nargs='+', help='Names of field(s) to accumulate')
argparser.add_argument('-t', '--tok', dest='tok', action='store', default='word', help='Name of field containing token strings (defaults to "word").')
argparser.add_argument('-s', '--subject', dest='subj', action='store', default='subject', help='Name of field containing subject ID (defaults to "subject").')
argparser.add_argument('-l', '--sentid', dest='sentid', action='store', default='sentid', help='Name of field containing sentence ID (defaults to "sentid").')
argparser.add_argument('-w', '--sentpos', dest='sentpos', action='store', default='sentpos', help='Name of field containing sentence position (defaults to "sentpos").')
argparser.add_argument('-d', '--debug', dest='DEBUG', action='store_true', help='Only print accumulated fields for debugging purposes.')
args, unknown = argparser.parse_known_args()

val = 0

def accum(row, col, fdur):
    global val
    if row['sentpos'] < 2:
        val = row[col]
    else:
        val += row[col]
    out = val
    if row[fdur] > 0:
        val = 0
    return out
        
def main():
    global val
    data = pd.read_csv(sys.stdin,sep=' ',skipinitialspace=True)
    data.sort([args.subj, args.sentid, args.sentpos], inplace=True)
    sys.stderr.write('Accumulating metrics over saccade regions.\n')

    fdur = None
    for name in ['fdur', 'fdurFP', 'fdurGP']:
        if name in data.columns.values:
            fdur = name
            break

    if fdur != None:
        if args.all:
            cols = data.columns.values[:]
        else:
            cols = args.cols
        for col in cols:
            if data[col].dtype==object:
                continue
            val = 0
            data['cum' + col] = data.apply(accum, axis=1, args=(col,fdur))
        if args.DEBUG:
            cols = [args.tok,fdur]+cols+['cum'+col for col in cols]
        else:
            cols = data.columns.values
    else:
        sys.stderr.write('No duration column. Returning input table.\n')
        cols = data.columns.values
    data.to_csv(sys.stdout, ' ', index=False, na_rep='nan', columns=cols)
           
main()
