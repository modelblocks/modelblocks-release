import sys, argparse
import pandas as pd

argparser = argparse.ArgumentParser('''
Calculates the difference between two user-specified fields in a space-delimited table of reading time data, and appends it as an additional field to the output.
''')
argparser.add_argument('cols', nargs='*', help='Names of columns to calculate delta metric from')
args = argparser.parse_args()

def main():
    data = pd.read_csv(sys.stdin, sep=' ', skipinitialspace=True)
    sys.stderr.write('Computing delta metric ({}-{}).\n'.format(args.cols[0], args.cols[1]))
    cols = args.cols
    assert len(cols) == 2, 'Exactly two columns need to be specified for the delta metric'
    data['d' + cols[0] + cols[1]] = data[cols[0]] - data[cols[1]]
    data.to_csv(sys.stdout, ' ', index=False, na_rep='NaN')

main()
