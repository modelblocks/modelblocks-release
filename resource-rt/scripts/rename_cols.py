import sys, argparse, pandas as pd

argparser = argparse.ArgumentParser('''
Reads a space-delimited data table and renames user-specified columns. Column source and target names must alternate: col_1_source col_1_target [col_2_source col_2_target [etc...
''')
argparser.add_argument('cols', nargs='+', type=str, help='column names to output')
args=argparser.parse_args()
assert len(args.cols) % 2 == 0, 'Odd-numbered sequence of column names disallowed (source names and target names must alternate)'

def main():
    data = pd.read_csv(sys.stdin, sep=' ', skipinitialspace=True)
    for i in range(len(args.cols) / 2):
        data.rename(columns={args.cols[2 *i]: args.cols[2 * i + 1]}, inplace=True)
    data.to_csv(sys.stdout, ' ', index=False)
    
main()
