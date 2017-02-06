import pandas as pd
import sys, argparse

argparser = argparse.ArgumentParser('''
Extracts RT tokenization from table of experiment data.
''')
argparser.add_argument('cols', type=str, nargs='+', help='Column(s) that define(s) a unique key for each RT token.')
args, unknown = argparser.parse_known_args()

def main():
    data = pd.read_csv(sys.stdin,sep=' ',skipinitialspace=True)
    data.sort_values(by=args.cols, inplace=True)
    data.drop_duplicates(args.cols, inplace=True)
    data.to_csv(sys.stdout, ' ', columns=['word'] + args.cols, index=False)

main()   
