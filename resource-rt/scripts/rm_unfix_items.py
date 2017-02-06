import sys, argparse, pandas as pd

argparser = argparse.ArgumentParser('''
Removes unfixated rows (fixation duration = 0) from a space-delimited table of reading time data. 
''')
argparser.add_argument('-f', '--fdur', dest='fdur', action='store', default='fdur', help='Name of column containing fixation duration information (defaults to "fdur")')
args, unknown = argparser.parse_known_args()

def main():
    data = pd.read_csv(sys.stdin, sep=' ', skipinitialspace=True)
    data = data[data[args.fdur] > 0]
    data.to_csv(sys.stdout, ' ', index=False, na_rep='nan')
    
main()
