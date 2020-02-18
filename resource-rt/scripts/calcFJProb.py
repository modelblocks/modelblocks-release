import sys, re, argparse, pandas as pd

argparser = argparse.ArgumentParser('''
Reads x-fabp parser output and appends a summed FJProb field.
''')
args, unknown = argparser.parse_known_args()

def main():
    data = pd.read_csv(sys.stdin, sep=' ', skipinitialspace=True)
    data['fjprob'] = data['F-L+']
    data['noFprob'] = data['F-L+'] + data['F-L-']
#    data['fjprob'] = data['F-L-B+'] + data['F-L+B+'] + data['F+L-B+'] + data['F+L+B+'] \
#                   + data['F-L+BNil'] + data['F-L+Badd'] + data['F-L+Bcdr']
    data.to_csv(sys.stdout, ' ', na_rep='NaN', index=False)
    
main()
