import sys, pandas as pd

def main():
    data = pd.read_csv(sys.stdin, sep=' ', skipinitialspace=True)
    sys.stderr.write('Removing NaN fdurs.\n')
    fdur = None
    for name in ['fdur', 'fdurFP', 'fdurGP']:
        if name in data.columns.values:
            fdur = name
            break
    if fdur:
        data = data.loc[data[fdur].notnull()]
    else:
        sys.stderr.write('No duration column to filter on. Returning input DF.')
    if 'correct' in data.columns:
        data = data[data['correct'] > 4]
    data.to_csv(sys.stdout, ' ', index=False, na_rep='nan')
    
main()
