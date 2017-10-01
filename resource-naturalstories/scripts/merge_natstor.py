import sys, argparse
import pandas as pd

argparser = argparse.ArgumentParser(description='Generates naturalstories.evmeasures data table.')
argparser.add_argument('f1', type=str, nargs=1, help='Path to experminet data table')
argparser.add_argument('f2', type=str, nargs=1, help='Path to itemmeasures data table')
args, unknown = argparser.parse_known_args()

def main():
    data1 = pd.read_csv(args.f1[0],sep=' ',skipinitialspace=True)
    data2 = pd.read_csv(args.f2[0],sep=' ',skipinitialspace=True)

    no_dups = [c for c in data2.columns.values if c not in data1.columns.values] + ['item', 'zone', 'word']
    data2 = data2.filter(items=no_dups)

    frames = []

    for s in data1['subject'].unique():
        data1_s = data1.loc[data1['subject'] == s]
        merged = pd.merge(data1_s, data2, how='inner', on=['item', 'zone', 'word'])
        merged['subject'] = s
        frames.append(merged)
    merged = pd.concat(frames)
    merged = merged * 1 # convert boolean to [1,0]
    merged.sort_values(['subject', 'item', 'zone'], inplace=True)
    merged.to_csv(sys.stdout, ' ', index=False, na_rep='nan')
      
main()   
