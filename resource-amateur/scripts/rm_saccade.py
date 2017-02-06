import sys, argparse
import pandas as pd

argparser = argparse.ArgumentParser(description='Removes non-fixated tokens from a table of eye-tracking reading time data')
args, unknown = argparser.parse_known_args()

def main():
    input = pd.read_csv(sys.stdin, sep=' ', skipinitialspace=True)
    rm_sac = input[input['FdurFP'] > 0]
    rm_sac.to_csv(sys.stdout, ' ', index=False)
