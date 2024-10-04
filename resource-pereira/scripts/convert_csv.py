import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Remove "sentences" column and change delimiter to space.')
parser.add_argument('filename', help='The CSV file name to read from and write to.')
args = parser.parse_args()

df = pd.read_csv(args.filename)
# delete the sentences col
if 'sentences' in df.columns:
    df = df.drop('sentences', axis=1)
# use space as deliminator
df.to_csv(args.filename, sep=' ', index=False)
