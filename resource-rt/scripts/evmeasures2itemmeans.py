import sys, re, argparse, pandas as pd
argparser = argparse.ArgumentParser('''Collapses user-specified numeric per-event measures from *.evmeasures input into per-item median and mean measures. Unspecified columns are assigned the first value encountered in the input data table for each item.''')
argparser.add_argument('-c', '--cols', dest='cols', nargs='+', help='Name(s) of columns to average. Names are parsed as regex, all columns that match any expression are averaged.')
args, unknown = argparser.parse_known_args()

cols_re = [re.compile(col) for col in args.cols]

events = pd.read_csv(sys.stdin, sep=' ', skipinitialspace=True)
# Initialize output table by grabbing first row for each item in the input.
# Unique items are identified by the key ['word','sentid','sentpos'].
items = events.drop_duplicates(['word','sentid','sentpos'],keep='first')
for col in events.columns.values:
    for pattern in cols_re:
        if pattern.match(col):
            mean = events.groupby(['word','sentid','sentpos'],as_index=False)[col].mean()
            meanName = col+'Mean'
            mean.rename(columns={col: meanName},inplace=True)
            median = events.groupby(['word','sentid','sentpos'],as_index=False)[col].median()
            medianName = col+'Median'
            median.rename(columns={col: medianName},inplace=True)
            # Append new calculated mean and median columns using an inner merge
            items = pd.merge(items,mean,on=['word','sentid','sentpos'],how='inner')
            items = pd.merge(items,median,on=['word','sentid','sentpos'],how='inner')
            # The following lines just ensure that the new mean and median
            # columns are inserted in the same place in the table where
            # the source column appeared. Makes the output more readable.
            columns = items.columns.tolist()
            ix = columns.index(col)
            columns.remove(meanName)
            columns.insert(ix+1,meanName)
            columns.remove(medianName)
            columns.insert(ix+2,medianName)
            items = items[columns]
            # We've appended the new summary measures, so no need to keep
            # the original source column.
            del items[col]
            # Target found, no need to keep looping through target columns
            break

#events.drop_duplicates(['word','sentid','sentpos'],keep='first',inplace=True)

items.to_csv(sys.stdout, ' ', na_rep='NaN', index=False)

