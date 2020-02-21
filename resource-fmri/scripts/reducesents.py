import sys, pandas as pd

data = pd.read_csv(sys.stdin, sep=' ', skipinitialspace=True)
## Remove non-numeric columns because they can't be reduced
data = data._get_numeric_data()

del data['sentpos']

dataGrouped = data.groupby('sentid')

dataSentID = dataGrouped[['sentid']].agg('max').rename(columns = lambda x: 'sentid')
dataNwords = dataGrouped[['sentid']].agg('count').rename(columns = lambda x: 'nwrds')
dataMean = dataGrouped.agg('mean').rename(columns = lambda x: x + 'Mean')
dataSum = dataGrouped.agg('sum').rename(columns = lambda x: x + 'Sum')
dataMax = dataGrouped.agg('max').rename(columns = lambda x: x + 'Max')
data = pd.concat([dataSentID, dataNwords, dataMean, dataSum, dataMax], axis=1)

data.to_csv(sys.stdout, ' ', index=False, na_rep='NaN')


