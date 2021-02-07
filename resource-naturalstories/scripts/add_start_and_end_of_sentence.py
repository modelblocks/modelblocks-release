import pandas as pd
import sys

data = pd.read_csv(sys.stdin,sep=' ',skipinitialspace=True)
data['startofsentence'] = (data.sentpos == 1).astype('int')
data['endofsentence'] = data.startofsentence.shift(-1).fillna(1).astype('int')
data.to_csv(sys.stdout, ' ', index=False)
