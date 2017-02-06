import pandas as pd
import sys

skip=[152]
data = pd.read_csv(sys.stdin,sep=' ',skipinitialspace=True)
gold = pd.read_csv(sys.argv[1],sep=' ',skipinitialspace=True)
goldids = gold['sentid'].unique()
data[data['sentid'].isin(goldids)].to_csv(sys.stdout, ' ', index=False)
