import pandas
import sys

df = pandas.read_csv(sys.stdin, sep=' ')
df[df["fdurGP"].notna()].to_csv(sys.stdout,sep=' ',index=False)
#df.dropna(thresh=1).to_csv(sys.stdout)
