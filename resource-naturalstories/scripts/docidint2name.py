import sys
import pandas as pd
import argparse
import pdb

#int2docname = {1:'Boar', 2:'Aqua', 3:'MatchstickSeller', 4:'KingOfBirds', 5:'Elvis', 6:'MrSticky', 7:'HighSchool', 8:'Roswell', 9:'Tulips', 10:'Tourettes'}

df = pd.read_csv(sys.stdin, sep=' ')

df['docid'] = df['docid'].replace(10,'Tourettes')
df['docid'] = df['docid'].replace(9,'Tulips')
df['docid'] = df['docid'].replace(8,'Roswell')
df['docid'] = df['docid'].replace(7,'HighSchool')
df['docid'] = df['docid'].replace(6,'MrSticky')
df['docid'] = df['docid'].replace(5,'Elvis')
df['docid'] = df['docid'].replace(4,'KingOfBirds')
df['docid'] = df['docid'].replace(3,'MatchstickSeller')
df['docid'] = df['docid'].replace(2,'Aqua')
df['docid'] = df['docid'].replace(1,'Boar')

sys.stdout.write(df.to_csv(sep=' ',index=False))

