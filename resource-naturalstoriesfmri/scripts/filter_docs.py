import sys
import pandas as pd

docs = ['Boar', 'Aqua', 'MatchstickSeller', 'KingOfBirds', 'Elvis', 'MrSticky', 'HighSchool', 'Roswell', 'Tulips', 'Tourettes']

d = pd.read_csv(sys.stdin, sep=' ')

d = d[d.docid.isin(docs)]

d.to_csv(sys.stdout, sep=' ', index=False, na_rep='nan')
