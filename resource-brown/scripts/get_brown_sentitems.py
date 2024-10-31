import sys
import pandas as pd

df = pd.read_csv(sys.stdin, sep=' ', keep_default_na=False)
gb = df.groupby('sentid')
curr_text_id = -1
for _, _df in gb:
    if list(_df.text_id)[0] != curr_text_id:
        print("!ARTICLE")
        curr_text_id = list(_df.text_id)[0]
    print(' '.join(list(_df.word)))
