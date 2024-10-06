import sys
import pandas as pd

df = pd.read_csv(sys.stdin, sep=' ')
gb = df.groupby("docid")
for _, df_doc in gb:
    print("!ARTICLE")
    gb_sent = df_doc.groupby("sentid")
    for _, df_sent in gb_sent:
        print(' '.join(list(df_sent.word)))
