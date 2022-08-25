import sys
import pandas as pd

stories = sys.argv[1:]

def match_story(x):
    for y in stories:
        if y.lower() == x.lower():
            return y
    return x

df = pd.read_csv(sys.stdin)
df = df[df.Run == 'A']
df = df[df.Network.isin(['Lang_SN', 'MD_HE'])]
df = df[df.ROI.str.startswith('Lang_LH') | df.ROI.str.startswith('MD')]
df['subject'] = 's' + df.UID.astype(str).str.zfill(3)
df['network'] = df.Network.map({'Lang_SN': 1, 'MD_HE': 0})
df['docid'] = df.Story
df.docid = df.docid.str.replace('_T', '')
df.docid = df.docid.str.replace('_N', '')
df.docid = df.docid.str.split('_').str[-1]
df.docid = df.docid.str.capitalize()
df.docid = df.docid.apply(match_story)
df = df[df.docid.isin(stories)]
df['fROI'] = df.ROI

del df['UID']
del df['Run']
del df['Session']
del df['Network']
del df['Story']
del df['ROI']

df = pd.melt(df, id_vars=['subject', 'docid', 'network', 'fROI'], var_name='tr', value_name='BOLD')
df = df.dropna()
df.tr = df.tr.str.replace('T_', '').astype(int)
df = df[df.tr > 8]
df.tr = df.tr - 8
df['time'] = (df.tr - 1) * 2
df['splitVal15'] = (df.subject.str[1:].astype(int) + df.tr) // 15

df = df.sort_values(['subject', 'docid', 'fROI', 'time'])

df.BOLD = df.BOLD - df.groupby(['subject', 'docid', 'fROI']).BOLD.transform('mean')
df.BOLD = df.BOLD / df.groupby(['subject', 'docid', 'fROI']).BOLD.transform('std')

df.to_csv(sys.stdout, sep=' ', na_rep='NaN', index=False)
