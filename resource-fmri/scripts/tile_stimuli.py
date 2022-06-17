import sys
import pandas as pd
import argparse

argparser = argparse.ArgumentParser('''
Tiles stimuli by subject into experiment data.        
''')
argparser.add_argument('stimuli', help='Path to data table containing stimuli')
argparser.add_argument('responses', help='Path to data table containing by-subject responses')
args = argparser.parse_args()

stimuli = pd.read_csv(args.stimuli, sep=' ')
responses = pd.read_csv(args.responses, sep=' ')
possible_id_cols = ['subject', 'docid', 'run', 'fROI']
id_cols = list()
for col in possible_id_cols:
    if col in responses.columns:
        id_cols.append(col)

ids = responses[id_cols]
unique_ids = ids.loc[(ids.shift() != ids).any(axis=1)]

out = []

for i in range(len(unique_ids)):
    docid = unique_ids.docid.iloc[i]
    stimuli_cur = stimuli[stimuli.docid == docid]
    for col in id_cols:
        col_val = unique_ids[col].iloc[i]
        stimuli_cur[col] = col_val
    out.append(stimuli_cur)

out = pd.concat(out, axis=0)

out.to_csv(sys.stdout, sep=' ', index=False, na_rep='NaN')
