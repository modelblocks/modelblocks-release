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
id_cols = ['subject', 'docid']
melted = False
if 'fROI' in responses.columns:
    id_cols.append('fROI')
    melted = True
ids = responses[id_cols]
subj_doc_pairs = ids.loc[(ids.shift() != ids).any(axis=1)]

out = []

for i in range(len(subj_doc_pairs)):
    subject = subj_doc_pairs.subject.iloc[i]
    docid = subj_doc_pairs.docid.iloc[i]
    stimuli_cur = stimuli[stimuli.docid == docid]
    stimuli_cur['subject'] = subject
    if melted:
        fROI = subj_doc_pairs.fROI.iloc[i]
        stimuli_cur['fROI'] = fROI
    out.append(stimuli_cur)

out = pd.concat(out, axis=0)

out.to_csv(sys.stdout, sep=' ', index=False, na_rep='NaN')
