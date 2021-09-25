# TODO get rid of this script and generalize
# resource-fmri/scripts/tile_stimuli.py to allow for different id_cols
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
id_cols = ['subject', 'run']
melted = False
if 'fROI' in responses.columns:
    id_cols.append('fROI')
    melted = True
ids = responses[id_cols]
subj_run_pairs = ids.loc[(ids.shift() != ids).any(axis=1)]

out = []

for i in range(len(subj_run_pairs)):
    subject = subj_run_pairs.subject.iloc[i]
    run = subj_run_pairs.run.iloc[i]
    #stimuli_cur = stimuli[stimuli.run == run]
    stimuli_cur = stimuli.copy()
    stimuli_cur['subject'] = subject
    if melted:
        fROI = subj_run_pairs.fROI.iloc[i]
        stimuli_cur['fROI'] = fROI
    out.append(stimuli_cur)

out = pd.concat(out, axis=0)

out.to_csv(sys.stdout, sep=' ', index=False, na_rep='NaN')

