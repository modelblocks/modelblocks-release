import sys
import pandas as pd

cols = ['Participant', 'time', 'Trial', 'Sentence_id', 'PennElementType', 'CorrectAnswer', 'Fixation_duration', 'repetition_index']

def rename(x):
    if x == 'Participant':
        return 'subject'
    if x == 'Trial':
        return 'trial'
    if x == 'Sentence_id':
        return 'sentid'
    if x == 'PennElementType':
        return 'word'
    if x == 'CorrectAnswer':
        return 'correct'
    if x == 'Fixation_duration':
        return 'fdur'
    if x == 'repetition_index':
        return 'repetitionix'
    return x

df = pd.read_csv(sys.stdin)[cols].rename(rename, axis=1)
df.sentid = df.sentid.astype(int) - 1
df = df.sort_values(['subject', 'time'])
df['sentpos'] = df.groupby(['subject', 'sentid', 'repetitionix']).cumcount() + 1

df.to_csv(sys.stdout, index=False, sep=' ', na_rep='NaN')

