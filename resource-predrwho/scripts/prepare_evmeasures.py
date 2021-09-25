import sys, numpy
import pandas as pd
from collections import defaultdict

T_MIN = 1
T_MAX = 213

# wide-format dataframe
df = pd.read_csv(open(sys.argv[1]))

## Normalize each row -- this takes a while to run
print('Normalizing each timecourse...', file=sys.stderr)

for index, row in df.iterrows():
    print('\r{}/{}'.format(index+1, len(df)), file=sys.stderr, end='')
    raw_bolds = list()
    for t in range(T_MIN, T_MAX+1):
        raw_bolds.append(df.loc[index, 'T_{}'.format(t)])
    mean = numpy.mean(raw_bolds)
    sd = numpy.std(raw_bolds)
    for t in range(T_MIN, T_MAX+1):
        curr = df.loc[index, 'T_{}'.format(t)]
        df.loc[index, 'T_{}'.format(t)] = (curr-mean) / sd

print('\nReorganizing into long format...', file=sys.stderr)

# key -> ROI -> time series
data = defaultdict(dict)

for index, row in df.iterrows():
    key = (row['UID'], row['Story'], row['Run'])
    roi = row['ROI']
    roi_bold = list()
    for t in range(T_MIN, T_MAX+1):
        bold_t = row['T_{}'.format(t)]
        roi_bold.append(bold_t)
    data[key][roi] = roi_bold

# make sure the same set of per-ROI time series exists for each (UID, Story, Run)
roi_sets = [ set(data[k].keys()) for k in data ]
assert all( s == roi_sets[0] for s in roi_sets )

rois = sorted(roi_sets[0])
roi_bold_cols = ['bold_'+roi for roi in rois]
col_labels = ['subject', 'story', 'run', 'tr', 'time', 'splitVal15', 'sampleid'] + roi_bold_cols
print(' '.join(col_labels))

for key in data:
    for t in range(T_MIN, T_MAX+1):
        # t is the index of the sample. Samples are taken two seconds
        # apart
        tr = t-1
        actual_t = 2 * (t-1)
        subj_number = key[0]
        split_val_15 = int((subj_number + t) / 15)
        sample_id = '1-{}'.format(str(t).zfill(5))
        cols = list(key) + [tr, actual_t, split_val_15, sample_id]
        for roi in rois:
            cols.append(data[key][roi][t-1])
        print(' '.join(str(c) for c in cols))
