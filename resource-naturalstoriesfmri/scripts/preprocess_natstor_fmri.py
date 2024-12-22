import sys, re

docids = sys.argv[1:] # yc: doesn't seemed to be using this?

headers = sys.stdin.readline().strip().split() + ['sampleid', 'tr', 'splitVal15']

docid_ix = headers.index('docid')
subj_ix = headers.index('subject')
time_ix = headers.index('time')
print(' '.join(headers))

for l in sys.stdin:
    row = l.strip().split()
    doc_name = row[docid_ix]

    time = int(row[time_ix])
    sample_number = (time // 2) + 1 # using 2 as the step size to generate tr
    sampleid = doc_name+'-'+'{0:05d}'.format(sample_number)
    
    subj = row[subj_ix]
    subj_number = int(subj[1:])

    split_val_15 = subj_number + int((sample_number) / 15)
    row += [sampleid, str(sample_number), str(split_val_15)]
    print(' '.join(row))
