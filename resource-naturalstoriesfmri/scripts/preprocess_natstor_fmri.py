import sys, re

docids = sys.argv[1:]

name2ix = {}
for i in range(len(docids)):
    name2ix[docids[i]] = str(i+1)

headers = sys.stdin.readline().strip().split() + ['sampleid', 'tr', 'splitVal15']

docid_ix = headers.index('docid')
subj_ix = headers.index('subject')
prev_doc = ''
sample_number = 1
print(' '.join(headers))

for l in sys.stdin:
    row = l.strip().split()
    doc_name = row[docid_ix]
    if doc_name != prev_doc:
        sample_number = 1
        prev_doc = doc_name
    else:
        sample_number += 1
    sampleid = doc_name+'-'+'{0:05d}'.format(sample_number)
    subj = row[subj_ix]
    subj_number = int(subj[1:])
    split_val_15 = int((subj_number + sample_number) / 15)
    row += [sampleid, str(sample_number), str(split_val_15)]
    print(' '.join(row))
