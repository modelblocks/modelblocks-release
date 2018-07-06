import sys, re

docids = sys.argv[1:]

name2ix = {}
for i in range(len(docids)):
    name2ix[docids[i]] = str(i+1)

headers = sys.stdin.readline().strip().split() + ['boldHip', 'boldHipZ', 'sampleid']

docid_ix = headers.index('docid')
LHip_ix = headers.index('boldLHip')
RHip_ix = headers.index('boldRHip')
LHipZ_ix = headers.index('boldLHipZ')
RHipZ_ix = headers.index('boldRHipZ')
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
    try:
        Hip = str(float(row[LHip_ix]) + float(row[RHip_ix]))
    except:
        Hip = 'boldHip'
    try:
        HipZ = str(float(row[LHipZ_ix]) + float(row[RHipZ_ix]))
    except:
        HipZ = 'boldHipZ' 
    row += [Hip, HipZ, sampleid]
    print(' '.join(row))
