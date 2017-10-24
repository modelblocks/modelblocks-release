import sys, re

docids = sys.argv[1:]

name2ix = {}
for i in range(len(docids)):
    name2ix[docids[i]] = str(i+1)

headers = sys.stdin.readline().strip().split() + ['sampleid']

docid_ix = headers.index('docid')
prev_doc = ''
sample_number = 1
print(' '.join(headers))

for l in sys.stdin:
    row = l.strip().split()
    doc_name = re.sub('_repeat.*$', '', row[docid_ix])
    doc_number = name2ix.get(doc_name, 'NaN')
    row[docid_ix] = doc_number
    if doc_name != prev_doc:
        sample_number = 1
        prev_doc = doc_name
    else:
        sample_number += 1
    sampleid = doc_name+'-'+'{0:05d}'.format(sample_number)
    row += [sampleid]
    print(' '.join(row))
