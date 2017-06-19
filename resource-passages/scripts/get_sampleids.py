import sys

header = True

line = sys.stdin.readline()

sentid_base = 0
sentid = -1
docid_last = None

while line:
    if line != '':
        if header == True:
            h = line.strip().split()
            docid_col = h.index('docid')
            sentid_col = h.index('sentid')
            print(' '.join(h + ['sampleid']))
            header = False
        else:
            row = line.strip().split()
            docid = row[docid_col]
            if docid != docid_last:
                docid_last = docid
                sentid_base += sentid 
            sentid = int(row[sentid_col]) + 1
            sampleid = sentid_base + sentid
            print(' '.join(row + ['%s' %sampleid]))
    line = sys.stdin.readline()
