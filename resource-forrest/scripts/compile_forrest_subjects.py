import csv
import gzip
import sys

snum = sys.argv[1]
sampleid = 0
for fname in sys.argv[2:]:
  with gzip.open(fname) as f:
    reader = csv.reader(f)
    header = True
    for row in reader:
      if header:
        header = False
        if sampleid != 0:
          #only print header for first line
          continue
        print 'subject sampleid', ' '.join(row)
      else:
        print str(snum), str(sampleid), ' '.join(row)
      sampleid += 1
