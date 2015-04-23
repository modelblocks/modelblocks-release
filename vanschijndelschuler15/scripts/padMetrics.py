# python padMetrics.py
# Will pad a file (from stdin) with 0's to ensure that all rows have the same number of colummns

import sys

numcols = 0
for line in sys.stdin.readlines():
    sline = line.strip().split()
    if numcols == 0:
        numcols = len(sline)
    if len(sline) < numcols:
        #if there aren't enough columns, create as many as you need by padding them with 0 (but keep the ngrams aligned with ngrams)
        sys.stdout.write(' '.join(sline[:4]) + ' 0'*(numcols - len(sline)) + ' ' + ' '.join(sline[4:]) + '\n')
    else:
        sys.stdout.write(line)
    
