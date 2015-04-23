# python addLineNums.py
# Adds unique line numbers to each line in stdin

import re
import sys

newfile = re.compile('subject word')

ctr = 0
for line in sys.stdin.readlines():
    if newfile.match(line):
        sys.stdout.write(line.strip() + ' corpusid\n')
    else:
        sys.stdout.write(line.strip() + ' ' + str(ctr) +'\n')
    ctr += 1
