# cat stimuli.txt | assign_sentids.py | paste -d' ' - ngramprobtoks
#assigns sentence ids to words output by the ngram models

#stimuli.txt is a list of sentences

import sys

data = sys.stdin.readlines()
print 'sentid'
for sentid,line in enumerate(data):
  for word in line.strip().split():
    print sentid
