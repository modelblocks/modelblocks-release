#python pull_surp.py > output
import scipy.io as io
import sys

data = io.loadmat('genmodel/ESNnew.mat')
sents = data['possents']
surps = data['rawsurp_esn']

sents = sents.ravel()
for size in xrange(6):
  for variant in xrange(3):
    print 'esnsurp{:d}{:d}'.format(size+1,variant),
print

for senti in xrange(len(sents)):
  for wordi in xrange(len(sents[senti].ravel())):
    for size in xrange(6):
      print surps[senti,0][wordi,size], surps[senti,1][wordi,size], surps[senti,2][wordi,size],
    print
