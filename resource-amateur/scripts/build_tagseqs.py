import numpy as np
import scipy.io as io
import sys

tagmap = io.loadmat('esns/ESN.mat') #location of trained ESNs

sents = []
sent = []
for line in sys.stdin.readlines():
  sline = line.strip()
  if sline == '':
    #end of sentence
    sents.append(np.array(sent)) #print ' '.join(sent)
    sent = []
  else:
    #new word
    sent.append(sline)
if sent != []:
  sents.append(np.array(sent))
  #print ' '.join(sent)
  
tagtypes = tagmap['pos_types']

for si,s in enumerate(sents):
  sents[si] = np.array([np.where(tagtypes.ravel() == word) for word in s]).ravel()
  #print ' '.join(s)
  #print sents[si]

print 'Writing tagset'
io.savemat('genmodel/postags_test.mat',{'possents':np.array(sents)})
