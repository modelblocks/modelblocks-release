import sys
sys.path.insert(1, '../resource-gcg/scripts/') #grant access to model.py
from model import Model

lexicon = Model('L')
sfile = open(sys.argv[1],'r')
opts = sys.argv[2:]

unkhold = [o for o in opts if '-u' in o]
if unkhold != []:
  unkhold = int(unkhold[0][2:])
else:
  unkhold = 0

for line in sfile.readlines():
  for word in line.split():
    lexicon[word] += 1
sfile.close()

for word in lexicon:
  if lexicon[word] < unkhold:
    lexicon[word] = 0.0

lexicon.write()
