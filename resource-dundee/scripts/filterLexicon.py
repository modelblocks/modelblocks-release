#python scripts/filterLexicon.py wordcounts
# flags all words that are not in the wordcounts file

import sys
sys.path.insert(1, '../resource-gcg/scripts/') #grant access to model.py 
from model import Model

lexicon = Model('L')

with open(sys.argv[1],'r') as lexFile:
  for line in lexFile.readlines():
    lexicon.read(line)

FIRST = True
header = []
for line in sys.stdin.readlines():
  sline = line.strip().split()
  if FIRST:
    sys.stdout.write(line[:-1] + ' unknown\n')
    FIRST = False
    continue
  if lexicon[sline[0]] > 4:
    #known word
    sys.stdout.write(line[:-1]+ ' False\n')
  else:
    #unknown word
    sys.stdout.write(line[:-1]+ ' True\n')
