import sys

model = {}

print('word unigramsurp')

with open(sys.argv[1],'r') as f:
  for line in f:
    if len(line) > 0 and line[0] == '-':
      ##only load lines that begin with logprobs
      sline = line[:-1].split('\t')
      if len(sline) > 1: model[sline[1]] = float(sline[0])

with open(sys.argv[2],'r') as f:
  for line in f:
    for word in line.strip().split():
      if word in model:
        print( word + ' ' + str(-model[word]) )
        #print '%s %s' %(word, -model[word])
      else:
        ##if we don't know a word, treat as UNK
        print( word + ' ' + str(-model['<unk>']) )
        #print '%s %s' %(word, -model['<unk>'])
