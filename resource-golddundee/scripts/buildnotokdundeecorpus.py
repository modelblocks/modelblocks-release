import sys
import re

finpunc = re.compile('[\.!\?][^a-zA-Z0-9]*$')

output = ''
for line in sys.stdin:
  word = line.strip()
  tmp = finpunc.search(word) #.!?
  if tmp != None:
    print(output + ' ' + word)
    output = ''
  else:
    output = output + ' ' + word
