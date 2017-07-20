import sys
import re
import storestate

UNKCOUNT = int(sys.argv[1])

WordCount = { }
for line in open(sys.argv[2]):
  count,word = line.split()
  WordCount[word] = int(count)

for line in sys.stdin:
  wordlist = re.findall('(?<= )[^\(\) ]+',line)
  for word in wordlist:
    ## if word not in WordCount:
    ##   sys.stderr.write(line)
    if word not in WordCount or WordCount[word] <= UNKCOUNT:
      line = re.sub('(?<= )'+re.escape(word)+'(?=[\(\) \n])',storestate.getUnkWord(word),line)
  sys.stdout.write( line )


'''
import sys
import codecs
import collections
import storestate

MINCOUNT = 2

Items  = [ ]
WCount  = collections.defaultdict(int)
for line in codecs.getreader('utf-8')(sys.stdin):
  cond = line.partition(' : ')[0]
  w = line.partition(' : ')[2][:-1]
  WCount[w] += 1
  Items.append((cond,w))

sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
for cond,w in Items:
  print ( cond + ' : ' + (w                           if WCount[w]>=MINCOUNT else\
                          storestate.getUnkWord(w)) )  # otherwise

#                          'unk+'+('ing' if w.endswith('ing') else\
#                                  'ed'  if w.endswith('ed') else\
#                                  's'   if w.endswith('s') else\
#                                  'cap' if w[0].isupper() else\
#                                  '')) )  # otherwise
'''

