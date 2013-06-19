import sys
import re

rword = re.compile('^[^ ]*')
#rlpar = re.compile('-LRB-')
rlpar = re.compile('[(]')
#rrpar = re.compile('-RRB-')
rrpar = re.compile('[)]')
rclitic = re.compile("(\'s)|(\'d)|(n\'t)|(\'ll)")

endcomma = re.compile(',[^a-zA-Z0-9]*$')
endquote = re.compile("\'[^a-zA-Z0-9\']*$")
endquotes = re.compile('\"[^a-zA-Z0-9]*$')
endtick = re.compile('`[^a-zA-Z0-9]*$')
endpunc = [endcomma,endquote,endquotes,endtick]
finpunc = re.compile('[\.!\?][^a-zA-Z0-9]*$')

output = ''
printit = False
for line in sys.stdin:
  word = rword.match(line).group()
  tmp = rlpar.search(word) #(
  if tmp != None:
    word = word[:tmp.end()]+' '+word[tmp.end():]
  tmp = rrpar.search(word) #)
  if tmp != None:
    word = word[:tmp.start()]+' '+word[tmp.start():]
  tmp = rclitic.search(word) #clitic
  if tmp != None:
    word = word[:tmp.start()]+' '+word[tmp.start():]
  for reg in endpunc: #,'"`
    tmpiter = reg.finditer(word)
    if tmpiter != None:
      for tmp in tmpiter:
        word = word[:tmp.start()]+' '+word[tmp.start():]
  tmp = finpunc.search(word) #.!?
  if tmp != None:
    word = word[:tmp.start()]+' '+word[tmp.start():]
    printit = True
  if word[-1] in '\',\";:': #bug with finditer requires this
    word = word[:-1]+' '+word[-1]
  if word[0] in '\'`\"': #initial punc
    word = word[0]+' '+word[1:]
  if word[-1] not in '.!?' and not printit:
    output = output + word + ' '

  if printit:
    output = output + word
    print(output)
    output = ''
    printit = False
