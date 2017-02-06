import sys
import re

rword = re.compile('^[^ ]*')
#rlpar = re.compile('-LRB-')
rlpar = re.compile('[(]')
#rrpar = re.compile('-RRB-')
rrpar = re.compile('[)]')
renclitic = re.compile("([a-zA-Z]\'s)|([a-zA-Z]\'d)|([a-zA-Z]n\'t)|([a-zA-Z]\'ll)|([a-zA-Z]\'m)|([a-zA-Z]'re)|([a-zA-Z]'ve)")
rproclitic = re.compile("([yY]'[k])")

endcomma = re.compile('[^ ],[^a-zA-Z0-9]*$')
endquote = re.compile("[^ ]\'[^a-zA-Z0-9]*$")
endquotes = re.compile('[^ ]\"[^a-zA-Z0-9]*$')
endtick = re.compile('[^ ]`[^a-zA-Z0-9]*$')
endpunc = [endcomma,endquote,endquotes,endtick]
finpunc = re.compile('[\.!\?][^a-zA-Z0-9]*$')

output = ''
printit = False
for line in sys.stdin:
  keeplooking = True
  startix = 0
  word = rword.match(line).group()
  tmp = rlpar.search(word) #(
  if tmp != None:
    startix += 2 #handle ' inside parens
    word = word[:tmp.end()]+' '+word[tmp.end():]
  tmp = rrpar.search(word) #)
  if tmp != None:
    word = word[:tmp.start()]+' '+word[tmp.start():]
  tmp = renclitic.search(word) #enclitic
  if tmp != None:
    word = word[:tmp.start()+1]+' '+word[tmp.start()+1:]
  tmp = rproclitic.search(word) #proclitic
  if tmp != None:
    word = word[:tmp.end()-1]+' '+word[tmp.end()-1:]
  for reg in endpunc: #,'"`
    tmpiter = reg.finditer(word)
    if tmpiter != None:
      for tmp in tmpiter:
        word = word[:tmp.start()+1]+' '+word[tmp.start()+1:]
  tmp = finpunc.search(word) #.!?
  if tmp != None:
    word = word[:tmp.start()]+' '+word[tmp.start():]
    printit = True
  if word[-1] in '\',\";:': #problem with finditer requires this
    word = word[:-1]+' '+word[-1]
  if word[startix] in '\'`\"': #initial punc
    word = word[:startix+1]+' '+word[startix+1:]
  if word[-1] not in '.!?' and not printit:
    output = output + word + ' '

  if printit:
    output = output + word
    print(output)
    output = ''
    printit = False
