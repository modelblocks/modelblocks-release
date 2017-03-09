import sys
#python removeAt.py inputfile

def init(l):
 outstr = ""
 pctr = 0
 pstr = ""
 tag = False
 skip = False
 for cix in range(0,len(l)):
  if l[cix] == "(":
   pctr += 1
   tag = True
  elif l[cix] == ")":
   if pstr[-1] == "-":
    pstr = pstr[:-1]
   else:
    outstr += ")"
    pstr = pstr[:-1]
    
  elif tag:
   if l[cix] == '@':
    tag = False
    skip = True
    pstr += "-"
   else:
    outstr += '('+l[cix]
    pstr += "("
    tag = False
  elif skip:
   if l[cix] == " ":
    skip = False
  else:
   outstr += l[cix]
 sys.stdout.write(outstr)
 return 0

for s in sys.stdin:
 init(s)
