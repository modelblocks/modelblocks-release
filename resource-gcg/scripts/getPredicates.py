import sys
import re
import lex

DEBUG = False
for a in sys.argv:
  if a=='-d':  DEBUG = True

## For each line...
for line in sys.stdin:
  ## For each paired preterminal and terminal...
  for preterm,term in re.findall( '\(([^\(\) ]+) ([^\(\) ]+)\)', line ):
    term = term.lower()
    out = lex.getFn( preterm, term )
    if DEBUG:  print( re.sub('^.*?:','',out), re.sub('^([A-Z])[A-Z0-9]+','\\1',out), preterm )
    else:      print( re.sub('^.*?:','',out), re.sub('^([A-Z])[A-Z0-9]+','\\1',out) )

