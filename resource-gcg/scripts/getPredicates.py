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
    if DEBUG:  print( lex.getFn( preterm, term ), preterm )
    else:      print( lex.getFn( preterm, term ) )

