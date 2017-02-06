import sys
import re

for line in sys.stdin:
  for preterm,term in re.findall( '([^\(\)]+) ([^\(\)]+)', line ):
    for lemmasuffix,formsuffix in re.findall( '-xX[^\*]*\*([^\*]*)\*([^\*]*)', preterm ):
      print( re.sub( formsuffix+'$', lemmasuffix, term ).lower() )
    if '-xX' not in preterm: print( term.lower() )

