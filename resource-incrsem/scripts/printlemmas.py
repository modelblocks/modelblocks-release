import sys
import re

for line in sys.stdin:
  for preterm,term in re.findall( '([^\(\)]+) ([^\(\)]+)', line ):
    term = term.lower()
    for _,formprefix,formsuffix,_,_,lemmaprefix,lemmasuffix in re.findall( '-x([^:|]*:|[^%:|]*)([^%:|]*?)%([^%:|]*)[|]([^%]?)([^-:|]*:|[^-%:|]*)([^-%:|]*?)%([^-%:|]*)', preterm ):
      term = re.sub( '^'+re.escape(formprefix)+'(.*)'+re.escape(formsuffix)+'$', lemmaprefix+'\\1'+lemmasuffix, term.lower() )
    print( term ) #.lower() )
