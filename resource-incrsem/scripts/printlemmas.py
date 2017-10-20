import sys
import re

for line in sys.stdin:
  for preterm,term in re.findall( '([^\(\)]+) ([^\(\)]+)', line ):
    for _,formprefix,formsuffix,_,_,lemmaprefix,lemmasuffix in re.findall( '-x([^-:|]*:|[^-%:|]*)([^-%:|]*?)%([^-%:|]*)[|]([^%]?)([^-:|]*:|[^-%:|]*)([^-%:|]*?)%([^-%:|]*)', preterm ):
      term = re.sub( '^'+re.escape(formprefix)+'(.*)'+re.escape(formsuffix)+'$', lemmaprefix+'\\1'+lemmasuffix, term )
    print( term.lower() )
