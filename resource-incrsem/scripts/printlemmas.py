import sys
import re

for line in sys.stdin:
  for preterm,term in re.findall( '([^\(\)]+) ([^\(\)]+)', line ):
    for _,formprefix,formsuffix,_,_,lemmaprefix,lemmasuffix in re.findall( '-x([^-:|]*:|[^-%:|]*)([^-%:|]*?)%([^-%:|]*)[|](.)([^-:|]*:|[^-%:|]*)([^-%:|]*?)%([^-%:|]*)', preterm ):
      term = re.sub( '^'+formprefix+'(.*)'+formsuffix+'$', lemmaprefix+'\\1'+lemmasuffix, term )
    print( term.lower() )
