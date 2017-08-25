import sys, os, re, collections
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
import tree
import storestatecuegraph

#VERBOSE = False

for a in sys.argv:
  if a=='-d': storestatecuegraph.VERBOSE = True


################################################################################

for line in sys.stdin:

  line = re.sub( '-iN-gN', '-gN-iN', line )  ## script puts nonlocal deps in wrong order; higher bound nolos should be buried deeper
  line = re.sub( '-rN-vN', '-vN-rN', line )

  tr = tree.Tree( )
  tr.read( line )

  G = storestatecuegraph.StoreStateCueGraph( tr )

  for x,l in sorted(G.keys()):
    if l!='A' and l!='B' and l!='s' and l!='w' and (l!='0' or x[-1] in 'er') and l[-1]!='\'':
#':' in G[x,l]) and l[-1]!='\'':
      sys.stdout.write( ' ' + x + ',' + l + ',' + G[x,l] )
  sys.stdout.write( '\n' )



