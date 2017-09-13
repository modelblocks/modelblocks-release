import sys, os, re, collections
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
import tree
import gcgtree
import semcuegraph

for a in sys.argv:
  if a=='-d':
    gcgtree.VERBOSE = True
    semcuegraph.VERBOSE = True

################################################################################

for line in sys.stdin:

  t = gcgtree.GCGTree( line )
  G = semcuegraph.SemCueGraph( t )
  print( str(G) )

