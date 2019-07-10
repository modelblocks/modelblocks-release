import sys, os, re, collections
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
import tree
import gcgtree
import semcuegraph

RELABEL = False

for a in sys.argv:
  if a=='-d':
    gcgtree.VERBOSE = True
    semcuegraph.VERBOSE = True
  if a=='r':
    RELABEL = True

################################################################################

for line in sys.stdin:

  if RELABEL: t = gcgtree.GCGTree( line )
  else:
    t = tree.Tree( )
    t.read( line )
  G = semcuegraph.SemCueGraph( t )
  print( str(G) )

