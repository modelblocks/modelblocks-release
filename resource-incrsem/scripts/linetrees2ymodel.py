import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
import tree
import gcgtree
import semcuegraph


################################################################################

def factorConj( t ):
  if   len(t.ch)==0:                         return [ (1.0,t) ]
  elif len(t.ch)==1:                         return [ (n,tree.Tree(t.c,[st])) for n,st in factorConj(t.ch[0]) ]
  elif len(t.ch)==2 and '-lC' in t.ch[0].c:
                                             lt = factorConj( t.ch[0] ) + factorConj( t.ch[1] )
                                             tot = sum([n for n,st in lt]) if '-c' not in t.c else 1.0
                                             return [ (n/tot,tree.Tree(t.c,st.ch)) for n,st in lt ]
  elif len(t.ch)==2 and '-c' in t.c:         return factorConj( t.ch[1] )
  elif len(t.ch)==2:                         return [ (n0*n1,tree.Tree(t.c,[t0,t1])) for n0,t0 in factorConj(t.ch[0]) for n1,t1 in factorConj(t.ch[1]) ]


################################################################################

def traverse( G, Marked, m='', z='' ):
  if z=='':
    Marked = { }
    if G!={}: z = max( [ x for x,l in G ] )
  Marked[ z ] = 1
  return tree.Tree( m+'|'+z,   [ traverse( G, Marked, '-'+l, x      ) for x,l in G if G[x,l]==z and x not in Marked and l!='0' ]      ## incoming deps 
                             + [ traverse( G, Marked, l,     G[x,l] ) for x,l in G if x==z and G[x,l] not in Marked            ] )    ## outgoing deps


################################################################################

for line in sys.stdin:

  Marked = { }
  lpt = factorConj( gcgtree.GCGTree(line) ) 
  lg  = [ (p,traverse(semcuegraph.SimpleCueGraph(semcuegraph.SemCueGraph(t)),Marked)) for p,t in lpt ]

  for p,t in lg:
    print( str(t) )

