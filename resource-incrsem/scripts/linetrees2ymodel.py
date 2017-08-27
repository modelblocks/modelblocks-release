import sys
import os
import numpy
import random
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

def makeTraversal( G, Marked, m='', z='' ):
  if z=='':
    Marked = { }
    if G!={}: z = max( [ x for x,l in G ] )
  Marked[ z ] = 1
  return tree.Tree( m+'|'+z,   [ makeTraversal( G, Marked, '-'+l, x      ) for x,l in G if G[x,l]==z and x not in Marked and l!='0' ]      ## incoming deps 
                             + [ makeTraversal( G, Marked, l,     G[x,l] ) for x,l in G if x==z and G[x,l] not in Marked            ] )    ## outgoing deps


################################################################################

KINTS = { }
def uniqInt( k ):
  global KINTS
  if k not in KINTS: KINTS[k]=len(KINTS)
  return KINTS[k]

################################################################################

def setKL( t, KINTS ):
  L = int( t.c.partition('|')[0] )+1 if t.c[0]!='|' else 1
  if t.c[0]=='0': uniqInt(t.c)
  for st in t.ch:
    k,l = setKL( st, KINTS )
    L = max(L,l)
  return len(KINTS),L

################################################################################

def calcBackward( t, L, M, N ):                ## bottom-up likelihood calculation
  t.u = numpy.ones((Y,1))
  for st in t.ch:
    l = int( st.c.partition('|')[0] ) if st.c[0]!='|' else 0
    calcBackward( st, L, M, N )
    if st.c[0]!='0':
      t.u = numpy.multiply( t.u, M[:,:,[L+l]].reshape((Y,Y)).dot( st.u )  )
    else:            t.u = numpy.multiply( t.u, N[:,[uniqInt(st.c)]] )

################################################################################

def sampleTypes( t, L, M, N, yAbove=0 ):        ## top-down sampling
  l = int( t.c.partition('|')[0] ) if t.c[0]!='|' else 0
  v = M[[yAbove],:,[L+l]]
  post = numpy.multiply( v, t.u.T ).reshape(Y)
  t.y = numpy.random.choice( Y, p=post/post.sum() )
  for st in t.ch:
    sampleTypes( st, L, M, N, t.y )

################################################################################

def addToCount( p, t, L, Mcount, Ncount, yAbove=0 ):
  l = int( t.c.partition('|')[0] ) if t.c[0]!='|' else 0    #int(t.c[:2] if t.c[0]=='-' else t.c[0])
  if t.c[0]=='0': Ncount[yAbove,uniqInt(t.c)] += p
  else:
    Mcount[yAbove,t.y   ,L+l] += p
    Mcount[t.y   ,yAbove,L-l] += p
  for st in t.ch:
    addToCount( p, st, L, Mcount, Ncount, t.y )

################################################################################

I = 100     ## number of iterations
Y = 7     ## number of types
L = 5       ## number of dep labels (arg positions)
beta = 0.2  ## pseudocount mass

## read in corpus and convert to traversal trees...
lptrav = [ ]
for line in sys.stdin:
  Marked = { }
  lpt = factorConj( gcgtree.GCGTree(line) ) 
  lptrav += [ (p,makeTraversal(semcuegraph.SimpleCueGraph(semcuegraph.SemCueGraph(t)),Marked)) for p,t in lpt ]

## set size params...
K,L = 0,0
for p,t in lptrav:
  K,l = setKL( t, KINTS )
  L = max(L,l)
print( str(K) + ' predicate constants, ' + str(L) + ' dependency labels.' )

## gibbs sampling iterations...
for i in range( I ):
  logprob = 0.0

  ## init counts, models...
  C, D = numpy.zeros((Y,Y,L*2)), numpy.zeros((Y,K))
  M, N = numpy.zeros((Y,Y,L*2)), numpy.zeros((Y,K))

  ## iterate over corpus...
  for p,t in lptrav:

    ## draw model from dirichlet given counts and pseudocounts...
    for l in range( 2*L ):
      for y in range( Y ):
        M[[y],:,[l]] = numpy.random.dirichlet( C[y,:,l] + beta )
    for y in range( Y ):
      N[[y],:]     = numpy.random.dirichlet( D[y,:]   + beta )

    ## draw types from model and observations...
    calcBackward ( t, L, M, N )   ## bot-up
    sampleTypes  ( t, L, M, N )   ## top-dn

    ## recalculate counts...
    C.fill(0.0)
    D.fill(0.0)
    addToCount( p, t, L, C, D )
    print( C )
    exit( 0 )

    logprob += numpy.log( p * M[0,:,0].dot( t.u ) )

  print( 'iteration ' + str(i) + ' logprob: ' + str(logprob) )

## dump models...
for l in range( 2*L ):
  for y in range( Y ):
    for z in range( Y ):
      print( 'M ' + str(l) + ' ' + str(y) + ' : ' + str(z) + ' = ' + str(M[y,z,l]) )
for l in range( 2*L ):
  for y in range( Y ):
    for k in KINTS:
      print( 'K ' + str(y) + ' : ' + k + ' = ' + str(N[y,KINTS[k]]) )

