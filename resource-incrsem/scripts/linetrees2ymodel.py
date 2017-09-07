import sys
import os
import re
import numpy
import random
import collections
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
import tree
import gcgtree
import semcuegraph

numpy.set_printoptions(linewidth=200)

################################################################################

def numberTerminals( t, n=0 ):
  if len(t.ch)==0:
    n += 1
    t.n = n
  for st in t.ch:
    n = numberTerminals( st, n )
  return n

################################################################################

def factorConj( t ):
  if   len(t.ch)==0:                         return [ (1.0,t) ]
  elif len(t.ch)==1:                         return [ (p,tree.Tree(t.c,[st])) for p,st in factorConj(t.ch[0]) ]
  elif len(t.ch)==2 and '-lC' in t.ch[0].c:
                                             lt = factorConj( t.ch[0] ) + factorConj( t.ch[1] )
                                             tot = sum([p for p,st in lt]) if '-c' not in t.c else 1.0
                                             return [ (p/tot,tree.Tree(t.c,st.ch)) for p,st in lt ]
  elif len(t.ch)==2 and '-c' in t.c:         return factorConj( t.ch[1] )
  elif len(t.ch)==2:                         return [ (p0*p1,tree.Tree(t.c,[t0,t1])) for p0,t0 in factorConj(t.ch[0]) for p1,t1 in factorConj(t.ch[1]) ]

################################################################################

def mapFactoredToOrig( t, N2N, ctr=0 ):
  if len(t.ch)==0:
    ctr += 1
    N2N[ctr] = t.n
  for st in t.ch:
    ctr = mapFactoredToOrig( st, N2N, ctr )
  return ctr

################################################################################

def makeTraversal( G, Marked, m='', z='' ):
  if z=='':
    Marked = { }
    if G!={}: z = max( [ x for x,l in G ] )
  if m!='0': Marked[ z ] = 1
#  print( 'i am doing ' + m + ' ' + z, Marked )
  return tree.Tree( m+'|'+z,   [ makeTraversal( G, Marked, '-'+l, x      ) for x,l in G if G[x,l]==z and x not in Marked and l!='0' ]      ## incoming deps 
                             + [ makeTraversal( G, Marked, l,     G[x,l] ) for x,l in G if x==z and G[x,l] not in Marked            ] )    ## outgoing deps

################################################################################
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
    if st.c[0]!='0': calcBackward( st, L, M, N )
    l = int( st.c.partition('|')[0] ) if st.c[0]!='|' else 0
    if st.c[0]!='0': t.u = numpy.multiply( t.u, M[[L+l],:,:].reshape((Y,Y)).dot( st.u )  )
    else:            t.u = numpy.multiply( t.u, N[:,[uniqInt(st.c)]] )

################################################################################

def sampleTypes( t, L, M, N, yAbove=0 ):        ## top-down sampling
  l = int( t.c.partition('|')[0] ) if t.c[0]!='|' else 0
  post = numpy.multiply( M[[L+l],[yAbove],:], t.u.T ).reshape(Y)
  t.y = numpy.random.choice( Y, p=post/post.sum() )
  for st in t.ch:
    if st.c[0]!='0': sampleTypes( st, L, M, N, t.y )

################################################################################

def addToCount( p, t, L, Mcount, Ncount, yAbove=0 ):
  l = int( t.c.partition('|')[0] ) if t.c[0]!='|' else 0
  if t.c[0]=='0': Ncount[yAbove,uniqInt(t.c)] += p
  else:
    Mcount[L+l,yAbove,t.y   ] += p
    Mcount[L-l,t.y   ,yAbove] += p
    for st in t.ch:
      addToCount( p, st, L, Mcount, Ncount, t.y )

################################################################################
################################################################################

def averageTypeOutcomes( p, t, Ymap, N2N, src='00', srcy=-1 ):
  if t.c[0]=='0': Ymap[ N2N[int(src[:2])] ][ '-x%:' + t.c[2:] + '|%y' + str(srcy) ] += p
  for st in t.ch:
    averageTypeOutcomes( p, st, Ymap, N2N, re.sub('.*[|]','',t.c), t.y )

################################################################################

def annotY( t, Ymap, ctr=0 ):
  if len(t.ch)==1 and len(t.ch[0].ch)==0:
    ctr += 1
#    print( ctr, Ymap )
    if ctr in Ymap:
      p,xy = max( [ (Ymap[ctr][y],y) for y in Ymap[ctr] ] )
      t.c += xy
  for st in t.ch:
    ctr = annotY( st, Ymap, ctr )
  return ctr

################################################################################
################################################################################

I = 10 #100    ## number of iterations
Y = 5 #50    ## number of types
L = 5       ## number of dep labels (arg positions)
alpha = 0.1
beta  = 0.1  ## pseudocount mass

## read in corpus and convert to traversal trees...
lt       = [ ]
llpttrav = [ ]
for line in sys.stdin:
  Marked = { }
  lt.append( gcgtree.GCGTree(line) )
  numberTerminals( lt[-1] )
  lptFactored = factorConj( lt[-1] )
  llpttrav.append( [ (p,t,makeTraversal(semcuegraph.SimpleCueGraph(semcuegraph.SemCueGraph(t)),Marked)) for p,t in lptFactored ] )

## set size params...
K,L = 0,0
for lpttrav in llpttrav:
  for p,t,trav in lpttrav:
    K,l = setKL( trav, KINTS )
    L = max(L,l)
print( str(K) + ' predicate constants, ' + str(L) + ' dependency labels.' )

## init models...
M, N = numpy.zeros((L*2,Y,Y)), numpy.zeros((Y,K))
C, D = numpy.zeros((L*2,Y,Y)), numpy.zeros((Y,K))

## gibbs sampling iterations...
for i in range( I ):

  ## draw model from dirichlet given counts and pseudocounts...
  for l in range( 2*L ):
    for y in range( Y ):
      M[[l],[y],:] = numpy.random.dirichlet( C[l,y,:] + alpha )
  for y in range( Y ):
    N[[y],:] = numpy.random.dirichlet( D[y,:] + beta )

  ## init counts...
  C.fill( 0.0 )
  D.fill( 0.0 )
  logprob = 0.0

  ## iterate over corpus...
  for lpttrav in llpttrav:
    for p,t,trav in lpttrav:

      ## draw types from model and observations...
      calcBackward ( trav, L, M, N )   ## bot-up
      sampleTypes  ( trav, L, M, N )   ## top-dn

      ## recalculate counts...
      addToCount( p, trav, L, C, D )

      logprob += numpy.log( p * M[0,0,:].dot( trav.u ) )

  sys.stderr.write( 'iteration ' + str(i) + ' logprob: ' + str(logprob) + '\n' )

## dump models...
for l in range( 2*L ):
  for y in range( Y ):
    for z in range( Y ):
      print( 'M ' + str(l) + ' ' + str(y) + ' : ' + str(z) + ' = ' + str(M[l,y,z]) )
for y in range( Y ):
  for k in KINTS:
    print( 'K ' + str(y) + ' : ' + k + ' = ' + str(N[y,KINTS[k]]) )

## print types on trees...
for i,lpttrav in enumerate( llpttrav ):

  Ymap = collections.defaultdict( lambda : collections.defaultdict(float) )
  for p,t,trav in lpttrav:
    N2N = { }
    mapFactoredToOrig( t, N2N )
#    print( trav )
#    print( N2N )
    averageTypeOutcomes( p, trav, Ymap, N2N )
#    print( Ymap )

#  print( semcuegraph.SemCueGraph(lt[i]) )
  annotY( lt[i], Ymap )
  print( str(lt[i]) )


