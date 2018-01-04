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
import morph

numpy.set_printoptions(linewidth=200)

I = 100                ## number of iterations
Y = int(sys.argv[1])   ## number of types
L = 5                  ## number of dep labels (arg positions) -- gets overridden by data
alpha = 0.1            ## pseudocount mass for word distribs
beta  = 0.1            ## pseudocount mass for transition distribs

for a in sys.argv:
  if a=='-d':
    gcgtree.VERBOSE = True
    semcuegraph.VERBOSE = True

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
uniqInt( '0|-' )     ## type for non-eventualities

################################################################################

def setKL( t, KINTS ):
  L = int( t.c.partition('|')[0] )+1 if t.c[0]!='|' else 1
  bHasPred = ( t.c[0]=='0' )
  if t.c[0]=='0': uniqInt(t.c)
  for st in t.ch:
    k,l = setKL( st, KINTS )
    L = max(L,l)
    if st.c[0]=='0': bHasPred = True
  if not bHasPred: t.ch += [ tree.Tree('0|-') ]
  return len(KINTS),L

################################################################################

def calcBackward( t, L, M, N ):                ## bottom-up likelihood calculation
  t.l = L + ( int( t.c.partition('|')[0] ) if t.c[0]!='|' else 0 )
  t.u = numpy.ones((Y,1))
  for st in t.ch:
    if st.c[0]!='0': calcBackward( st, L, M, N )
    t.u = numpy.multiply( t.u, M[st.l].dot( st.u ) if st.c[0]!='0' else N[:,[uniqInt(st.c)]] )

V0 = numpy.zeros((1,Y))
V0[0,0] = 1

def calcForward( t, M, N, vAbove=V0 ):
  t.v = vAbove
  ## cumulative from right...
  vR = numpy.ones((1,Y))
  for st in reversed( t.ch ):
    st.v = vR
    vR = numpy.multiply( vR, M[st.l].dot( st.u ).T if st.c[0]!='0' else N[:,[uniqInt(st.c)]].T )
  ## cumulative from left...
  vL = t.v.dot( M[t.l] )
  for st in t.ch:
    calcForward( st, M, N, numpy.multiply( st.v, vL ) )
    vL = numpy.multiply( vL, M[st.l].dot( st.u ).T if st.c[0]!='0' else N[:,[uniqInt(st.c)]].T )

def calcViterbi( t, LOGM, LOGN ):
  if t.c[0]=='0':
    t.u = LOGN[:,[uniqInt(t.c)]]
  else:
    tmp = numpy.copy( LOGM[t.l] )
    for st in t.ch:
      calcViterbi( st, LOGM, LOGN )
      tmp += st.u.T
    t.z = numpy.argmax ( tmp, axis=1 )    ## for each parent type, lists the root type (column number) of the best subtree
    t.u = numpy.max    ( tmp, axis=1 )    ## for each parent type, lists the likelihood of the best subtree

################################################################################

def sampleTypes( t, M, N, yAbove=0 ):        ## top-down sampling
  post = numpy.multiply( M[[t.l],[yAbove],:], t.u.T ).reshape(Y)
  t.y = numpy.random.choice( Y, p=post/post.sum() )
  for st in t.ch:
    if st.c[0]!='0': sampleTypes( st, M, N, t.y )

def maximizeTypes( t, yAbove=0 ):
  t.y = int(t.z[yAbove])
  for st in t.ch:
    if st.c[0]!='0': maximizeTypes( st, t.y )

################################################################################

def addToCount( p, t, Mcount, Ncount, yAbove=0 ):
  if t.c[0]=='0': Ncount[yAbove,uniqInt(t.c)] += p
  else:
    Mcount[t.l,    yAbove,t.y   ] += p
    Mcount[2*L-t.l,t.y   ,yAbove] += p
    for st in t.ch:
      addToCount( p, st, Mcount, Ncount, t.y )

def addToModel( p, t, M, N, C, D ):
  if t.c[0]=='0':
    contrib = t.v * N[:,uniqInt(t.c)].reshape((1,Y))
    D[:,uniqInt(t.c)] += contrib.reshape(Y) * ( p / contrib.sum() )
  else:
    contrib = numpy.multiply( t.v.T, numpy.multiply( M[t.l], t.u.T ) )
    if contrib.sum() > 0.0: contrib *= p / contrib.sum()
    C[t.l]     += contrib
    C[2*L-t.l] += contrib.T
  for st in t.ch:
    addToModel( p, st, M, N, C, D )

################################################################################
################################################################################

def mergeTypeOutcomes( p, t, Ymap, N2N, src='00', srcy=-1 ):
#  if t.c[0]=='0' and t.c!='0|-' and len(src)>2 and src[2] in 're': Ymap[ N2N[int(src[:2])] ][ '-x%' + t.c.split(':')[1] + '|%y' + str(srcy) ] += p
  if t.c[0]=='0' and t.c!='0|-' and len(src)>2 and src[2] in 're': Ymap[ N2N[int(src[:2])] ][ '-x' + t.c.split('|')[1].split(':')[0] + '%' + t.c.split(':')[1] + '|' + ('N' if t.c[2]=='N' else 'B') + '%y' + str(srcy) ] += p
  for st in t.ch:
    mergeTypeOutcomes( p, st, Ymap, N2N, re.sub('.*[|]','',t.c), t.y )

################################################################################

def annotY( t, Ymap, ctr=0 ):
  if len(t.ch)==1 and len(t.ch[0].ch)==0:
    ctr += 1
    if ctr in Ymap:
      p,xy = max( [ (Ymap[ctr][y],y) for y in Ymap[ctr] ] )
      t.c += xy
    else:
      t.c += '-x%' + morph.getLemma(t.c,t.ch[0].c) + '|%Bot' 
  for st in t.ch:
    ctr = annotY( st, Ymap, ctr )
  return ctr

################################################################################
################################################################################

## read in corpus and convert to traversal trees...
lt       = [ ]
llpttrav = [ ]
for line in sys.stdin:
  Marked = { }
  lt.append( gcgtree.GCGTree(line) )
#  print( semcuegraph.StoreStateCueGraph(lt[-1]) )
  numberTerminals( lt[-1] )
  lptFactored = factorConj( lt[-1] )
  llpttrav.append( [ (p,t,makeTraversal(semcuegraph.SimpleCueGraph(semcuegraph.SemCueGraph(t)),Marked)) for p,t in lptFactored ] )
#  for p,t in lptFactored:
#    print( '==FULL==>', str( semcuegraph.SemCueGraph(t) ) )
#    print( '--SIMPLE->', str( semcuegraph.SimpleCueGraph(semcuegraph.SemCueGraph(t)) ) )

## set size params...
K,L = 0,0
for lpttrav in llpttrav:
  for p,t,trav in lpttrav:
    K,l = setKL( trav, KINTS )
    L = max(L,l)
sys.stderr.write( str(K) + ' predicate constants, ' + str(L) + ' dependency labels.\n' )

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
      calcBackward ( trav, L, M, N )   ## bot-up (inside)
      sampleTypes  ( trav,    M, N )   ## top-dn (outside)

      ## recalculate counts...
      addToCount( p, trav, C, D )

#      logprob += numpy.log( p * M[L+0,0].dot( trav.u ) )
      logprob += numpy.log( p * M[L+0,0,:].reshape((1,Y)).dot( trav.u ) )

  sys.stderr.write( 'iteration ' + str(i) + ' logprob: ' + str(logprob) + '\n' )

## expectation maximization iterations...
for i in range( I ):
  logprob = 0.0
  ## iterate over corpus...
  for lpttrav in llpttrav:
    for p,t,trav in lpttrav:
      calcBackward ( trav, L, M, N )   ## bot-up (inside)
      calcForward  ( trav,    M, N )   ## top-dn (outside)
      logprob += numpy.log( p * trav.v.dot( M[trav.l] ).dot( trav.u ) )    ###M[L+0,0].dot( trav.u ) )   ####trav.v.dot( trav.u ) )
  sys.stderr.write( 'iteration ' + str(i) + ' logprob: ' + str(logprob) + '\n' )
  C.fill( 0.0 )
  D.fill( 0.0 )
  ## iterate over corpus...
  for lpttrav in llpttrav:
    for p,t,trav in lpttrav:
      addToModel( p, trav, M, N, C, D )
  M.fill( 0.0 )
  N.fill( 0.0 )
  for l in range( 2*L ):
    for y in range( Y ):
      denom = C[l,y].sum()
      if denom!=0.0: M[l,y] = C[l,y] / denom
#  M = C / numpy.linalg.norm(C,ord=1,axis=2)[:,:,None]
  N = D / numpy.linalg.norm(D,ord=1,axis=1)[:,None]

## dump models...
for l in range( 2*L ):
  for y in range( Y ):
    for p,z in sorted( [ (M[l,y,z],z) for z in range(Y) ], reverse=True ):   #range( Y ):
      sys.stderr.write( 'M ' + str(l) + ' ' + str(y) + ' : ' + str(z) + ' = ' + str(p) + '\n' )   #str(M[l,y,z]) )
for y in range( Y ):
  for p,k in sorted( [ (N[y,KINTS[k]],k) for k in KINTS ], reverse=True ):     #KINTS:
    sys.stderr.write( 'K ' + str(y) + ' : ' + k + ' = ' + str(p) + '\n' )    #str(N[y,KINTS[k]]) )

LOGM = numpy.log( M )
LOGN = numpy.log( N )

## print types on trees...
for i,lpttrav in enumerate( llpttrav ):

  Ymap = collections.defaultdict( lambda : collections.defaultdict(float) )
  for p,t,trav in lpttrav:

    calcViterbi   ( trav, LOGM, LOGN )
    maximizeTypes ( trav )

    N2N = { }
    mapFactoredToOrig( t, N2N )
    mergeTypeOutcomes( p, trav, Ymap, N2N )

  annotY( lt[i], Ymap )
  print( str(lt[i]) )

