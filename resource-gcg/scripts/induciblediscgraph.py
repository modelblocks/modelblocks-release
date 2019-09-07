###############################################################################
##                                                                           ##
## This file is part of ModelBlocks. Copyright 2009, ModelBlocks developers. ##
##                                                                           ##
##    ModelBlocks is free software: you can redistribute it and/or modify    ##
##    it under the terms of the GNU General Public License as published by   ##
##    the Free Software Foundation, either version 3 of the License, or      ##
##    (at your option) any later version.                                    ##
##                                                                           ##
##    ModelBlocks is distributed in the hope that it will be useful,         ##
##    but WITHOUT ANY WARRANTY; without even the implied warranty of         ##
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          ##
##    GNU General Public License for more details.                           ##
##                                                                           ##
##    You should have received a copy of the GNU General Public License      ##
##    along with ModelBlocks.  If not, see <http://www.gnu.org/licenses/>.   ##
##                                                                           ##
###############################################################################

import sys, os, collections, sets
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
import discgraph

VERBOSE = False
for a in sys.argv:
  if a=='-d':
    VERBOSE = True

################################################################################

class InducibleDiscGraph( discgraph.DiscGraph ):


  def getChainFromSup( D, xLo ):
    return [ xLo ] + [ x for l,xHi in D.Inhs.get(xLo,{}).items() if l!='w' and l!='o' for x in D.getChainFromSup(xHi) ]
  def getChainFromSub( D, xHi ):
    return [ xHi ] + [ x for xLo in D.Subs.get(xHi,[]) for x in D.getChainFromSub(xLo) ]


  def __init__( D, line ):
    discgraph.DiscGraph.__init__( D, line )
    ## List of referents that participate in elementary predications (which does not include the elementary predication itself)...
    D.Participants = sets.Set([ x for pred in D.PredTuples for x in pred[2:] ])
    ## List of heirs for each inherited referent...
    D.Heirs = collections.defaultdict( list )
    for xHi in D.Subs:
      D.Heirs[ xHi ] = D.getHeirs( xHi )
    if VERBOSE: print( 'Heirs = ' + str(D.Heirs) )
    ## List of heirs for each participant...
    D.HeirsOfParticipants = [ xLo for xHi in D.Participants for xLo in D.Heirs.get(xHi,[]) ] 
    if VERBOSE: print( 'HeirsOfParticipants = ' + str(D.HeirsOfParticipants) )
    ## Obtain inheritance chain for each reft...
    D.Chains = { x : sets.Set( D.getChainFromSup(x) + D.getChainFromSub(x) ) for x in D.Referents }
    if VERBOSE: print( 'Chains = ' + str(D.Chains) )
#    Inheritances = { x : sets.Set( getChainFromSup(x) ) for x in Referents }
    ## Mapping from referent to elementary predications containing it...
#    D.RefToPredTuples = { xOrig : [ (ptup,xInChain)  for xInChain in D.Chains[xOrig]  for ptup in D.PredTuples  if xInChain in ptup[2:] ]  for xOrig in D.Referents }
    def orderTuplesFromSups( x ):
      Out = []
      if x in D.Nuscos:
        for src in D.Nuscos[x]:
          Out += [ (ptup,src) for ptup in D.PredTuples if src in ptup[2:] ]
      Out += [ (ptup,x) for ptup in D.PredTuples if x in ptup[2:] ]
      for lbl,dst in D.Inhs.get(x,{}).items():
        Out += orderTuplesFromSups( dst )
      return Out
    def orderTuplesFromSubs( x ):
      Out = []
      for src in D.Subs.get(x,[]):
        Out += orderTuplesFromSubs( src )
        Out += [ (ptup,src) for ptup in D.PredTuples if src in ptup[2:] ]
      return Out
    D.RefToPredTuples = { x : orderTuplesFromSubs(x) + orderTuplesFromSups(x)  for x in D.Referents }
    if VERBOSE: print( 'RefToPredTuples = ' + str(D.RefToPredTuples) )
    ## Calculate ceilings of scoped refts...
    D.AnnotatedCeilings = sets.Set([ y  for y in D.Referents  for x in D.Scopes.keys()  if D.ceiling(x) in D.Chains[y] ]) #D.Chains[D.ceiling(x)]  for x in D.Scopes.keys() ])
    if len(D.AnnotatedCeilings) == 0:
      print(           '#WARNING: Discourse contains no scope annotations -- using first word of title.' )
      sys.stderr.write( 'WARNING: Discourse contains no scope annotations -- using first word of title.\n' )
      D.AnnotatedCeilings = sets.Set([ '0001s' ])
    DisjointCeilingPairs = [ (x,y)  for x in D.AnnotatedCeilings  for y in D.AnnotatedCeilings  if x<y and not D.reachesInChain( x, y ) ]
    if len(DisjointCeilingPairs) > 0:
      print(           '#WARNING: Maxima of scopal annotations are disjoint: ' + str(DisjointCeilingPairs) + ' -- disconnected annotations cannot all be assumed dominant.' )
      sys.stderr.write( 'WARNING: Maxima of scopal annotations are disjoint: ' + str(DisjointCeilingPairs) + ' -- disconnected annotations cannot all be assumed dominant.\n' )
    if VERBOSE: print( 'AnnotatedCeilings = ' + str(D.AnnotatedCeilings) )
    D.NotOutscopable = [ x for x in D.Referents if D.ceiling(x) in D.AnnotatedCeilings ]
    if VERBOSE: print( 'NotOutscopable = ' + str(D.NotOutscopable) )
    D.PredToTuple = { xOrig : ptup  for ptup in D.PredTuples  for xOrig in D.Chains[ ptup[1] ] }
    if VERBOSE: print( 'PredToTuple = ' + str(D.PredToTuple) )
    def allInherited( src ):
      Out = []
      for lbl,dst in D.Inhs.get(src,{}).items():
        if lbl!='w' and lbl!='o':
          Out += [ dst ] + allInherited( dst )
      return Out
    D.AllInherited = { x : allInherited( x )  for x in D.Referents }
    if VERBOSE: print( 'AllInherited = ' + str(D.AllInherited) )


  def ceiling( D, x ):
    y = sorted( D.getBossesInChain(x) )[0]
    return y if y in D.NuscoValues or y not in D.Nuscos else D.Nuscos[y][0]


  def getHeirs( D, xHi ):
    Out = [ xHi ]
    for xLo in D.Subs.get(xHi,[]):
      Out += D.getHeirs( xLo )
    return Out


  ## Helper function to determine if one ref state outscopes another
  def reachesFromSup( D, xLo, xHi ):
#    print( 'reachesFromSup ' + xLo + ' ' + xHi )
    return True if xLo in D.Chains.get(xHi,[]) else D.reachesInChain( D.Scopes[xLo], xHi ) if xLo in D.Scopes else any( [ D.reachesFromSup(xSup,xHi) for l,xSup in D.Inhs.get(xLo,{}).items() if l!='w' and l!='o' ] )
  def reachesFromSub( D, xLo, xHi ):
#    print( 'reachesFromSub ' + xLo + ' ' + xHi )
    return True if xLo in D.Chains.get(xHi,[]) else D.reachesInChain( D.Scopes[xLo], xHi ) if xLo in D.Scopes else any( [ D.reachesFromSub(xSub,xHi) for xSub in D.Subs.get(xLo,[]) ] )
  def reachesInChain( D, xLo, xHi ):
#    print( 'reachesInChain ' + xLo + ' ' + xHi )
    return D.reachesFromSup( xLo, xHi ) or D.reachesFromSub( xLo, xHi )


  def satisfyPred( D, ptup, xSplice, step ):
    if VERBOSE: print( '  '*step + str(step) + ': trying to satisfy pred tuple ' + ' '.join(ptup) + ' for ' + xSplice + '...' )
    ## For unary predicates...
    if len(ptup) == 3:
      ## If elem pred already outscoped by arg, do nothing...
      if   D.reachesInChain( ptup[1], ptup[2] ):            return []
      ## If arg already outscopes splice, scope elem pred to splice...
      elif D.reachesInChain( xSplice, ptup[2] ):            return [ (ptup[1],xSplice) ] if xSplice!='' else []
      ## If arg is elem pred, recurse to that pred...
      elif ptup[2] in D.PredToTuple:                        return [ (ptup[1],ptup[2]) ] + D.satisfyPred( D.PredToTuple[ptup[2]], xSplice, step+1 )
      elif D.ceiling(ptup[2]) not in D.AnnotatedCeilings:   return [ (ptup[1],ptup[2]), (ptup[2],xSplice) ] if xSplice!='' else [ (ptup[1],ptup[2]) ] 
      elif xSplice=='' and D.ceiling(ptup[2]) in D.AnnotatedCeilings: return [ (ptup[1],ptup[2]) ]
      else:
        print(           '#ERROR: unary could not scope: ' + ' '.join(ptup) ) 
        sys.stderr.write( 'ERROR: unary could not scope: ' + ' '.join(ptup) + '\n' ) 
        exit( 1 )
    ## For binary predicates...
    if len(ptup) == 4:
#      if VERBOSE: print( '  '*step + str(step) + ': note that reach of splice, ptup3 =' + str(D.reachesInChain(xSplice,ptup[3])) + ', reach of ptup2, ptup3 =' + str(D.reachesInChain(ptup[2],ptup[3])) )
      ## If elem pred already outscoped by both args, do nothing...
      if   D.reachesInChain( ptup[1], ptup[2] ) and D.reachesInChain( ptup[1], ptup[3] ): return []
      ## If 1st arg already outscopes splice and 2nd arg already outscopes 1st arg, scope elem pred to 2nd arg...
      elif D.reachesInChain( xSplice, ptup[2] ) and D.reachesInChain( ptup[3], ptup[2] ): return [ (ptup[1],ptup[3]) ]
      ## If 2nd arg already outscopes splice and 1st arg already outscopes 2nd arg, scope elem pred to 1st arg...
      elif D.reachesInChain( xSplice, ptup[3] ) and D.reachesInChain( ptup[2], ptup[3] ): return [ (ptup[1],ptup[2]) ]
      ## If 1st arg already outscopes splice and 2nd arg already outscopes 1st arg, scope elem pred to 2nd arg...
      elif xSplice=='' and D.ceiling(ptup[2]) in D.AnnotatedCeilings and D.reachesInChain( ptup[3], ptup[2] ): return [ (ptup[1],ptup[3]) ]
      ## If 2nd arg already outscopes splice and 1st arg already outscopes 2nd arg, scope elem pred to 1st arg...
      elif xSplice=='' and D.ceiling(ptup[3]) in D.AnnotatedCeilings and D.reachesInChain( ptup[2], ptup[3] ): return [ (ptup[1],ptup[2]) ]
      ## If 1st arg already outscopes splice and 2nd arg is elem pred...
      elif D.reachesInChain( xSplice, ptup[2] ) and ptup[3] in D.PredToTuple:  return [ (ptup[1],ptup[3]) ] + D.satisfyPred( D.PredToTuple[ptup[3]], xSplice, step+1 )
      ## If 2nd arg already outscopes splice and 1st arg is elem pred...
      elif D.reachesInChain( xSplice, ptup[3] ) and ptup[2] in D.PredToTuple:  return [ (ptup[1],ptup[2]) ] + D.satisfyPred( D.PredToTuple[ptup[2]], xSplice, step+1 )
      ## If 1st arg already outscopes splice and 2nd arg is elem pred...
      elif D.reachesInChain( xSplice, ptup[2] ) and D.ceiling(ptup[3]) not in D.AnnotatedCeilings:  return [ (ptup[1],ptup[3]), (ptup[3],xSplice) ]
      ## If 2nd arg already outscopes splice and 1st arg is elem pred...
      elif D.reachesInChain( xSplice, ptup[3] ) and D.ceiling(ptup[2]) not in D.AnnotatedCeilings:  return [ (ptup[1],ptup[2]), (ptup[2],xSplice) ]
      elif D.ceiling(ptup[2]) not in D.AnnotatedCeilings and ptup[3] in D.PredToTuple:  return [ (ptup[1],ptup[2]), (ptup[2],ptup[3]) ] + D.satisfyPred( D.PredToTuple[ptup[3]], xSplice, step+1 )
      elif D.ceiling(ptup[3]) not in D.AnnotatedCeilings and ptup[2] in D.PredToTuple:  return [ (ptup[1],ptup[3]), (ptup[3],ptup[2]) ] + D.satisfyPred( D.PredToTuple[ptup[2]], xSplice, step+1 )
      elif D.ceiling(ptup[2]) not in D.AnnotatedCeilings and ptup[3] in D.getChainFromSup( ptup[2] ): return [ (ptup[1],ptup[3]), (ptup[3],xSplice) ]
      elif D.ceiling(ptup[3]) not in D.AnnotatedCeilings and ptup[2] in D.getChainFromSup( ptup[3] ): return [ (ptup[1],ptup[2]), (ptup[2],xSplice) ]
      elif D.ceiling(ptup[2]) not in D.AnnotatedCeilings and D.reachesInChain( ptup[3], D.ceiling(ptup[2]) ): return [ (ptup[1],ptup[3]), (ptup[3],xSplice) ]
      elif D.ceiling(ptup[2]) not in D.AnnotatedCeilings and D.ceiling(ptup[3]) not in D.AnnotatedCeilings: return [ (ptup[1],ptup[3]), (ptup[3],ptup[2]), (ptup[2],xSplice) ]
      else:
        print(           '#ERROR: binary predicate participants could not scope, possibly in different branches: ' + ' '.join(ptup) )
        sys.stderr.write( 'ERROR: binary predicate participants could not scope, possibly in different branches: ' + ' '.join(ptup) + '\n' )
        exit( 1 )

  def tryScope( D, RecencyConnected, step=1 ):
#    active = True
#    while active:
      if VERBOSE: print( 'RecencyConnected = ' + str(RecencyConnected) )
      active = False
      l = []
      for _,xHiOrig in RecencyConnected[:]:
        if VERBOSE: print( '  ' + D.strGraph() )
        if VERBOSE: print( '  '*step + str(step) + ': working on refstate ' + str(xHiOrig) + '...' )
        for ptup,xSplice in D.RefToPredTuples.get( xHiOrig, [] ) + ( [ ( D.PredToTuple[xHiOrig], '' ) ] if xHiOrig in D.PredToTuple else [] ):
          l = D.satisfyPred( ptup, xSplice, step+1 )
          if VERBOSE: print( '  '*step + str(step) + '  l=' + str(l) )
          for xLo,xHi in l:
            if VERBOSE: print( '  '*step + str(step) + '  scoping ' + D.ceiling(xLo) + ' to ' + xHi )
            D.Scopes[ D.ceiling(xLo) ] = xHi
            RecencyConnected = [ (step,x) for x in D.Chains.get(xLo,[]) ] + RecencyConnected
#            if VERBOSE: print( '  '*step + str(step) + '  trying ' + xLo + ' to ' + xHi + '...' )
#            for x in D.Referents:
#              if any([ y in [x] + D.AllInherited.get(x,[])  for y in [xLo] + D.AllInherited.get(xLo,[]) ]) and x not in D.Subs:   #D.reachesInChain( x, D.ceiling(xLo) ) and x not in D.Subs ]:
#                if VERBOSE: print( '  '*step + str(step) + '  because of ' + xLo + ', scoping ' + x + ' to ' + xHi )
#                D.Scopes[ D.ceiling(x) ] = xHi
#                RecencyConnected = [ (step,y) for y in D.Chains.get(x,[]) ] + RecencyConnected
          if l!=[]: D.tryScope( RecencyConnected, step+1 )
#            active = True



