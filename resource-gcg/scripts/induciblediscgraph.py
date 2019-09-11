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

def complain( s ):
  print(           '#ERROR: ' + s )
  sys.stderr.write( 'ERROR: ' + s + '\n' )
  exit( 1 )


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
      print(           '#WARNING: Discourse contains no scope annotations -- using longest chain.' )
      sys.stderr.write( 'WARNING: Discourse contains no scope annotations -- using longest chain.\n' )
      D.AnnotatedCeilings = sets.Set( sorted([ (len(chain),chain)  for x,chain in D.Chains.items() ])[-1][1] )   # sets.Set(D.Chains['0001s'])
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


  ## Helper function to determine if one ref state outscopes another...
  def reachesFromSup( D, xLo, xHi ):
#    print( 'reachesFromSup ' + xLo + ' ' + xHi )
    return True if xLo in D.Chains.get(xHi,[]) else D.reachesInChain( D.Scopes[xLo], xHi ) if xLo in D.Scopes else any( [ D.reachesFromSup(xSup,xHi) for l,xSup in D.Inhs.get(xLo,{}).items() if l!='w' and l!='o' ] )
  def reachesFromSub( D, xLo, xHi ):
#    print( 'reachesFromSub ' + xLo + ' ' + xHi )
    return True if xLo in D.Chains.get(xHi,[]) else D.reachesInChain( D.Scopes[xLo], xHi ) if xLo in D.Scopes else any( [ D.reachesFromSub(xSub,xHi) for xSub in D.Subs.get(xLo,[]) ] )
  def reachesInChain( D, xLo, xHi ):
#    print( 'reachesInChain ' + xLo + ' ' + xHi )
    return D.reachesFromSup( xLo, xHi ) or D.reachesFromSub( xLo, xHi )


  ## Helper function to determine if ref state is already connected to other ref state...
  def alreadyConnected( D, x, xGoal ):
#    print( 'ceiling of ' + x + ' is ' + D.ceiling(x) )
    return ( xGoal == '' and D.ceiling( x ) in D.AnnotatedCeilings ) or D.reachesInChain( x, xGoal )


  ## Method to return list of scope (src,dst) pairs to connect target to goal...
  def scopesToConnect( D, xTarget, xGoal, step ):
    if VERBOSE: print( '  '*step + str(step) + ': trying to satisfy pred ' + xTarget + ' for ' + xGoal + '...' )

#    print( [ xSub  for xSub in D.Subs.get(xTarget,[])  if D.Inhs.get(xSub,{}).get('r','') != xTarget ] )

    ## If any non-'r' heirs, return results for heirs (elementary predicates are always final heirs)...
    if [] != [ xSub  for xSub in D.Subs.get(xTarget,[])  if D.Inhs.get(xSub,{}).get('r','') != xTarget ]:
      return [ sco  for xSub in D.Subs.get( xTarget, [] )  if D.Inhs.get(xSub,{}).get('r','') != xTarget  for sco in D.scopesToConnect( xSub, xGoal, step+1 ) ]

    ## If zero-ary (non-predicate)...
    if xTarget not in D.PredToTuple:
      if xGoal == '' or D.reachesInChain( xTarget, xGoal ): return []
      else:                                                 return [ (xTarget,xGoal) ]
#      else: complain( xTarget + ' is not already connected to goal and is not predicate, so cannot be outscoped by ' + xGoal )

    ptup = D.PredToTuple[ xTarget ]
#    ## Sanity check...
#    if ptup[1] != xTarget:
#      complain( 'too weird -- elem pred ' + xTarget + ' not equal to ptup[1]: ' + ptup[1] )
    xTarget = ptup[1]
    ## Report any cycles from participant to elementary predicate...
    for x in ptup[2:]:
      if D.reachesInChain( x, xTarget ):
        complain( 'elementary predication ' + ptup[0] + ' ' + xTarget + ' must be outscoped by argument ' + x + ' which outscopes it!' ) 
    ## If all participants reachable from elem pred, nothing to do...
    if all([ D.reachesInChain( xTarget, x )  for x in ptup[2:] ]):
      if xGoal == '' or D.reachesInChain( xTarget, xGoal ):
        return []
      else:
        complain( 'elementary predication ' + ptup[0] + ' ' + xTarget + ' is already fully bound, cannot become outscoped by goal referent ' + xGoal )

    ## If unary predicate...
    if len( ptup ) == 3:
      if D.reachesInChain( xGoal, ptup[2] ):
        if VERBOSE: print( ' ' + '  '*step + str(step) + ': case a' )
        return [ (xTarget,xGoal) ]
      else:
        if VERBOSE: print( ' ' + '  '*step + str(step) + ': case b' )
        return [ (xTarget,ptup[2]) ] + D.scopesToConnect( ptup[2], xGoal, step+1 )

    ## If binary predicate...
    elif len( ptup ) == 4:
      if D.reachesInChain( xGoal, ptup[2] ):
        if VERBOSE: print( ' ' + '  '*step + str(step) + ': case 1' )
        return [ (xTarget,ptup[3]) ] + D.scopesToConnect( ptup[3], xGoal,   step+1 )
      if D.reachesInChain( xGoal, ptup[3] ):
        if VERBOSE: print( ' ' + '  '*step + str(step) + ': case 2' )
        return [ (xTarget,ptup[2]) ] + D.scopesToConnect( ptup[2], xGoal,   step+1 )
      if D.alreadyConnected( ptup[2], xGoal ) and not D.reachesInChain( ptup[2], ptup[3] ):
        if VERBOSE: print( ' ' + '  '*step + str(step) + ': case 3' )
        return [ (xTarget,ptup[3]) ] + D.scopesToConnect( ptup[3], ptup[2], step+1 )
      if D.alreadyConnected( ptup[3], xGoal ) and not D.reachesInChain( ptup[3], ptup[2] ):
        if VERBOSE: print( ' ' + '  '*step + str(step) + ': case 4' )
        return [ (xTarget,ptup[2]) ] + D.scopesToConnect( ptup[2], ptup[3], step+1 )
      if xGoal == '' and ptup[3] in D.getHeirs( ptup[2] ):
        if VERBOSE: print( ' ' + '  '*step + str(step) + ': case 5' )
        return [ (xTarget,ptup[3]) ] + D.scopesToConnect( ptup[3], ptup[2], step+1 )
      if xGoal == '':
        if VERBOSE: print( ' ' + '  '*step + str(step) + ': case 6' )
        return [ (xTarget,ptup[2]) ] + D.scopesToConnect( ptup[2], ptup[3], step+1 )
      else:
        if VERBOSE: print( ' ' + '  '*step + str(step) + ': case 7' )
        return [ (xTarget,ptup[2]) ] + D.scopesToConnect( ptup[2], ptup[3], step+1 ) + D.scopesToConnect( ptup[3], xGoal, step+1 )
#complain( 'predicate ' + xTarget + ' with goal ' + xGoal + ' not sufficiently constrained; danger of garden-pathing' )

    ## If trinary and higher predicates...
    else:
      complain( 'no support for super-binary predicates: ' + ' '.join(ptup) )


  ## Method to fill in deterministic or truth-functionally indistinguishable scope associations (e.g. for elementary predications) that are not explicitly annotated...
  def tryScope( D, RecencyConnected, step=1 ):
      if VERBOSE: print( 'RecencyConnected = ' + str(RecencyConnected) )
      active = False
      l = []
      for _,xHiOrig in RecencyConnected[:]:
        if VERBOSE: print( '  ' + D.strGraph() )
        if VERBOSE: print( '  '*step + str(step) + ': working on refstate ' + str(xHiOrig) + '...' )
        for ptup,xGoal in D.RefToPredTuples.get( xHiOrig, [] ) + ( [ ( D.PredToTuple[xHiOrig], '' ) ] if xHiOrig in D.PredToTuple else [] ):
          l = D.scopesToConnect( ptup[1], '', step+1 )
          if VERBOSE: print( '  '*step + str(step) + '  l=' + str(l) )
          for xLo,xHi in l:
            if VERBOSE: print( '  '*step + str(step) + '  scoping ' + D.ceiling(xLo) + ' to ' + xHi )
            D.Scopes[ D.ceiling(xLo) ] = xHi
            RecencyConnected = [ (step,x) for x in D.Chains.get(xLo,[]) ] + RecencyConnected
          if l!=[]: D.tryScope( RecencyConnected, step+1 )



