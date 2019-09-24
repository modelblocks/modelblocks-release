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
    D.Legators = collections.defaultdict( list )
    D.Heirs    = collections.defaultdict( list )
    for xLo in D.Inhs:
      D.Legators[ xLo ] = D.getLegators( xLo )
    for xHi in D.Subs:
      D.Heirs[ xHi ] = D.getHeirs( xHi )
    if VERBOSE: print( 'Legators = ' + str(D.Legators) )
    if VERBOSE: print( 'Heirs = ' + str(D.Heirs) )
    def getTopUnaryLegators( xLo ):
      L = [ xLeg  for l,xHi in D.Inhs.get( xLo, {} ).items()  if l!='w' and l!='o' and len( D.Subs.get(xHi,[]) ) < 2  for xLeg in getTopUnaryLegators(xHi) ]
      return L if L != [] else [ xLo ]
#       if D.Inhs.get( xLo, {} ).items() != []  else [ xLo ]
#      UnaryL = [ xLeg  for xLeg in D.Legators.get(xLo,[])  if all([ xLo in D.Heirs.get(xHeir,[])  for xHeir in D.Legators.get(xLo,[])  if xHeir in D.Heirs.get(xLeg,[]) ]) ]
#      return [ x  for x in UnaryL  if not any([ x in D.Heirs.get(y,[])  for y in UnaryL  if y != x ]) ] 
    def getTopLegators( xLo ):
      L = [ xLeg  for l,xHi in D.Inhs.get( xLo, {} ).items()  if l!='w' and l!='o'  for xLeg in getTopLegators(xHi) ]
      return L if L != [] else [ xLo ]
#       if D.Inhs.get( xLo, {} ).items() != []  else [ xLo ]
    D.TopLegators = { xLo : sets.Set( getTopLegators(xLo) )  for xLo in D.Inhs }
    if VERBOSE: print( 'TopLegators = ' + str(D.TopLegators) )
    D.TopUnaryLegators = { xLo : sets.Set( getTopUnaryLegators(xLo) )  for xLo in D.Inhs }
    if VERBOSE: print( 'TopUnaryLegators = ' + str(D.TopUnaryLegators) )
#    D.PredRecency = { }
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
      Out += [ (ptup,x) for ptup in D.PredTuples if x in ptup[2:] ]
      for src in D.Subs.get(x,[]):
        Out += orderTuplesFromSubs( src )
#        Out += [ (ptup,src) for ptup in D.PredTuples if src in ptup[2:] ]
      return Out
    D.FullRefToPredTuples = { x : orderTuplesFromSubs(x) + orderTuplesFromSups(x)  for x in D.Referents }
    D.WeakRefToPredTuples = { x : orderTuplesFromSubs( D.Inhs.get(x,{}).get('r',x) )  for x in D.Referents }
    D.BareRefToPredTuples = { x : [ (ptup,x)  for ptup in D.PredTuples  if x in ptup[2:] ]  for x in D.Referents }
    if VERBOSE: print( 'FullRefToPredTuples = ' + str(D.FullRefToPredTuples) )
    if VERBOSE: print( 'WeakRefToPredTuples = ' + str(D.WeakRefToPredTuples) )
    if VERBOSE: print( 'BareRefToPredTuples = ' + str(D.BareRefToPredTuples) )
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

  def getLegators( D, xLo ):
    Out = [ xLo ]
    for l,xHi in D.Inhs.get(xLo,{}).items():
      Out += D.getLegators( xHi )
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
  def alreadyConnected( D, x, xGoal, Connected ):
#    print( 'ceiling of ' + x + ' is ' + D.ceiling(x) )
#    return ( xGoal == '' and any([ y in Connected  for y in D.Chains[x] ]) ) or D.reachesInChain( x, xGoal )
    return ( xGoal == '' and x in Connected ) or D.reachesInChain( x, xGoal )
#    return ( xGoal == '' and D.ceiling( x ) in D.AnnotatedCeilings ) or D.reachesInChain( x, xGoal )
#    return ( xGoal == '' and any([ D.ceiling( x ) in D.AnnotatedCeilings ] + [ D.ceiling( y ) in D.AnnotatedCeilings  for l,y in D.Inhs.get(x,{}).items()  if l!='w' and l!='o' ]) ) or D.reachesInChain( x, xGoal )
  def weaklyConnected( D, x, xGoal, Connected ):
    return ( xGoal == '' and any([ y in Connected  for y in D.Chains[x] ]) ) or D.reachesInChain( x, xGoal )


  ## Method to return list of scope (src,dst) pairs to connect target to goal...
  def scopesToConnect( D, xTarget, xGoal, step, Connected ):
    if VERBOSE: print( '  '*step + str(step) + ': trying to satisfy pred ' + xTarget + ' for ' + xGoal + '...' )

#    print( [ xSub  for xSub in D.Subs.get(xTarget,[])  if D.Inhs.get(xSub,{}).get('r','') != xTarget ] )

    ## If any non-'r' heirs, return results for heirs (elementary predicates are always final heirs)...
    if [] != [ xSub  for xSub in D.Subs.get(xTarget,[]) ]:
      return [ sco  for xSub in D.Subs.get( xTarget, [] )  for sco in D.scopesToConnect( xSub, xGoal, step+1, Connected ) ]
#    if [] != [ xSub  for xSub in D.Subs.get(xTarget,[])  if D.Inhs.get(xSub,{}).get('r','') != xTarget ]:
#      return [ sco  for xSub in D.Subs.get( xTarget, [] )  if D.Inhs.get(xSub,{}).get('r','') != xTarget  for sco in D.scopesToConnect( xSub, xGoal, step+1 ) ]

    ## If zero-ary (non-predicate)...
    if xTarget not in D.PredToTuple:
      if xGoal == '' or D.reachesInChain( xTarget, xGoal ): return []
      else:                                                 return [ (xTarget,xGoal) ]
#      else: complain( xTarget + ' is not already connected to goal and is not predicate, so cannot be outscoped by ' + xGoal )

    ptup = D.PredToTuple[ xTarget ]
#    ## Sanity check...
#    if ptup[1] != xTarget:
#      complain( 'too weird -- elem pred ' + xTarget + ' not equal to ptup[1]: ' + ptup[1] )
    xLowest = ptup[1]
    if len(ptup) > 2: xOther1 = ptup[2]
    if len(ptup) > 3: xOther2 = ptup[3]
    if len(ptup) > 2 and D.Scopes.get(ptup[2],'') == ptup[1]: xLowest,xOther1 = ptup[2],ptup[1]
    if len(ptup) > 3 and D.Scopes.get(ptup[2],'') == ptup[1]: xLowest,xOther1,xOther2 = ptup[2],ptup[1],ptup[3]
    if len(ptup) > 3 and D.Scopes.get(ptup[3],'') == ptup[1]: xLowest,xOther1,xOther2 = ptup[3],ptup[1],ptup[2]
    ## Report any cycles from participant to elementary predicate...
    for x in ptup[2:]:
      if D.reachesInChain( x, xLowest ) and not x.endswith('\''):
        complain( 'elementary predication ' + ptup[0] + ' ' + xLowest + ' should not outscope argument ' + x + '!' ) 
    ## If all participants reachable from elem pred, nothing to do...
    if any([ all([ D.reachesInChain(xLo,xHi)  for xHi in ptup[1:]  if xHi != xLo ])  for xLo in ptup[1:] ]):   #all([ D.reachesInChain( xLowest, x )  for x in ptup[2:] ]):
#      if VERBOSE: print( '  '*step + str(step) + ': all args of pred ' + ptup[0] + ' ' + ptup[1] + ' are reachable' )
      if xGoal == '' or any([ D.reachesInChain( x, xGoal )  for x in ptup[1:] ]):   #D.reachesInChain( xLowest, xGoal ):
        return []
      elif D.alreadyConnected( ptup[1], '', Connected ):
        complain( 'elementary predication ' + ptup[0] + ' ' + ptup[1] + ' is already fully bound, cannot become outscoped by goal referent ' + xGoal )

    ## If unary predicate...
#    if len( ptup ) == 3:
    for xLowest,xOther1 in [ (xLowest,xOther1) ] if len(ptup)==3 else [ (ptup[1],ptup[3]) ] if len(ptup)==4 and D.Scopes.get(ptup[2],'')==ptup[1] else [ (ptup[1],ptup[2]) ] if len(ptup)==4 and D.Scopes.get(ptup[3],'')==ptup[1] else []:
      ## Update any existing scopes to ensure scope chain targets are or are inherited by arguments...
      if D.reachesInChain( xLowest, xOther1 ):
        for xScopeParent in D.Heirs[ xOther1 ]:
          for xScopeChild in [ xC  for xC,xP in D.Scopes.items()  if xP == xScopeParent  if xP == xScopeParent and D.reachesInChain( xLowest, xC ) ]:   #D.ScopeChildren.get( xScopeParent, {} ):
            if xScopeParent != xOther1:
              D.Scopes[ xScopeChild ] = xOther1
              if VERBOSE: print( '  '*step + str(step) + ': changing heir scope ' + xScopeChild + ' ' + xScopeParent + ' to legator scope ' + xScopeChild + ' ' + xOther1 )
      ## Recommend scopes...
      if D.alreadyConnected( xOther1, xLowest, Connected ):
        complain( 'elementary predication ' + ptup[0] + ' ' + xLowest + ' should not outscope argument ' + xOther1 )
      if D.reachesInChain( xGoal, xOther1 ):
        if VERBOSE: print( ' ' + '  '*step + str(step) + ': case a' )
        return ( [ (xLowest,xGoal) ] if xLowest == ptup[1] else [] )
      else:
        if VERBOSE: print( ' ' + '  '*step + str(step) + ': case b' )
        return ( [ (xLowest,xOther1) ] if xLowest == ptup[1] else [] ) + ( D.scopesToConnect( xOther1, xGoal, step+1, Connected ) if xOther1 != ptup[1] else [ (xOther1,xGoal) ] )

    ## If binary predicate...
    if len( ptup ) == 4:
      ## Update any existing scopes to ensure scope chain targets are or are inherited by arguments...
      for xLo,xHi in [ (xLowest,xOther1), (xLowest,xOther2), (xOther1,xLowest), (xOther1,xOther2), (xOther2,xLowest), (xOther2,xOther1) ]:
        if D.reachesInChain( xLo, xHi ):
          for xScopeParent in D.Heirs[ xHi ]:
            for xScopeChild in [ xC  for xC,xP in D.Scopes.items()  if xP == xScopeParent and D.reachesInChain( xLo, xC ) ]:
              if xScopeParent != xHi:
                D.Scopes[ xScopeChild ] = xHi
                if VERBOSE: print( '  '*step + str(step) + ': changing heir scope ' + xScopeChild + ' ' + xScopeParent + ' to legator scope ' + xScopeChild + ' ' + xHi )
      ## Recommend scopes...
      if D.alreadyConnected( xOther1, xGoal, Connected ) and D.alreadyConnected( xOther2, xGoal, Connected ) and not D.alreadyConnected( xOther1, xOther2, Connected ) and not D.alreadyConnected( xOther2, xOther1, Connected ):
        complain( 'arguments ' + xOther1 + ' and ' + xOther2 + ' of elementary predication ' + ptup[0] + ' ' + ptup[1] + ' outscoped in different branches -- possibly due to disconnected scope annotations' )
      if D.reachesInChain( xGoal, xOther1 ) and D.reachesInChain( xGoal, xOther2 ):
        if VERBOSE: print( ' ' + '  '*step + str(step) + ': case 0' )
        return ( [ (xLowest,xGoal) ] if xLowest == ptup[1] else [] ) 
      for xLo,xMd,xHi in [ (xLowest,xOther2,xOther1), (xLowest,xOther1,xOther2) ]:
        if D.reachesInChain( xGoal, xHi ):
          if VERBOSE: print( ' ' + '  '*step + str(step) + ': case 1 ' + xLo + ' under ' + xMd + ' under goal ' + xGoal + ' under ' + xHi )
          return ( [ (xLo,xMd) ] if xLo == ptup[1] else [] ) + ( D.scopesToConnect( xMd, xGoal,   step+1, Connected ) if xMd != ptup[1] else [ (xMd,xGoal) ] )
#      if D.reachesInChain( xGoal, xOther1 ):
#        if VERBOSE: print( ' ' + '  '*step + str(step) + ': case 1' )
#        return ( [ (xLowest,xOther2) ] if xLowest == ptup[1] else [] ) + ( D.scopesToConnect( xOther2, xGoal,   step+1, Connected ) if xOther2 != ptup[1] else [ (xOther2,xGoal) ] )
#      if D.reachesInChain( xGoal, xOther2 ):
#        if VERBOSE: print( ' ' + '  '*step + str(step) + ': case 2' )
#        return ( [ (xLowest,xOther1) ] if xLowest == ptup[1] else [] ) + ( D.scopesToConnect( xOther1, xGoal,   step+1, Connected ) if xOther1 != ptup[1] else [ (xOther1,xGoal) ] )
      print( 'D.alreadyConnected( ' + xOther1 + ', ' + xGoal + ' ) = ' + str( D.alreadyConnected( xOther1, xGoal, Connected ) ) + 'D.reachesInChain( ' + xOther1 + ', ' + xOther2 + ' ) = ' + str( D.reachesInChain( xOther1, xOther2 ) ) ) 
      for xLo,xMd,xHi in [ (xLowest,xOther1,xOther2), (xLowest,xOther2,xOther1) ]:
        if D.alreadyConnected( xHi, xGoal, Connected ) and not D.reachesInChain( xHi, xMd ):
          if VERBOSE: print( ' ' + '  '*step + str(step) + ': case 3 ' + xLo + ' under ' + xMd + ' under ' + xHi + ' under goal ' + xGoal )
          return ( [ (xLowest,xMd) ] if xLo == ptup[1] else [] ) + ( D.scopesToConnect( xMd, xHi, step+1, Connected ) if xMd != ptup[1] else [ (xMd,xHi) ] )
      for xLo,xMd,xHi in [ (xLowest,xOther1,xOther2), (xLowest,xOther2,xOther1) ]:
        if D.weaklyConnected( xHi, xGoal, Connected ) and not D.reachesInChain( xHi, xMd ):
          if VERBOSE: print( ' ' + '  '*step + str(step) + ': case 4 ' + xLo + ' under ' + xMd + ' under ' + xHi + ' under goal ' + xGoal )
          return ( [ (xLowest,xMd) ] if xLo == ptup[1] else [] ) + ( D.scopesToConnect( xMd, xHi, step+1, Connected ) if xMd != ptup[1] else [ (xMd,xHi) ] )
#      if D.alreadyConnected( xOther1, xGoal, Connected ) and not D.reachesInChain( xOther1, xOther2 ):
#        if VERBOSE: print( ' ' + '  '*step + str(step) + ': case 3' )
#        return ( [ (xLowest,xOther2) ] if xLowest == ptup[1] else [] ) + ( D.scopesToConnect( xOther2, xOther1, step+1, Connected ) if xOther2 != ptup[1] else [ (xOther2,xOther1) ] )
#      if D.alreadyConnected( xOther2, xGoal, Connected ) and not D.reachesInChain( xOther2, xOther1 ):
#        if VERBOSE: print( ' ' + '  '*step + str(step) + ': case 4' )
#        return ( [ (xLowest,xOther1) ] if xLowest == ptup[1] else [] ) + ( D.scopesToConnect( xOther1, xOther2, step+1, Connected ) if xOther1 != ptup[1] else [ (xOther1,xOther2) ] )
      if xGoal == '' and xOther2 in D.getHeirs( xOther1 ):
        if VERBOSE: print( ' ' + '  '*step + str(step) + ': case 5' )
        return ( [ (xLowest,xOther2) ] if xLowest == ptup[1] else [] ) + ( D.scopesToConnect( xOther2, xOther1, step+1, Connected ) if xOther2 != ptup[1] else [ (xOther2,xOther1) ] )
      if xGoal == '':
        if VERBOSE: print( ' ' + '  '*step + str(step) + ': case 6' )
        return ( [ (xLowest,xOther1) ] if xLowest == ptup[1] else [] ) + ( D.scopesToConnect( xOther1, xOther2, step+1, Connected ) if xOther1 != ptup[1] else [ (xOther1,xOther2) ] )
      else:
        if VERBOSE: print( ' ' + '  '*step + str(step) + ': case 7' )
        return ( [ (xLowest,xOther1) ] if xLowest == ptup[1] else [] ) + ( D.scopesToConnect( xOther1, xOther2, step+1, Connected ) if xOther1 != ptup[1] else [ (xOther1,xOther2) ] ) + ( D.scopesToConnect( xOther2, xGoal, step+1, Connected ) if xOther2 != ptup[1] else [ (xOther2,xGoal) ] )
#complain( 'predicate ' + xLowest + ' with goal ' + xGoal + ' not sufficiently constrained; danger of garden-pathing' )

    ## If trinary and higher predicates...
    else:
      complain( 'no support for super-binary predicates: ' + ' '.join(ptup) )


  def constrainDeepestReft( D, xTarg, step, Connected, isFull=False ):
    if VERBOSE: print( '  '*step + str(step) + ': recursing to ' + xTarg + '...' )
    ## If any non-'r' heirs, return results for heirs (elementary predicates are always final heirs)...
#    if [] != [ xSub  for xSub in D.Subs.get( xTarg, [] ) ]:
#      return [ sco   for xSub in D.Subs.get( xTarg, [] )  for sco in D.constrainDeepestReft( xSub, step+1, Connected, isFull ) ]
    for xSub in D.Subs.get( xTarg, [] ):
      l = D.constrainDeepestReft( xSub, step+1, Connected, isFull )
      if l != []: return l
    ## First, recurse down scope tree...
#    for x in D.Chains.get( D.Inhs.get(xTarg,{}).get('r',xTarg), D.Chains[xTarg] ):
    for xLo,xHi in D.Scopes.items():
      if xHi == xTarg:
        for xLeg in D.TopLegators.get( xLo, [xLo] ) if isFull else D.TopUnaryLegators.get( xLo, [xLo] ):
          l = D.constrainDeepestReft( xLeg, step+1, Connected, isFull )
          if l != []: return l
    ## Second, try all preds...
#    for x in D.Chains[ xTarg ]:
    for ptup,_ in D.BareRefToPredTuples.get( xTarg, [] ):   #D.FullRefToPredTuples.get( xTarg, [] ) if isFull else D.WeakRefToPredTuples.get( xTarg, [] ):
        l = D.scopesToConnect( ptup[1], '', step+1, Connected )
        if l != []: return l
    return []


  ## Method to fill in deterministic or truth-functionally indistinguishable scope associations (e.g. for elementary predications) that are not explicitly annotated...
  def tryScope( D, RecencyConnected, isFull, step=1 ):
        if VERBOSE: print( 'RecencyConnected = ' + str(RecencyConnected) )
#      active = False
#      l = []
##      RecencyConnected = sorted( [ ( s if not any([y in D.Scopes.values()  for y in D.Chains.get(x,[]) ]) else -1, x )  for (s,x) in RecencyConnected ] )

#      for _,xHiOrig in RecencyConnected[:]:
#        if VERBOSE: print( '  '*step + 'GRAPH: ' + D.strGraph() )
#        if VERBOSE: print( '  '*step + str(step) + ': working on refstate ' + str(xHiOrig) + '...' )

        '''
#      TupList = sorted([ (0 if any([ xHi in D.Scopes  for xLo in ptup  for xHi in D.Legators[xLo] ]) else -1,ptup)  for ptup in D.PredTuples ], reverse = True)
      TupList = sorted( [ ( D.PredRecency.get(ptup,0) if any([ xHi in D.Scopes  for xLo in ptup[1:]  for xHi in D.Legators[xLo] ]) and not any([ xHi in D.Scopes.values()  for xLo in ptup[1:]  for xHi in D.Legators[xLo] ]) else 
                            D.PredRecency.get(ptup,0)-1 if any([ xHi in D.Scopes  for xLo in ptup[1:]  for xHi in D.Chains[xLo] ]) and not any([ xHi in D.Scopes.values()  for xLo in ptup[1:]  for xHi in D.Chains[xLo] ]) else 
                            D.PredRecency.get(ptup,0)-2 if any([ xHi in D.Scopes  for xLo in ptup[1:]  for xHi in D.Legators[xLo] ]) else
                            D.PredRecency.get(ptup,0)-3 if any([ xHi in D.Scopes  for xLo in ptup[1:]  for xHi in D.Chains[xLo] ]) else
                            D.PredRecency.get(ptup,0)-4,ptup)  for ptup in D.PredTuples ], reverse = True )
      if VERBOSE: print( 'TUP LIST: ' + str(TupList) )
      for _,ptup in TupList:
        '''
#        for ptup,xGoal in D.RefToPredTuples.get( xHiOrig, [] ) + ( [ ( D.PredToTuple[xHiOrig], '' ) ] if xHiOrig in D.PredToTuple else [] ):
#          l = D.scopesToConnect( ptup[1], '', step+1, [ x  for s,x in RecencyConnected ] )

#      active = True
#      while active:
#          active = False
        if VERBOSE: print( '  '*step + 'GRAPH: ' + D.strGraph() )
        for xTarget in [ x  for x in D.AnnotatedCeilings  if not any([ y  for y in D.AnnotatedCeilings  if x != y and x in D.Heirs.get(y,[]) ]) ]:
          l = D.constrainDeepestReft( xTarget, step+1, [ x  for s,x in RecencyConnected ], isFull )
          if VERBOSE: print( '  '*step + str(step) + '  l=' + str(l) )
          for xLo,xHi in sets.Set(l):
            if VERBOSE: print( '  '*step + str(step) + '  scoping ' + D.ceiling(xLo) + ' to ' + xHi )
            D.Scopes[ D.ceiling(xLo) ] = xHi
#            D.PredRecency[ ptup ] = step
            RecencyConnected = [ (step,x) for x in D.Chains.get(xLo,[]) ] + RecencyConnected
          if l!=[]:
            D.tryScope( RecencyConnected, isFull, step+1 )
#            active = True


