##############################################################################
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

import sys
import os
import collections
import sets
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
import discgraph

VERBOSE = False
for a in sys.argv:
  if a=='-d':
    VERBOSE = True

################################################################################

## variable replacement
def replaceVarName( struct, xOld, xNew ):
  if type(struct)==str: return xNew if struct==xOld else struct
  return tuple( [ replaceVarName(x,xOld,xNew) for x in struct ] )

## lambda expression format...
def lambdaFormat( expr, inAnd = False ):
  if len( expr ) == 0:                 return 'T'
  elif isinstance( expr, str ):        return expr
  elif expr[0] == 'lambda':            return '(\\' + expr[1] + ' ' + ' '.join( [ lambdaFormat(subexpr,False) for subexpr in expr[2:] ] ) + ')'
  elif expr[0] == 'and' and not inAnd: return '(^ ' +                 ' '.join( [ lambdaFormat(subexpr,True ) for subexpr in expr[1:] if len(subexpr)>0 ] ) + ')'
  elif expr[0] == 'and' and inAnd:     return                         ' '.join( [ lambdaFormat(subexpr,True ) for subexpr in expr[1:] if len(subexpr)>0 ] )
  else:                                return '('   + expr[0] + ' ' + ' '.join( [ lambdaFormat(subexpr,False) for subexpr in expr[1:] ] ) + ')'

## find unbound vars...
def findUnboundVars( expr, bound = [] ):
  if   len( expr ) == 0: return
  elif isinstance( expr, str ):
    if expr not in bound and expr != '_':
      sys.stderr.write( 'ERROR: unbound var: ' + expr + '\n' )
      print(            'ERROR: unbound var: ' + expr  )
  elif expr[0] == 'lambda':
    for subexpr in expr[2:]:
      findUnboundVars( subexpr, bound + [ expr[1] ] )
  else:
    for subexpr in expr[1:]:
      findUnboundVars( subexpr, bound )

## Check off consts used in expr...
def checkConstsUsed( expr, OrigConsts ):
  if len( expr ) == 0: return
  if isinstance( expr, str ): return
  if expr[0] in OrigConsts:
    OrigConsts.remove( expr[0] )
  for subexpr in expr:
    checkConstsUsed( subexpr, OrigConsts )


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
    if len(D.AnnotatedCeilings) > 1:
      print(           '#WARNING: Multiple annotated ceilings: ' + str(D.AnnotatedCeilings) )
      sys.stderr.write( 'WARNING: Multiple annotated ceilings: ' + str(D.AnnotatedCeilings) + '\n' )
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
#      if step==36: print( '  '*step + 'reachesFromSup ' + xLo + ' ' + xHi )
    return True if xLo in D.Chains.get(xHi,[]) else D.reachesInChain( D.Scopes[xLo], xHi ) if xLo in D.Scopes else any( [ D.reachesFromSup(xSup,xHi) for l,xSup in D.Inhs.get(xLo,{}).items() if l!='w' and l!='o' ] )
  def reachesFromSub( D, xLo, xHi ):
#      if step==36: print( '  '*step + 'reachesFromSub ' + xLo + ' ' + xHi )
    return True if xLo in D.Chains.get(xHi,[]) else D.reachesInChain( D.Scopes[xLo], xHi ) if xLo in D.Scopes else any( [ D.reachesFromSub(xSub,xHi) for xSub in D.Subs.get(xLo,[]) ] )
  def reachesInChain( D, xLo, xHi ):
#      if step==36: print( '  '*step + 'reachesInChain ' + xLo + ' ' + xHi )
    return D.reachesFromSup( xLo, xHi ) or D.reachesFromSub( xLo, xHi )


  def satisfyPred( D, ptup, xSplice, step ):
    if VERBOSE: print( '  '*step + str(step) + ': trying to satisfy pred tuple ' + ' '.join(ptup) + ' for ' + xSplice + '...' )
    ## For unary predicates...
    if len(ptup) == 3:
      ## If elem pred already outscoped by arg, do nothing...
      if   D.reachesInChain( ptup[1], ptup[2] ): return []
      ## If arg already outscopes splice, scope elem pred to splice...
      elif D.reachesInChain( xSplice, ptup[2] ): return [ (ptup[1],xSplice) ] if xSplice!='' else []
      ## If arg is elem pred, recurse to that pred...
      elif ptup[2] in D.PredToTuple:                        return [ (ptup[1],ptup[2]) ] + D.satisfyPred( D.PredToTuple[ptup[2]], xSplice, step+1 )
      elif D.ceiling(ptup[2]) not in D.AnnotatedCeilings:   return [ (ptup[1],ptup[2]), (ptup[2],xSplice) ] if xSplice!='' else [ (ptup[1],ptup[2]) ] 
      else:                                                 print( 'ERROR: unary could not scope: ' + ' '.join(ptup) ) 
    ## For binary predicates...
    if len(ptup) == 4:
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
      else:                                                                                      print( 'ERROR: binary could not scope: ' + ' '.join(ptup) )

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
          if l!=[]: D.tryScope( RecencyConnected, step+1 )
#            active = True


################################################################################

discctr = 0

## For each discourse graph...
for line in sys.stdin:

  discctr += 1
  print( '#DISCOURSE ' + str(discctr) )

  #### I. READ IN AND PREPROCESS DISCOURSE GRAPH...

  line = line.rstrip()
  if VERBOSE: print( 'GRAPH: ' + line )

  D = InducibleDiscGraph( line )

  #### II. ENFORCE NORMAL FORM (QUANTS AND SCOPE PARENTS AT MOST SPECIFIC INHERITANCES...

  D.check()
  D.normForm()

  #### III. INDUCE UNANNOTATED SCOPES AND EXISTENTIAL QUANTS...

  ## Helper functions to explore inheritance chain...
  def outscopingFromSup( xLo ):
    return True if xLo in D.Scopes.values() else any( [ outscopingFromSup(xHi) for l,xHi in D.Inhs.get(xLo,{}).items() if l!='w' and l!='o' ] )
  def outscopingFromSub( xHi ):
    return True if xHi in D.Scopes.values() else any( [ outscopingFromSub(xLo) for xLo in D.Subs.get(xHi,[]) ] )
  def outscopingInChain( x ):
    return outscopingFromSup( x ) or outscopingFromSub( x )
  ScopeLeaves = [ ]
  for x in D.Referents:
    if not outscopingInChain(x): ScopeLeaves.append( x )

  ## List of original (dominant) refts...
  RecencyConnected = sorted( [ ((0 if x not in D.Subs else -1) + (0 if x in ScopeLeaves else -2),x)  for x in D.Referents  if D.ceiling(x) in D.AnnotatedCeilings ], reverse = True )   # | sets.Set([ ceiling(x) for x in Scopes.values() ])


  if VERBOSE: print( 'running tryScope...' )
#  D.Scopes = tryScope( D.Scopes, RecencyConnected )
  D.tryScope( RecencyConnected )
  if VERBOSE: print( D.Scopes )
  if VERBOSE: print( 'GRAPH: ' + D.strGraph() )

  ## Induce low existential quants when only scope annotated...
#  for xCh in sorted([x if x in NuscoValues else Nuscos[x] for x in Scopes.keys()] + [x for x in Scopes.values() if x in NuscoValues]):  #sorted([ s for s in NuscoValues if 'r' not in Inhs.get(Inhs.get(s,{}).get('r',''),{}) ]): #Scopes:
#  ScopeyNuscos = [ x for x in NuscoValues if 'r' not in Inhs.get(Inhs.get(x,{}).get('r',''),{}) and (x in Scopes.keys()+Scopes.values() or Inhs.get(x,{}).get('r','') in Scopes.keys()+Scopes.values()) ]
  ScopeyNuscos = [ x for x in D.Referents | sets.Set(D.Inhs.keys()) if (x not in D.Nuscos or x in D.NuscoValues) and 'r' not in D.Inhs.get(D.Inhs.get(x,{}).get('r',''),{}) and (x in D.Scopes.keys()+D.Scopes.values() or D.Inhs.get(x,{}).get('r','') in D.Scopes.keys()+D.Scopes.values()) ]
  if VERBOSE: print( 'ScopeyNuscos = ' + str(ScopeyNuscos) )
  if VERBOSE: print( 'Referents = ' + str(D.Referents) )
  if VERBOSE: print( 'Nuscos = ' + str(D.Nuscos) )
  for xCh in ScopeyNuscos:
    if xCh not in [s for _,_,_,s,_ in D.QuantTuples]: # + [r for q,e,r,s,n in QuantTuples]:
      if D.Inhs[xCh].get('r','') == '': D.Inhs[xCh]['r'] = xCh+'r'
      if VERBOSE: print( 'Inducing existential quantifier: ' + str([ 'D:someQ', xCh+'P', D.Inhs[xCh]['r'], xCh, '_' ]) )
      D.QuantTuples.append( ( 'D:someQ', xCh+'P', D.Inhs[xCh]['r'], xCh, '_' ) )


  #### IV. ENFORCE NORMAL FORM (QUANTS AND SCOPE PARENTS AT MOST SPECIFIC INHERITANCES...

  D.normForm()

  #### V. TRANSLATE TO LAMBDA CALCULUS...

  Translations = [ ]
  Abstractions = collections.defaultdict( list )  ## Key is lambda.
  Expressions  = collections.defaultdict( list )  ## Key is lambda.

  ## Iterations...
  i = 0
  active = True
  while active: #PredTuples != [] or QuantTuples != []:
    i += 1
    active = False

    if VERBOSE: 
      print( '---- ITERATION ' + str(i) + ' ----' )
      print( 'P = ' + str(sorted(D.PredTuples)) )
      print( 'Q = ' + str(sorted(D.QuantTuples)) )
      print( 'S = ' + str(sorted(D.Scopes.items())) )
      print( 't = ' + str(sorted(D.Traces.items())) )
      print( 'I = ' + str(sorted(D.Inhs.items())) )
      print( 'T =  ' + str(sorted(Translations)) )
      print( 'A = ' + str(sorted(Abstractions.items())) )
      print( 'E = ' + str(sorted(Expressions.items())) )

    ## P rule...
    for ptup in list(D.PredTuples):
      for x in ptup[1:]:
        if x not in D.Scopes.values() and x not in D.Inhs:
          if VERBOSE: print( 'applying P to make \\' + x + '. ' + lambdaFormat(ptup) )
          Abstractions[ x ].append( ptup )
          if ptup in D.PredTuples: D.PredTuples.remove( ptup )
          active = True

    ## C rule...
    for var,Structs in Abstractions.items():
      if len(Structs) > 1:
        if VERBOSE: print( 'applying C to make \\' + var + '. ' + lambdaFormat( tuple( ['and'] + Structs ) ) )
        Abstractions[var] = [ tuple( ['and'] + Structs ) ]
        active = True

    ## M rule...
    for var,Structs in Abstractions.items():
      if len(Structs) == 1 and var not in D.Scopes.values() and var not in D.Inhs:
        if VERBOSE: print( 'applying M to make \\' + var + '. ' + lambdaFormat(Structs[0]) )
        Expressions[var] = Structs[0]
        del Abstractions[var]
        active = True
 
    ## Q rule...
    for q,e,r,s,n in list(D.QuantTuples):
      if r in Expressions and s in Expressions:
        if VERBOSE: print( 'applying Q to make ' + lambdaFormat( ( q, n, ( 'lambda', r, Expressions[r] ), ( 'lambda', s, Expressions[s] ) ) ) )   ## (' + q + ' (\\' + r + '. ' + str(Expressions[r]) + ') (\\' + s + '. ' + str(Expressions[s]) + '))' )
        Translations.append( ( q, n, ( 'lambda', r, Expressions[r] ), ( 'lambda', s, Expressions[s] ) ) )
        D.QuantTuples.remove( (q, e, r, s, n) )
        active = True

    ## I1 rule...
    for src,lbldst in D.Inhs.items():
      for lbl,dst in lbldst.items():
        if dst not in D.Inhs and dst not in Abstractions and dst not in Expressions and dst not in D.Scopes.values() and dst not in [ x for ptup in D.PredTuples for x in ptup ]:
          if VERBOSE: print( 'applying I1 to make \\' + dst + ' True' )
          Abstractions[ dst ].append( () )
          active = True

    ## I2,I3,I4 rule...
    for src,lbldst in D.Inhs.items():
      for lbl,dst in lbldst.items():
        if dst in Expressions:
          if src in D.Scopes and dst in D.Traces and D.Scopes[src] in D.Scopes and D.Traces[dst] in D.Traces:
            Abstractions[ src ].append( replaceVarName( replaceVarName( replaceVarName( Expressions[dst], dst, src ), D.Traces[dst], D.Scopes[src] ), D.Traces[D.Traces[dst]], D.Scopes[D.Scopes[src]] ) )    ## I4 rule.
            if VERBOSE: print( 'applying I4 to replace ' + dst + ' with ' + src + ' and ' + D.Traces[dst] + ' with ' + D.Scopes[src] + ' and ' + D.Traces[D.Traces[dst]] + ' with ' + D.Scopes[D.Scopes[src]] + ' to make \\' + src + ' ' + lambdaFormat(Abstractions[src][-1]) )
#            del Traces[dst]
          elif src in D.Scopes and dst in D.Traces:
            Abstractions[ src ].append( replaceVarName( replaceVarName( Expressions[dst], dst, src ), D.Traces[dst], D.Scopes[src] ) )    ## I4 rule.
            if VERBOSE: print( 'applying I4 to replace ' + dst + ' with ' + src + ' and ' + D.Traces[dst] + ' with ' + D.Scopes[src] + ' to make \\' + src + ' ' + lambdaFormat(Abstractions[src][-1]) )
#            del Traces[dst]
          else:
            if VERBOSE: print( 'applying I2/I3 to replace ' + dst + ' with ' + src + ' to make \\' + src + ' ' + lambdaFormat(replaceVarName( Expressions[dst], dst, src )) )   #' in ' + str(Expressions[dst]) )
            Abstractions[ src ].append( replaceVarName( Expressions[dst], dst, src ) )
            if dst in D.Scopes and src in [s for q,e,r,s,n in D.QuantTuples] + [r for q,e,r,s,n in D.QuantTuples]:  D.Scopes[src if src in D.NuscoValues else D.Nuscos[src][0]] = D.Scopes[dst]     ## I3 rule.
          del D.Inhs[src][lbl]
          if len(D.Inhs[src])==0: del D.Inhs[src]
          active = True
      ## Rename all relevant abstractions with all inheritances, in case of multiple inheritance...
      for dst in D.AllInherited[ src ]:
        if dst in Expressions:
          Abstractions[ src ] = [ replaceVarName( a, dst, src )  for a in Abstractions[src] ]

    ## S1 rule...
    for q,n,R,S in list(Translations):
      if S[1] in D.Scopes:
        if VERBOSE: print( 'applying S1 to make (\\' + D.Scopes[ S[1] ] + ' ' + q + ' ' + str(R) + ' ' + str(S) + ')' )
        Abstractions[ D.Scopes[ S[1] ] ].append( (q, n, R, S) )
        del D.Scopes[ S[1] ]
#        if R[1] in Scopes: del Scopes[ R[1] ]   ## Should use 't' trace assoc.
        Translations.remove( (q, n, R, S) )
        active = True

  expr = tuple( ['and'] + Translations )
  print( lambdaFormat(expr) )
#  for expr in Translations:
  findUnboundVars( expr )
  checkConstsUsed( expr, D.OrigConsts )
  for k in D.OrigConsts:
    print(           '#WARNING: const does not appear in translations: ' + k )
    sys.stderr.write( 'WARNING: const does not appear in translations: ' + k + '\n' )

