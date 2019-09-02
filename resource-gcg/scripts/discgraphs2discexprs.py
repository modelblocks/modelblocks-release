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
import collections
import sets

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

class DiscGraph:

  def __init__( D, line ):

    ## Initialize associations...
    D.PorQs  = collections.defaultdict( list )                                     ## Key is elem pred.
    D.Scopes = { }                                                                 ## Key is outscoped.
    D.Traces = { }                                                                 ## Key is outscoped.
    D.Inhs   = collections.defaultdict( lambda : collections.defaultdict(float) )  ## Key is inheritor.
    D.Nuscos = collections.defaultdict( list )                                     ## Key is restrictor.
    D.NuscoValues = { }
    D.Inheriteds = { }
   
    ## For each assoc...
    for assoc in sorted( line.split(' ') ):
      src,lbl,dst = assoc.split(',')
      if lbl.isdigit():  D.PorQs  [src].insert( int(lbl), dst )   ## Add preds and quants.
      elif lbl == 's':   D.Scopes [src]      = dst                ## Add scopes.
      elif lbl == 't':   D.Traces [src]      = dst                ## Add traces.
      else:              D.Inhs   [src][lbl] = dst                ## Add inheritances.
      if lbl == 'r':     D.Nuscos [dst].append( src )             ## Index nusco of each restr.
      if lbl == 'r':     D.NuscoValues[src]  = True
      if lbl == 'e':     D.Inheriteds[dst]   = True

    D.PredTuples  = [ ]
    D.QuantTuples = [ ] 

    ## Distinguish preds and quants...
    for elempred,Particips in D.PorQs.items():
      ## If three participants and last restriction-inherits from previous, it's a quant...
  #    if len( Particips ) == 3 and Inhs.get(Particips[2],{}).get('r','') == Particips[1]:  QuantTuples.append( tuple( [ Particips[0] ] + [ elempred ] + Particips[1:] ) )
      if Particips[0].endswith('Q'):  D.QuantTuples.append( tuple( [ Particips[0] ] + [ elempred ] + Particips[1:] + (['_'] if len(Particips)<4 else []) ) )
      else:                           D.PredTuples.append ( tuple( [ Particips[0] ] + [ elempred ] + Particips[1:] ) )

    D.OrigConsts = [ ep[0] for ep in D.PredTuples ] + [ q[0] for q in D.QuantTuples ]

    ## Report items...
    if VERBOSE: 
      print( 'P = ' + str(sorted(D.PredTuples)) )
      print( 'Q = ' + str(sorted(D.QuantTuples)) )
      print( 'S = ' + str(sorted(D.Scopes.items())) )

    ## Construct list of inheriting refstates...
    D.Subs = collections.defaultdict( list )
    for xLo,lxHi in D.Inhs.items():
      for l,xHi in lxHi.items():
        if l!='w' and l!='o':
          D.Subs[ xHi ].append( xLo )
    print( 'Subs = ' + str(D.Subs) )


  def strGraph( D, HypScopes = None ):  # PredTuples, QuantTuples, Inhs, Scopes ):
    if HypScopes == None: HypScopes = D.Scopes
    G = []
    ## List elementary predications...
    for ptup in D.PredTuples:
      G.append( ptup[1] + ',0,' + ptup[0] )
      for n,x in enumerate( ptup[2:] ):
        G.append( ptup[1] + ',' + str(n+1) + ',' + x )
    ## List quantifiers...
    for qtup in D.QuantTuples:
      G.append( qtup[1] + ',0,' + qtup[0] )
      for n,x in enumerate( qtup[2:] ):
        if x != '_':
          G.append( qtup[1] + ',' + str(n+1) + ',' + x )
    ## List inheritances...
    for xLo,lxHi in D.Inhs.items():
      for l,xHi in lxHi.items():
        G.append( xLo + ',' + l +',' + xHi )
    ## List scopes...
    for xLo,xHi in HypScopes.items():
      G.append( xLo + ',s,' + xHi )
    ## print out...
    return ' '.join( sorted( G ) )

  ## Check that no reft has multiple outscopers...
  def getBossesFromSup( D, xLo ):
#      print( 'now getting sup ' + xLo )
    if xLo in D.Scopes: return D.getBossesInChain( D.Scopes[xLo] )
    return sets.Set( [ y  for l,xHi in D.Inhs.get(xLo,{}).items() if l!='w' and l!='o'  for y in D.getBossesFromSup(xHi) ] )
  def getBossesFromSub( D, xHi ):
#      print( 'now getting sub ' + xHi )
    if xHi in D.Scopes: return D.getBossesInChain( D.Scopes[xHi] )
    return sets.Set( [ y  for xLo in D.Subs.get(xHi,[])  for y in D.getBossesFromSub(xLo) ] )
  def getBossesInChain( D, x ):
    out = D.getBossesFromSup(x) | D.getBossesFromSub(x)
    return out if len(out)>0 else sets.Set( [x] )

  def check( D ):
    ## Check for inheritance cycles...
    def checkInhCycles( xLo, L=[] ):
      if xLo in L: sys.stderr.write( 'ERROR: inheritance cycle: ' + str(L+[xLo]) + '\n' )
      for l,xHi in D.Inhs.get(xLo,{}).items():
        checkInhCycles( xHi, L + [xLo] )
    ## Check for scope cycles...
    def checkScopeCyclesFromSup( xLo, L=[] ):
      if xLo in L:
        print( 'WARNING: scope cycle: ' + str(L+[xLo]) )
        sys.stderr.write( 'WARNING: scope cycle: ' + str(L+[xLo]) + '\n' )
        return True
      return checkScopeCyclesInChain(D.Scopes[xLo],L+[xLo]) if xLo in D.Scopes else any([ checkScopeCyclesFromSup(xHi,L+[xLo]) for l,xHi in D.Inhs.get(xLo,{}).items() if l!='w' and l!='o' ])
    def checkScopeCyclesFromSub( xHi, L=[] ):
      if xHi in L:
        print( 'WARNING: scope cycle: ' + str(L+[xHi]) )
        sys.stderr.write( 'WARNING: scope cycle: ' + str(L+[xHi]) + '\n' )
        return True
      return checkScopeCyclesInChain(D.Scopes[xHi],L+[xHi]) if xHi in D.Scopes else any([ checkScopeCyclesFromSub(xLo,L+[xHi]) for xLo in D.Subs.get(xHi,[]) ])
    def checkScopeCyclesInChain( x, L=[] ):
      return checkScopeCyclesFromSup( x, L ) or checkScopeCyclesFromSub( x, L )
    ## Check for inheritance cycles...
    for x in D.Referents:
      checkInhCycles( x )
    ## Check for scopecycles...
    for x in D.Referents:
      checkScopeCyclesInChain( x )
    ## Check for multiple outscopings...
    for x in D.Referents:
      if len( D.getBossesInChain(x) ) > 1: sys.stderr.write( 'WARNING: ' + x + ' has multiple outscopings in inheritance chain: ' + str( D.getBossesInChain(x) ) + '\n' )
      if VERBOSE: print( 'Bosses of ' + x + ': ' + str(D.getBossesInChain(x)) )

  def normForm( D ):
    ## Smite redundant nuscos of predicative noun phrases out of Subs...
    for xHi,l in D.Subs.items():
      for xLo in l:
        if 'r' in D.Inhs.get(D.Inhs.get(xLo,[]).get('r',''),[]):
          if VERBOSE: print( 'Smiting ' + xLo + ' out of Subs, for being redundant.' )
          D.Subs[xHi].remove(xLo)
          if len(D.Subs[xHi])==0: del D.Subs[xHi]
    ## Propagate scopes down inheritance chains...
    active = True
    while active:
      active = False
      for xHi in D.Scopes.keys():
        for xLo in D.Subs.get(xHi,[]):
          if xLo not in D.Scopes:
            if VERBOSE: print( 'Inheriting scope parent ' + D.Scopes[xHi] + ' from ' + xHi + ' to ' + xLo + '.' )
            D.Scopes[ xLo ] = D.Scopes[ xHi ]
            active = True
    ## Propagate quants down inheritance chains...
    active = True
    while active:
      active = False
      for q,e,r,xHi,n in D.QuantTuples[:]:
        for xLo in D.Subs.get(xHi,[]):
          if xLo not in [s for _,_,_,s,_ in D.QuantTuples]:
            if VERBOSE: print( 'Inheriting quant ' + q + ' from ' + xHi + ' to ' + xLo + '.' )
            D.QuantTuples.append( (q,e,r,xLo,n) )
            active = True
    ## Clean up abstract scopes...
    for xHi in D.Scopes.keys():
      if xHi in D.Subs: #for xLo in Subs.get(xHi,[]):
        if VERBOSE: print( 'Removing redundant abstract scope parent ' + D.Scopes[xHi] + ' from ' + xHi + ' because of inheritance at ' + str(D.Subs[xHi]) )
        del D.Scopes[xHi]
    ## Clean up abstract quants...
    for q,e,r,s,n in D.QuantTuples[:]:
      if s in D.Subs:
        if VERBOSE: print( 'Removing redundant abstract quant ' + q + ' from ' + s + ' because of inheritance at ' + D.Subs[s][0] )
        D.QuantTuples.remove( (q,e,r,s,n) )

    ## Report items...
    if VERBOSE: 
      print( 'P = ' + str(sorted(D.PredTuples)) )
      print( 'Q = ' + str(sorted(D.QuantTuples)) )
      print( 'S = ' + str(sorted(D.Scopes.items())) )


  ## Scope ceiling...
#  def getCeilingFromSup( xLo ):
#    return getCeilingInChain( Scopes[xLo] ) if xLo in Scopes else sets.Set( [ y  for l,xHi in Inhs.get(xLo,{}).items() if l!='w'  for y in getCeilingFromSup(xHi) ] )
#  def getCeilingFromSub( xHi ):
#    return getCeilingInChain( Scopes[xHi] ) if xHi in Scopes else sets.Set( [ y  for xLo in Subs.get(xHi,[])  for y in getCeilingFromSub(xLo) ] )
#  def getCeilingInChain( x ):
#    out = getCeilingFromSup( x ) | getCeilingFromSub( x )
#    return out if len(out)>0 else sets.Set( [x] )


################################################################################

class InducibleDiscGraph( DiscGraph ):

  def getChainFromSup( D, xLo ):
    return [ xLo ] + [ x for l,xHi in D.Inhs.get(xLo,{}).items() if l!='w' and l!='o' for x in D.getChainFromSup(xHi) ]
  def getChainFromSub( D, xHi ):
    return [ xHi ] + [ x for xLo in D.Subs.get(xHi,[]) for x in D.getChainFromSub(xLo) ]

  def __init__( D, line ):
    DiscGraph.__init__( D, line )
    ## List of referents that are or participate in elementary predications...
    D.Referents = sets.Set( [ x for pred in D.PredTuples for x in pred[1:] ] + D.Inhs.keys() )
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
      print( 'WARNING: Multiple annotated ceilings: ' + str(D.AnnotatedCeilings) )
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
  def reachesFromSup( D, HypScopes, xLo, xHi ):
#      if step==36: print( '  '*step + 'reachesFromSup ' + xLo + ' ' + xHi )
    return True if xLo in D.Chains.get(xHi,[]) else D.reachesInChain( HypScopes, HypScopes[xLo], xHi ) if xLo in HypScopes else any( [ D.reachesFromSup(HypScopes,xSup,xHi) for l,xSup in D.Inhs.get(xLo,{}).items() if l!='w' and l!='o' ] )
  def reachesFromSub( D, HypScopes, xLo, xHi ):
#      if step==36: print( '  '*step + 'reachesFromSub ' + xLo + ' ' + xHi )
    return True if xLo in D.Chains.get(xHi,[]) else D.reachesInChain( HypScopes, HypScopes[xLo], xHi ) if xLo in HypScopes else any( [ D.reachesFromSub(HypScopes,xSub,xHi) for xSub in D.Subs.get(xLo,[]) ] )
  def reachesInChain( D, HypScopes, xLo, xHi ):
#      if step==36: print( '  '*step + 'reachesInChain ' + xLo + ' ' + xHi )
    return D.reachesFromSup( HypScopes, xLo, xHi ) or D.reachesFromSub( HypScopes, xLo, xHi )

  '''
  def outscopeWith( D, HypScopes, xLo, xHi, step ):
    print( '  '*step + str(step) + ': trying to outscope ' + xLo + ' with ' + xHi + '...' )
    if D.reachesInChain(HypScopes,xLo,xHi): return []
    elif not D.reachesInChain( HypScopes, xLo, D.ceiling(xHi) ): return [ ( D.ceiling(xLo), xHi ) ]
    else: print( 'ERROR: ' + xLo + ' coapical with but not outscoped by ' + xHi + '!' )
    return [ ]

  def outscopeArgWith( D, HypScopes, xLo, xHi, step ):
    print( '  '*step + str(step) + ': trying to outscope arg ' + xLo + ' with ' + xHi + '...' )
    if xLo in D.PorQs: return D.satisfy( HypScopes, tuple( [ D.PorQs[xLo][0] ] + [ xLo ] + D.PorQs[xLo][1:] ), step+1, xHi )
    else: return D.outscopeWith( HypScopes, xLo, xHi, step+1 )

  def satisfy( D, HypScopes, ptup, step, xHi=None ):
    print( '  '*step + str(step) + ': trying to satisfy pred tuple ' + ' '.join(ptup) + ' with hi ' + str(xHi) + '...' )
    if len(ptup)==3 and D.reachesInChain( HypScopes, xHi, ptup[2] ): return []
    if xHi != None:
      l = D.satisfy( HypScopes, ptup, step+1 )
      if l!=[]: return l + D.outscopeArgWith( HypScopes, l[-1][-1], xHi, step+1 )
      else: return []
    elif   len(ptup)==2:
      print( 'ERROR: zero-ary predicate: ' + str(ptup) )
      exit( 1 )
    elif len(ptup)==3:  return D.outscopeWith( HypScopes, ptup[1], ptup[2], step+1 )
    elif len(ptup)==4:
      if   D.reachesInChain( HypScopes, ptup[2], ptup[3] ) and not D.reachesInChain( HypScopes, ptup[1], ptup[2]):
        return D.outscopeWith( HypScopes, ptup[1], ptup[2], step+1 )
      elif D.reachesInChain( HypScopes, ptup[1], ptup[2] ) and not D.reachesInChain( HypScopes, ptup[2], ptup[3]) and ptup[2] not in D.NotOutscopable:
        return D.outscopeArgWith( HypScopes, ptup[2], ptup[3], step+1 )
      elif D.reachesInChain( HypScopes, ptup[3], ptup[2] ) and not D.reachesInChain( HypScopes, ptup[1], ptup[3]):
        return D.outscopeWith( HypScopes, ptup[1], ptup[3], step+1 )
      elif D.reachesInChain( HypScopes, ptup[1], ptup[3] ) and not D.reachesInChain( HypScopes, ptup[3], ptup[2]) and ptup[2] not in D.NotOutscopable:
        return D.outscopeArgWith( HypScopes, ptup[3], ptup[2], step+1 )
      elif not D.reachesInChain( HypScopes, ptup[1], D.ceiling(ptup[2]) ) and not D.reachesInChain( HypScopes, ptup[2], D.ceiling(ptup[3]) ) and ptup[2] not in D.NotOutscopable:
        return D.outscopeWith( HypScopes, ptup[1], ptup[2], step+1 ) + D.outscopeArgWith( HypScopes, ptup[2], ptup[3], step+1 )
      elif not D.reachesInChain( HypScopes, ptup[1], D.ceiling(ptup[3]) ) and not D.reachesInChain( HypScopes, ptup[3], D.ceiling(ptup[2]) ) and ptup[3] not in D.NotOutscopable:
        return D.outscopeWith( HypScopes, ptup[1], ptup[3], step+1 ) + D.outscopeArgWith( HypScopes, ptup[3], ptup[2], step+1 )
    else:
      print( 'ERROR: ptup too large: ' + str(ptup) )
      exit(1)
  '''

  def satisfyPred( D, HypScopes, ptup, xSplice, step ):
    print( '  '*step + str(step) + ': trying to satisfy pred tuple ' + ' '.join(ptup) + ' for ' + xSplice + '...' )
    ## For unary predicates...
    if len(ptup) == 3:
      ## If elem pred already outscoped by arg, do nothing...
      if   D.reachesInChain( HypScopes, ptup[1], ptup[2] ): return []
      ## If arg already outscopes splice, scope elem pred to splice...
      elif D.reachesInChain( HypScopes, xSplice, ptup[2] ): return [ (ptup[1],xSplice) ] if xSplice!='' else []
      ## If arg is elem pred, recurse to that pred...
      elif ptup[2] in D.PredToTuple:                        return [ (ptup[1],ptup[2]) ] + D.satisfyPred( HypScopes, D.PredToTuple[ptup[2]], xSplice, step+1 )
      elif D.ceiling(ptup[2]) not in D.AnnotatedCeilings:   return [ (ptup[1],ptup[2]), (ptup[2],xSplice) ] if xSplice!='' else [ (ptup[1],ptup[2]) ] 
      else:                                                 print( 'ERROR: unary could not scope: ' + ' '.join(ptup) ) 
    ## For binary predicates...
    if len(ptup) == 4:
      ## If elem pred already outscoped by both args, do nothing...
      if   D.reachesInChain( HypScopes, ptup[1], ptup[2] ) and D.reachesInChain( HypScopes, ptup[1], ptup[3] ): return []
      ## If 1st arg already outscopes splice and 2nd arg already outscopes 1st arg, scope elem pred to 2nd arg...
      elif D.reachesInChain( HypScopes, xSplice, ptup[2] ) and D.reachesInChain( HypScopes, ptup[3], ptup[2] ): return [ (ptup[1],ptup[3]) ]
      ## If 2nd arg already outscopes splice and 1st arg already outscopes 2nd arg, scope elem pred to 1st arg...
      elif D.reachesInChain( HypScopes, xSplice, ptup[3] ) and D.reachesInChain( HypScopes, ptup[2], ptup[3] ): return [ (ptup[1],ptup[2]) ]
      ## If 1st arg already outscopes splice and 2nd arg already outscopes 1st arg, scope elem pred to 2nd arg...
      elif xSplice=='' and D.ceiling(ptup[2]) in D.AnnotatedCeilings and D.reachesInChain( HypScopes, ptup[3], ptup[2] ): return [ (ptup[1],ptup[3]) ]
      ## If 2nd arg already outscopes splice and 1st arg already outscopes 2nd arg, scope elem pred to 1st arg...
      elif xSplice=='' and D.ceiling(ptup[3]) in D.AnnotatedCeilings and D.reachesInChain( HypScopes, ptup[2], ptup[3] ): return [ (ptup[1],ptup[2]) ]
      ## If 1st arg already outscopes splice and 2nd arg is elem pred...
      elif D.reachesInChain( HypScopes, xSplice, ptup[2] ) and ptup[3] in D.PredToTuple:  return [ (ptup[1],ptup[3]) ] + D.satisfyPred( HypScopes, D.PredToTuple[ptup[3]], xSplice, step+1 )
      ## If 2nd arg already outscopes splice and 1st arg is elem pred...
      elif D.reachesInChain( HypScopes, xSplice, ptup[3] ) and ptup[2] in D.PredToTuple:  return [ (ptup[1],ptup[2]) ] + D.satisfyPred( HypScopes, D.PredToTuple[ptup[2]], xSplice, step+1 )
      ## If 1st arg already outscopes splice and 2nd arg is elem pred...
      elif D.reachesInChain( HypScopes, xSplice, ptup[2] ) and D.ceiling(ptup[3]) not in D.AnnotatedCeilings:  return [ (ptup[1],ptup[3]), (ptup[3],xSplice) ]
      ## If 2nd arg already outscopes splice and 1st arg is elem pred...
      elif D.reachesInChain( HypScopes, xSplice, ptup[3] ) and D.ceiling(ptup[2]) not in D.AnnotatedCeilings:  return [ (ptup[1],ptup[2]), (ptup[2],xSplice) ]
      elif D.ceiling(ptup[2]) not in D.AnnotatedCeilings and ptup[3] in D.PredToTuple:  return [ (ptup[1],ptup[2]), (ptup[2],ptup[3]) ] + D.satisfyPred( HypScopes, D.PredToTuple[ptup[3]], xSplice, step+1 )
      elif D.ceiling(ptup[3]) not in D.AnnotatedCeilings and ptup[2] in D.PredToTuple:  return [ (ptup[1],ptup[3]), (ptup[3],ptup[2]) ] + D.satisfyPred( HypScopes, D.PredToTuple[ptup[2]], xSplice, step+1 )
      elif D.ceiling(ptup[2]) not in D.AnnotatedCeilings and ptup[3] in D.getChainFromSup( ptup[2] ): return [ (ptup[1],ptup[3]), (ptup[3],xSplice) ]
      elif D.ceiling(ptup[3]) not in D.AnnotatedCeilings and ptup[2] in D.getChainFromSup( ptup[3] ): return [ (ptup[1],ptup[2]), (ptup[2],xSplice) ]
      elif D.ceiling(ptup[2]) not in D.AnnotatedCeilings and D.reachesInChain( HypScopes, ptup[3], D.ceiling(ptup[2]) ): return [ (ptup[1],ptup[3]), (ptup[3],xSplice) ]
      elif D.ceiling(ptup[2]) not in D.AnnotatedCeilings and D.ceiling(ptup[3]) not in D.AnnotatedCeilings: return [ (ptup[1],ptup[3]), (ptup[3],ptup[2]), (ptup[2],xSplice) ]
      else:                                                                                      print( 'ERROR: binary could not scope: ' + ' '.join(ptup) )

  def tryScope( D, HypScopes, RecencyConnected, step=1 ):
#    active = True
#    while active:
      print( RecencyConnected )
      active = False
      l = []
      for _,xHiOrig in RecencyConnected[:]:
        print( '  ' + D.strGraph() )
        print( '  '*step + str(step) + ': working on refstate ' + str(xHiOrig) + '...' )
        for ptup,xSplice in D.RefToPredTuples.get( xHiOrig, [] ) + ( [ ( D.PredToTuple[xHiOrig], '' ) ] if xHiOrig in D.PredToTuple else [] ):
          l = D.satisfyPred( D.Scopes, ptup, xSplice, step+1 )
          print( '  '*step + str(step) + '  l=' + str(l) )
          for xLo,xHi in l:
            print( '  '*step + str(step) + '  scoping ' + D.ceiling(xLo) + ' to ' + xHi )
            D.Scopes[ D.ceiling(xLo) ] = xHi
            RecencyConnected = [ (step,x) for x in D.Chains.get(xLo,[]) ] + RecencyConnected
          if l!=[]: D.tryScope( HypScopes, RecencyConnected, step+1 )
#            active = True

  '''
  Heirs = Subs.deepcopy()
  active = True
  while active:
    active = False
    for xHi,lLo in Heirs.items:
      for xLo in lLo:
        if xLo in Subs:
          Heirs[ xHi ] += Subs[ xLo ]
          active = True
  '''

  '''
  ## Deterministically add scopes...
  active = True
  while active:
    ## Calculate ceilings of scoped refts...
    AnnotatedCeilings = sets.Set([ ceiling(x) for x in Scopes.keys() ])
    if VERBOSE: print( 'AnnotatedCeilings = ' + str(AnnotatedCeilings) )
    ## List of original (dominant) refts...
    HighAnnotated = sets.Set([ x for x in Referents if ceiling(x) in AnnotatedCeilings ])  # | sets.Set([ ceiling(x) for x in Scopes.values() ])
    if VERBOSE: print( 'HighAnnotated = ' + str(HighAnnotated) )

    active = False
    for pred in PredTuples:
      if pred[1] not in HeirsOfParticipants:
        for xAnn in pred[2:]:
          if xAnn in HighAnnotated:
            if len(pred) == 3 and pred[1] not in Scopes:
              if VERBOSE: print( 'Unconstrained elementary predicate ' + pred[0] + ' ' + pred[1] + ' deterministically binding self to annotated participant ' + xAnn )
              Scopes[ pred[1] ] = xAnn
              active = True
            if len(pred) == 4:
              for xNotAnn in pred[2:]:
                if xAnn != xNotAnn and xNotAnn not in HighAnnotated and pred[1] not in Scopes:
                  if VERBOSE: print( 'Unconstrained elementary predicate ' + pred[0] + ' ' + pred[1] + ' deterministically binding self to non-annotated participant ' + xNotAnn + ' given other participant ' + xAnn + ' is annotated.' )
                  Scopes[ pred[1] ] = xNotAnn
                  active = True
  '''

#    if len(pred) == 3:
#      print( 'Deterministically scoping unary elementary predicate ' + pred[0] + ' ' + pred[1] + ' to participant ' + pred[2] )
#      Scopes[ pred[1] ] = pred[2]
#    elif pred[2] in PorQs:
#      print( 'Deterministically scoping elementary predicate ' + pred[0] + ' ' + pred[1] + ' to elementary predicate participant ' + pred[2] )
#      Scopes[ pred[1] ] = pred[2]
#    elif pred[3] in PorQs:
#      print( 'Deterministically scoping elementary predicate ' + pred[0] + ' ' + pred[1] + ' to elementary predicate participant ' + pred[3] )
#      Scopes[ pred[1] ] = pred[3]

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

  '''
  ## Recursive function to search space of scopings...
  def tryScope( HypScopes, RecencyConnected, step=1 ):

    if VERBOSE: print( '  '*step + 'HypScopes = ' + str(sorted(HypScopes.items())) )
    if VERBOSE: print( '  '*step + 'RecencyConnected = ' + str(RecencyConnected) )
    if VERBOSE: print( '  '*step + D.strGraph(HypScopes) ) #strGraph( PredTuples, QuantTuples, Inhs, HypScopes ) )

#    if step > 8: exit(0)

    ## Helper function to determine if one ref state outscopes another
    def reachesFromSup( xLo, xHi ):
#      if step==36: print( '  '*step + 'reachesFromSup ' + xLo + ' ' + xHi )
      return True if xLo in D.Chains.get(xHi,[]) else reachesInChain( HypScopes[xLo], xHi ) if xLo in HypScopes else any( [ reachesFromSup(xSup,xHi) for l,xSup in D.Inhs.get(xLo,{}).items() if l!='w' ] )
    def reachesFromSub( xLo, xHi ):
#      if step==36: print( '  '*step + 'reachesFromSub ' + xLo + ' ' + xHi )
      return True if xLo in D.Chains.get(xHi,[]) else reachesInChain( HypScopes[xLo], xHi ) if xLo in HypScopes else any( [ reachesFromSub(xSub,xHi) for xSub in D.Subs.get(xLo,[]) ] )
    def reachesInChain( xLo, xHi ):
#      if step==36: print( '  '*step + 'reachesInChain ' + xLo + ' ' + xHi )
      return reachesFromSup( xLo, xHi ) or reachesFromSub( xLo, xHi )

#    ## Helper functions to explore inheritance chain...
#    def outscopingFromSup( xLo ):
#      return True if xLo in HypScopes.values() else any( [ outscopingFromSup(xHi) for l,xHi in Inhs.get(xLo,{}).items() if l!='w' ] )
#    def outscopingFromSub( xHi ):
#      return True if xHi in HypScopes.values() else any( [ outscopingFromSub(xLo) for xLo in Subs.get(xHi,[]) ] )
#    def outscopingInChain( x ):
#      return outscopingFromSup( x ) or outscopingFromSub( x )

    ## Scope Ceiling...
    def getCeilingFromSup( xLo ):
      return getCeilingInChain( HypScopes[xLo] ) if xLo in HypScopes else sets.Set( [ y  for l,xHi in D.Inhs.get(xLo,{}).items() if l!='w'  for y in getCeilingFromSup(xHi) ] )
    def getCeilingFromSub( xHi ):
      return getCeilingInChain( HypScopes[xHi] ) if xHi in HypScopes else sets.Set( [ y  for xLo in D.Subs.get(xHi,[])  for y in getCeilingFromSub(xLo) ] )
    def getCeilingInChain( x ):
      out = getCeilingFromSup( x ) | getCeilingFromSub( x )
      return out if len(out)>0 else sets.Set( [x] )
    def ceiling( x ):
      y = sorted( getCeilingInChain(x) )[0]
      return y if y in D.NuscoValues or y not in D.Nuscos else D.Nuscos[y][0]

    xHiOrig = RecencyConnected[0][1]
    ## Ensure no elem pred with unbound arguments...
    for xHi in D.Chains[ xHiOrig ]:
      for ptup in D.PredTuples:
        if xHi == ptup[1] and not all([ reachesInChain(xHi,x) for x in ptup[2:] ]):
          if VERBOSE: print( '  '*step + str(step) + ': dead end -- ' + xHiOrig + ' is elementary predication ' + xHi + ' with unbound arguments.' )
          return None
    for ptup,xHi in D.RefToPredTuples.get( xHiOrig, [] ):
      if not reachesInChain( ptup[1], xHi ) and reachesInChain( xHi, ceiling(ptup[1]) ):
        if VERBOSE: print( '  '*step + str(step) + ': dead end -- ' + xHiOrig + ' aka ' + xHi + ' is coapical with but does not outscope elementary predication ' + ptup[1] + '.' )
        return None

    unsatisfied = False
    for _,xHiOrig in RecencyConnected:
      ## Recurse...
      for ptup,xHi in D.RefToPredTuples[ xHiOrig ]:
        for xLo in reversed(ptup[1:]):
#          print('  '*step + 'gonna call reaches on ' + xLo + ' ' + xHi )
          if xLo == xHi or xLo in NotOutscopable:
            continue  ## Don't scope to ones self, or to protected attested annotations.
          elif reachesInChain( xLo, xHi ):
            if VERBOSE: print( '  '*step + str(step) + ': ' + xHi + ' outscopes ' + xLo + ' to satisfy ' + ptup[0] + ' ' + ptup[1] + '.' )
          elif reachesInChain( xHi, xLo ):
            if VERBOSE: print( '  '*step + str(step) + ': ' + xLo + ' outscopes ' + xHi + ' to satisfy ' + ptup[0] + ' ' + ptup[1] + '.' )
          else:
            unsatisfied = True
#            print('  '*step + 'gonna call ceiling/reaches on ' + xLo + ' ' + xHi )
            if xLo == ptup[1] and reachesInChain( xHi, ceiling(xLo) ):   #ceiling( xLo ) == ceiling( xHi ):
              if VERBOSE: print( '  '*step + str(step) + ': dead end -- ' + xHi + ' coapical with but does not outscope elementary predication ' + xLo + ' to satisfy ' + ptup[0] + ' ' + ptup[1] + '.' )
              return None
            elif xLo == ptup[1] and not all([ reachesInChain(xHi,x) for x in ptup[2:] if x!=xHi ]):
              if VERBOSE: print( '  '*step + str(step) + ': ' + xHi + ' cannot outscope ' + xLo + ' to satisfy ' + ptup[0] + ' ' + ptup[1] + ' because other arguments would be excluded from scope chain.' )
            elif reachesInChain( xHi, ceiling(xLo) ):  #ceiling( xLo ) == ceiling( xHi ):
              if VERBOSE: print( '  '*step + str(step) + ': ' + xLo + ' coapical with ' + xHi + ', lacking scope chain.' )
            else:
              if VERBOSE: print( '  '*step + str(step) + ': ' + xHi + ' does not outscope ' + xLo + ' to satisfy ' + ptup[0] + ' ' + ptup[1] + ', try ceiling ' + xLo + ' = ' + ceiling(xLo) + ' to ' + xHi + '...' )
              AppendedHypScopes = HypScopes.copy()
              AppendedHypScopes[ ceiling(xLo) ] = xHi
              OutputScopes = tryScope( AppendedHypScopes, [ (step,xLo) ] + RecencyConnected, step+1 )
              if OutputScopes != None: return OutputScopes
#    unsatisfied = False
#    for pred in PredTuples:  ## Note: for speed, try unary first
##      if len(pred) > 3:
#        for xHi in pred[2:]:
#          if reachesInChain( pred[1], xHi ):
#            if VERBOSE: print( '  '*step + str(step) + ': ' + pred[0] + ' ' + pred[1] + ' bound by ' + xHi + '.' )
#          else:
#            unsatisfied = True
#            if xHi in HighAnnotated and any([not reachesInChain(pred[1],x) and x not in HighAnnotated for x in pred[2:] if x!=xHi]):
#              if VERBOSE: print( '  '*step + str(step) + ': ' + pred[0] + ' ' + pred[1] + ' dispreferred to annotated ' + xHi + ' when other participants not bound: ' + ','.join([x for x in pred[2:] if x!=xHi and not reachesInChain(pred[1],x) and x not in HighAnnotated ]) + '.' )
#            else:
#              if ceiling(pred[1]) == ceiling(xHi):
#                if VERBOSE: print( '  '*step + str(step) + ': dead end -- ' + pred[1] + ' coapical with but not bound by ' + xHi + '.' )
#                return None
#              else:
#                if VERBOSE: print( '  '*step + str(step) + ': ' + pred[0] + ' ' + pred[1] + ' not bound by ' + xHi + ', try ceiling ' + pred[1] + ' = ' + ceiling(pred[1]) + ' to ' + xHi + '...' )
#                AppendedHypScopes = HypScopes.copy()
#                AppendedHypScopes[ ceiling(pred[1]) ] = xHi
#                OutputScopes = tryScope( AppendedHypScopes, step+1 )
#                if OutputScopes != None: return OutputScopes
    if not unsatisfied:
      if VERBOSE: print( 'Found scoping:' )
      if VERBOSE: print( HypScopes )
      return HypScopes
    else:
      print( 'Failed.' )
      print( D.strGraph( HypScopes ) )
      exit( 1 )
    return None
  '''

  if VERBOSE: print( 'running tryScope...' )
#  D.Scopes = tryScope( D.Scopes, RecencyConnected )
  D.tryScope( D.Scopes, RecencyConnected )
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

  '''
  ## Propagate scopes down inheritance chains...
  active = True
  while active:
    active = False
    for xHi in Scopes.keys():
      for xLo in Subs.get(xHi,[]):
        if xLo not in Scopes:
          if VERBOSE: print( 'Inheriting scope parent ' + Scopes[xHi] + ' from ' + xHi + ' to ' + xLo + '.' )
          Scopes[ xLo ] = Scopes[ xHi ]
          active = True
  ## Propagate quants down inheritance chains...
  active = True
  while active:
    active = False
    for q,e,r,xHi,n in QuantTuples[:]:
      for xLo in Subs.get(xHi,[]):
        if xLo not in [s for _,_,_,s,_ in QuantTuples]:
          if VERBOSE: print( 'Inheriting quant ' + q + ' from ' + xHi + ' to ' + xLo + '.' )
          QuantTuples.append( (q,e,r,xLo,n) )
          active = True
  ## Clean up abstract scopes...
  for xHi in Scopes.keys():
    if xHi in Subs: #for xLo in Subs.get(xHi,[]):
      if VERBOSE: print( 'Removing redundant abstract scope parent ' + Scopes[xHi] + ' from ' + xHi + ' because of inheritance at ' + str(Subs[xHi]) )
      del Scopes[xHi]
  ## Clean up abstract quants...
  if VERBOSE: print( 'Subs = ' + str(Subs) )
  for q,e,r,s,n in QuantTuples[:]:
    if s in Subs:
      if VERBOSE: print( 'Removing redundant abstract quant ' + q + ' from ' + s + ' because of inheritance at ' + Subs[s][0] )
      QuantTuples.remove( (q,e,r,s,n) )
  if VERBOSE: print( 'QuantTuples = ' + str(QuantTuples) )
  '''

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

  for expr in Translations:
    print( lambdaFormat(expr) )
    findUnboundVars( expr )
    checkConstsUsed( expr, D.OrigConsts )
  for k in D.OrigConsts:
    sys.stderr.write( 'WARNING: const does not appear in translations: ' + k + '\n' )
    print(            'WARNING: const does not appear in translations: ' + k )

