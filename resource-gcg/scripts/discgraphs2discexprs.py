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
import copy
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
import discgraph
import induciblediscgraph

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
      sys.stderr.write( '    DOWNSTREAM LAMBDA EXPRESSION ERROR: unbound var: ' + expr + '\n' )
      print(           '#    DOWNSTREAM LAMBDA EXPRESSION ERROR: unbound var: ' + expr  )
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

discctr = 0

## For each discourse graph...
for line in sys.stdin:

  discctr += 1
  DiscTitle = sorted([ asc  for asc in line.split(' ')  if ',0,' in asc and asc.startswith('000') ])
  print(            '#DISCOURSE ' + str(discctr) + '... (' + ' '.join(DiscTitle) + ')' )
  sys.stderr.write( '#DISCOURSE ' + str(discctr) + '... (' + ' '.join(DiscTitle) + ')\n' )

  #### I. READ IN AND PREPROCESS DISCOURSE GRAPH...

  line = line.rstrip()
  if VERBOSE: print( 'GRAPH: ' + line )

  D = discgraph.DiscGraph( line )
  if not D.check(): continue
  D.checkMultipleOutscopers()

  OrigScopes = D.Scopes

  D = induciblediscgraph.InducibleDiscGraph( line )

  #### II. ENFORCE NORMAL FORM (QUANTS AND SCOPE PARENTS AT MOST SPECIFIC INHERITANCES...

  D.normForm()
  ## Copy quants down to final heirs -- NOTE: this is the same as using inheritance in Q rules...
  for q,e,r,x,n in D.QuantTuples[:]:
    for xFin in D.Heirs.get(x,[]):
      if xFin not in D.Subs  and  xFin not in [s for _,_,_,s,_ in D.QuantTuples]:
        D.QuantTuples.append( (q,e,r,xFin,n) )
  ## Copy scopes down to final heirs -- NOTE: this is the same as using inheritance in S rules...
  for x in D.Scopes.keys():
    for xFin in D.Heirs.get(x,[]):
      if xFin not in D.Subs  and  xFin not in D.Scopes:
        D.Scopes[ xFin ] = D.Scopes[ x ]

  #### III. INDUCE UNANNOTATED SCOPES AND EXISTENTIAL QUANTS...

  ## Add dummy args below eventualities...
  for xt in D.PredTuples:
    for x in xt[2:]:
      if len( D.ConstrainingTuples.get(x,[]) )==1 and x.endswith('\''):   #x.startswith(xt[1][0:4] + 's') and x.endswith('\''):
        D.Scopes[x] = xt[1]
        if VERBOSE: print( 'Scoping dummy argument ' + x + ' to predicate ' + xt[1] )

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


  L1 = [ x  for x in sorted((sets.Set(D.Referents) | sets.Set(D.Subs)) - sets.Set(D.Inhs.keys()))  if any([ y in D.Chains.get(x,[])  for y in OrigScopes.values() ]) and not any([ y in D.Chains.get(x,[])  for y in OrigScopes ]) ]
  if len(L1) > 1:
    print(           '#WARNING: Discourse scope annotations do not converge to single top-level ancestor: ' + ' '.join(L1) + ' -- possibly due to missing anaphora between sentences' )
    sys.stderr.write( 'WARNING: Discourse scope annotations do not converge to single top-level ancestor: ' + ' '.join(L1) + ' -- possibly due to missing anaphora between sentences\n' ) 
  elif L1 == []:
    L2 = [ x  for x in sorted((sets.Set(D.Referents) | sets.Set(D.Subs)) - sets.Set(D.Inhs.keys()))  if any([ r in D.Chains.get(x,[])  for q,e,n,r,s in D.QuantTuples ]) and not any([ y in D.Chains.get(x,[])  for y in OrigScopes ]) ]
    print(           '#NOTE: Discourse contains no scope annotations -- defaulting to legators of explicit quantifiers: ' + ' '.join(L2) )
    sys.stderr.write( 'NOTE: Discourse contains no scope annotations -- defaulting to legators of explicit quantifiers: ' + ' '.join(L2) + '\n' ) 
    if L2 == []:
#      L = [ x  for x in sorted((sets.Set(D.Referents) | sets.Set(D.Subs)) - sets.Set(D.Inhs.keys()))  if not any([ y in D.Chains.get(x,[])  for y in OrigScopes ]) ]
      print(           '#WARNING: No explicit quantifiers annotated -- instead iterating over all legator referents' )
      sys.stderr.write( 'WARNING: No explicit quantifiers annotated -- instead iterating over all legator referents\n' )

  ## List of original (dominant) refts...
#  RecencyConnected = sorted( [ ((0 if x not in D.Subs else -1) + (0 if x in ScopeLeaves else -2),x)  for x in D.Referents  if D.ceiling(x) in D.Chains.get(L[0],[]) ], reverse = True )   # | sets.Set([ ceiling(x) for x in Scopes.values() ])
  RecencyConnected = [ y  for x in L1  for y in D.Referents  if D.ceiling(y) in D.Chains.get(x,[]) ]
  if VERBOSE: print( 'RecencyConnected = ' + str(RecencyConnected) )


##  D.Scopes = tryScope( D.Scopes, RecencyConnected )
#  D.tryScope( RecencyConnected, False )
#  if VERBOSE: print( 're-running tryScope...' )
#  RecencyConnected = [ (0,x)  for x in D.Referents  if D.ceiling(x) in L ]
  ok = True
  Complete = []
  while ok:
    ## Try using increasingly coarse sets of top-level scopes, starting with top of annotated scopes (preferred)...
    L = [ x  for x in sorted(sets.Set(D.Referents) - sets.Set(Complete) - sets.Set(D.Inhs.keys()))  if any([ y in D.Chains.get(x,[])  for y in OrigScopes.values() ]) and not any([ y in D.Chains.get(x,[])  for y in OrigScopes ]) ]
    if VERBOSE and L != []: print( 'Legators as roots of annotated scope: ' + str(L) )
    ## Back off to explicitly annotated quantifiers (preferred)...
    if L == []:
      L = [ x  for x in sorted(sets.Set(D.Referents) - sets.Set(Complete) - sets.Set(D.Inhs.keys()))  if any([ r in D.Chains.get(x,[])  for q,e,n,r,s in D.QuantTuples ]) and not any([ y in D.Chains.get(x,[])  for y in D.Scopes ]) ]

      if VERBOSE and L != []: print( 'Legators as explicit quantifiers: ' + str(L) )
    ## Back off to any legator (dispreferred)...
    if L == []:
      L = [ x  for x in sorted(sets.Set(D.Referents) - sets.Set(Complete) - sets.Set(D.Inhs.keys()))  if any([ y in D.Chains.get(x,[])  for tup in D.PredTuples  for y in tup[1:] ]) and not any([ y in D.Chains.get(x,[])  for y in D.Scopes ]) ]
      if VERBOSE and L != []: print( 'Legators as explicit quantifiers: ' + str(L) )
      if L != []:
        print(           '#WARNING: Insufficient explicitly annotated quantifiers, backing off to full set of legators: ' + str(L) )
        sys.stderr.write( 'WARNING: Insufficient explicitly annotated quantifiers, backing off to full set of legators: ' + str(L) + '\n' )
    ## Exit if no uncompleted legators...
    if L == []: break

    if VERBOSE: print( 'Trying to induce scopes below ' + L[0] + '...' )
    RecencyConnected += D.Chains.get(L[0],[])    ## Account target as connected (root).
    ok = D.tryScope( L[0], RecencyConnected, True )
    Complete.append( L[0] )

  if not ok: continue
#  out = D.tryScope( RecencyConnected, True )
#  if out == False: continue
  if VERBOSE: print( D.Scopes )
  if VERBOSE: print( 'GRAPH: ' + D.strGraph() )

  for xTarget in sets.Set( D.Scopes.values() ):
    if not any([ x in D.Scopes  for x in D.Chains.get(xTarget,[]) ]) and not any([ s in D.Chains.get(xTarget,[])  for q,e,r,s,n in D.QuantTuples ]):
      print(           '#ERROR: Top-scoping referent ' + xTarget + ' has no annotated quantifier, and will not be induced!' )
      sys.stderr.write( 'ERROR: Top-scoping referent ' + xTarget + ' has no annotated quantifier, and will not be induced!\n' )

#  DisjointPreds = sets.Set([ ( D.ceiling(xt[1]), D.ceiling(yt[1]) )  for xt in D.PredTuples  for yt in D.PredTuples  if xt[1] < yt[1] and not D.reachesInChain( xt[1], D.ceiling(yt[1]) ) ])
#  if len(DisjointPreds) > 0:
#    print(           '#WARNING: Scopal maxima not connected, possibly due to missing anaphora between sentences: ' + str(DisjointPreds) )
#    sys.stderr.write( 'WARNING: Scopal maxima not connected, possibly due to missing anaphora between sentences: ' + str(DisjointPreds) + '\n' )

#  DisjointRefts = sets.Set([ ( D.ceiling(x), D.ceiling(y) )  for xt in D.PredTuples  for x in xt[1:]  for yt in D.PredTuples  for y in yt[1:]  if x < y and not D.reachesInChain( x, D.ceiling(y) ) ])
#  if len(DisjointRefts) > 0:
#    print(           '#WARNING: Scopal maxima not connected, possibly due to missing anaphora between sentences or unscoped argument of scoped predicate: ' + str(DisjointRefts) )
#    sys.stderr.write( 'WARNING: Scopal maxima not connected, possibly due to missing anaphora between sentences or unscoped argument of scoped predicate: ' + str(DisjointRefts) + '\n' )

  ## Copy lowest (if multiple) scopes down chain to final heirs -- NOTE: this is the same as using inheritance in S rules...
  for x in D.Scopes.keys():
    if not any([ y in D.Scopes  for y in D.Heirs.get(x,[])  if y != x ]):
      for xFin in D.Heirs.get(x,[]):
        if xFin not in D.Subs  and  xFin not in D.Scopes:
          D.Scopes[ xFin ] = D.Scopes[ x ]
  ## Copy quants down to final heirs -- NOTE: this is the same as using inheritance in Q rules...
  for q,e,r,x,n in D.QuantTuples[:]:
    if not any([ y in D.Scopes  for y in D.Heirs.get(x,[])  if y != x ]):
      for xFin in D.Heirs.get(x,[]):
        if xFin not in D.Subs  and  xFin not in [s for _,_,_,s,_ in D.QuantTuples]:
          D.QuantTuples.append( (q,e,r,xFin,n) )
  if VERBOSE: print( 'GRAPH: ' + D.strGraph() )

  ## Induce low existential quants when only scope annotated...
#  for xCh in sorted([x if x in NuscoValues else Nuscos[x] for x in Scopes.keys()] + [x for x in Scopes.values() if x in NuscoValues]):  #sorted([ s for s in NuscoValues if 'r' not in Inhs.get(Inhs.get(s,{}).get('r',''),{}) ]): #Scopes:
#  ScopeyNuscos = [ x for x in NuscoValues if 'r' not in Inhs.get(Inhs.get(x,{}).get('r',''),{}) and (x in Scopes.keys()+Scopes.values() or Inhs.get(x,{}).get('r','') in Scopes.keys()+Scopes.values()) ]
#  ScopeyNuscos = [ x for x in D.Referents | sets.Set(D.Inhs.keys()) if (x not in D.Nuscos or x in D.NuscoValues) and 'r' not in D.Inhs.get(D.Inhs.get(x,{}).get('r',''),{}) and (x in D.Scopes.keys()+D.Scopes.values() or D.Inhs.get(x,{}).get('r','') in D.Scopes.keys()+D.Scopes.values()) ]
  ScopeyNuscos = D.Scopes.keys()
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
  ## Remove redundant non-terminal quants with no scope parent...
  for q,e,r,s,n in D.QuantTuples[:]:
    if s in D.Subs and s not in D.Scopes and any([ x in D.Heirs.get(s,[])  for _,_,_,x,_ in D.QuantTuples if x!=s ]):
      D.QuantTuples.remove( (q,e,r,s,n) )
      if VERBOSE: print( 'Removing non-terminal quantifier ' + q + ' ' + e + ' ' + r + ' ' + s + ' ' + n )

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

    '''
    ## P rule...
    for ptup in list(D.PredTuples):
      for x in ptup[1:]:                                  ## extended here below
        if x not in D.Scopes.values() and x not in D.Inhs: # and not any([ y in D.Scopes.values() for y in D.Chains.get(x,[]) ]):
          if VERBOSE: print( 'applying P to make \\' + x + '. ' + lambdaFormat(ptup) )
          Abstractions[ x ].append( ptup )
          if ptup in D.PredTuples: D.PredTuples.remove( ptup )
          active = True
    '''
    ## P1 rule...
    for ptup in list(D.PredTuples):
      x = ptup[1]  
      if x not in D.Scopes.values() and x not in D.Inhs:
        if VERBOSE: print( 'applying P to move from P to A: \\' + x + '. ' + lambdaFormat(ptup) )
        Abstractions[ x ].append( ptup )
        if ptup in D.PredTuples: D.PredTuples.remove( ptup )
        active = True
    ## P2 rule...
    for ptup in list(D.PredTuples):
      for x in ptup[2:]:
        if D.Scopes.get(x,'')==ptup[1] and x not in D.Scopes.values() and x not in D.Inhs:
          if VERBOSE: print( 'applying P to move from P to A: \\' + x + '. ' + lambdaFormat(ptup) )
          Abstractions[ x ].append( ptup )
          if ptup in D.PredTuples: D.PredTuples.remove( ptup )
          active = True

    ## C rule...
    for var,Structs in Abstractions.items():
      if len(Structs) > 1:
        if VERBOSE: print( 'applying C to add from A to A: \\' + var + '. ' + lambdaFormat( tuple( ['and'] + Structs ) ) )
        Abstractions[var] = [ tuple( ['and'] + Structs ) ]
        active = True

    ## M rule...
    for var,Structs in Abstractions.items():
      if len(Structs) == 1 and var not in D.Scopes.values() and var not in D.Inhs:
        if VERBOSE: print( 'applying M to move from A to E: \\' + var + '. ' + lambdaFormat(Structs[0]) )
        Expressions[var] = Structs[0]
        del Abstractions[var]
        active = True
 
    ## Q rule...
    for q,e,r,s,n in list(D.QuantTuples):
      if r in Expressions and s in Expressions:
        if VERBOSE: print( 'applying Q to move from Q to T: ' + lambdaFormat( ( q, n, ( 'lambda', r, Expressions[r] ), ( 'lambda', s, Expressions[s] ) ) ) )   ## (' + q + ' (\\' + r + '. ' + str(Expressions[r]) + ') (\\' + s + '. ' + str(Expressions[s]) + '))' )
        Translations.append( ( q, n, ( 'lambda', r, Expressions[r] ), ( 'lambda', s, Expressions[s] ) ) )
        D.QuantTuples.remove( (q, e, r, s, n) )
        active = True

    ## I1 rule...
    for src,lbldst in D.Inhs.items():
      for lbl,dst in lbldst.items():
        if dst not in D.Inhs and dst not in Abstractions and dst not in Expressions and dst not in D.Scopes.values() and dst not in [ x for ptup in D.PredTuples for x in ptup ]:
          if VERBOSE: print( 'applying I1 to add to A: \\' + dst + ' True' )
          Abstractions[ dst ].append( () )
          active = True

    ## I2,I3,I4 rule...
    for src,lbldst in D.Inhs.items():
      for lbl,dst in lbldst.items():
        if dst in Expressions:
          if src in D.Scopes and dst in D.Traces and D.Scopes[src] in D.Scopes and D.Traces[dst] in D.Traces:
            Abstractions[ src ].append( replaceVarName( replaceVarName( replaceVarName( Expressions[dst], dst, src ), D.Traces[dst], D.Scopes[src] ), D.Traces[D.Traces[dst]], D.Scopes[D.Scopes[src]] ) )    ## I4 rule.
            if VERBOSE: print( 'applying I4 to add from A to A replacing ' + dst + ' with ' + src + ' and ' + D.Traces[dst] + ' with ' + D.Scopes[src] + ' and ' + D.Traces[D.Traces[dst]] + ' with ' + D.Scopes[D.Scopes[src]] + ' to make \\' + src + ' ' + lambdaFormat(Abstractions[src][-1]) )
#            del Traces[dst]
          elif src in D.Scopes and dst in D.Traces:
            Abstractions[ src ].append( replaceVarName( replaceVarName( Expressions[dst], dst, src ), D.Traces[dst], D.Scopes[src] ) )    ## I4 rule.
            if VERBOSE: print( 'applying I4 to add from A to A replacing ' + dst + ' with ' + src + ' and ' + D.Traces[dst] + ' with ' + D.Scopes[src] + ' to make \\' + src + ' ' + lambdaFormat(Abstractions[src][-1]) )
#            del Traces[dst]
          else:
            if VERBOSE: print( 'applying I2/I3 to add from A to A replacing ' + dst + ' with ' + src + ' to make \\' + src + ' ' + lambdaFormat(replaceVarName( Expressions[dst], dst, src )) )   #' in ' + str(Expressions[dst]) )
            Abstractions[ src ].append( replaceVarName( Expressions[dst], dst, src ) )
            if dst in D.Scopes and src not in D.Scopes and D.Nuscos.get(src,[''])[0] not in D.Scopes and src in [s for q,e,r,s,n in D.QuantTuples] + [r for q,e,r,s,n in D.QuantTuples]:  D.Scopes[src if src in D.NuscoValues else D.Nuscos[src][0]] = D.Scopes[dst]     ## I3 rule.
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
        if VERBOSE: print( 'applying S1 to move from T to A: (\\' + D.Scopes[ S[1] ] + ' ' + q + ' ' + str(R) + ' ' + str(S) + ')' )
        Expr = copy.deepcopy( (q,n,R,S) )
        for x in D.Chains.get( D.Scopes[ S[1] ], [] ):
          if x != D.Scopes[ S[1] ]:
            Expr = replaceVarName( Expr, x, D.Scopes[ S[1] ] )
        Abstractions[ D.Scopes[ S[1] ] ].append( Expr )
        del D.Scopes[ S[1] ]
#        if R[1] in Scopes: del Scopes[ R[1] ]   ## Should use 't' trace assoc.
        Translations.remove( (q, n, R, S) )
        active = True

  expr = tuple( ['and'] + Translations )
  print( lambdaFormat(expr) )
#  for expr in Translations:
  findUnboundVars( expr )
  if VERBOSE: print( 'D.OrigConsts = ' + str(D.OrigConsts) )
  checkConstsUsed( expr, D.OrigConsts )
  for k in D.OrigConsts:
    print(           '#    DOWNSTREAM LAMBDA EXPRESSION WARNING: const does not appear in translations: ' + k )
    sys.stderr.write( '    DOWNSTREAM LAMBDA EXPRESSION WARNING: const does not appear in translations: ' + k + '\n' )

