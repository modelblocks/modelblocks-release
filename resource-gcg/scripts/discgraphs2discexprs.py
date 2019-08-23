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
    if expr not in bound and expr != '_': print( 'ERROR: unbound var: ' + expr )
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

## For each discourse graph...
for line in sys.stdin:

  #### I. READ IN AND PREPROCESS DISCOURSE GRAPH...

  line = line.rstrip()
  if VERBOSE: print( 'GRAPH: ' + line )

  ## Initialize associations...
  PorQs  = collections.defaultdict( list )                                     ## Key is elem pred.
  Scopes = { }                                                                 ## Key is outscoped.
  Traces = { }                                                                 ## Key is outscoped.
  Inhs   = collections.defaultdict( lambda : collections.defaultdict(float) )  ## Key is inheritor.
  Nuscos = collections.defaultdict( list )                                     ## Key is restrictor.
  NuscoValues = { }
  Inheriteds = { }
 
  ## For each assoc...
  for assoc in sorted( line.split(' ') ):
    src,lbl,dst = assoc.split(',')
    if lbl.isdigit():  PorQs  [src].insert( int(lbl), dst )   ## Add preds and quants.
    elif lbl == 's':   Scopes [src]      = dst                ## Add scopes.
    elif lbl == 't':   Traces [src]      = dst                ## Add traces.
    else:              Inhs   [src][lbl] = dst                ## Add inheritances.
    if lbl == 'r':     Nuscos [dst].append( src )             ## Index nusco of each restr.
    if lbl == 'r':     NuscoValues[src]  = True
    if lbl == 'e':     Inheriteds[dst]   = True

  Preds  = [ ]
  Quants = [ ] 

  ## Distinguish preds and quants...
  for elempred,Particips in PorQs.items():
    ## If three participants and last restriction-inherits from previous, it's a quant...
#    if len( Particips ) == 3 and Inhs.get(Particips[2],{}).get('r','') == Particips[1]:  Quants.append( tuple( [ Particips[0] ] + [ elempred ] + Particips[1:] ) )
    if Particips[0].endswith('Q'):  Quants.append( tuple( [ Particips[0] ] + [ elempred ] + Particips[1:] + (['_'] if len(Particips)<4 else []) ) )
    else:                           Preds.append ( tuple( [ Particips[0] ] + [ elempred ] + Particips[1:] ) )

  OrigConsts = [ ep[0] for ep in Preds ] + [ q[0] for q in Quants ]

  ## Report items...
  if VERBOSE: 
    print( 'P = ' + str(sorted(Preds)) )
    print( 'Q = ' + str(sorted(Quants)) )
    print( 'S = ' + str(sorted(Scopes.items())) )


  ## Construct list of inheriting refstates...
  Subs = collections.defaultdict( list )
  for xLo,lxHi in Inhs.items():
    for l,xHi in lxHi.items():
      if l!='w':
        Subs[ xHi ].append( xLo )
#  print( 'Subs = ' + str(Subs) )


  ## Check that no reft has multiple outscopers...
  def getBossesFromSup( xLo ):
#    print( 'now getting sup ' + xLo )
    if xLo in Scopes: return getBossesInChain( Scopes[xLo] )
    return sets.Set( [ y  for l,xHi in Inhs.get(xLo,{}).items() if l!='w'  for y in getBossesFromSup(xHi) ] )
  def getBossesFromSub( xHi ):
#    print( 'now getting sub ' + xHi )
    if xHi in Scopes: return getBossesInChain( Scopes[xHi] )
    return sets.Set( [ y  for xLo in Subs.get(xHi,[])  for y in getBossesFromSub(xLo) ] )
  def getBossesInChain( x ):
    out = getBossesFromSup(x) | getBossesFromSub(x)
    return out if len(out)>0 else sets.Set( [x] )
  for pred in Preds:
    for x in pred[1:]:
      if len( getBossesInChain(x) ) > 1: sys.stderr.write( 'WARNING: ' + x + ' has multiple outscopings in inheritance chain: ' + str( getBossesInChain(x) ) + '\n' )
      print( 'Bosses of ' + x + ': ' + str(getBossesInChain(x)) )


  #### II. ENFORCE NORMAL FORM (QUANTS AND SCOPE PARENTS AT MOST SPECIFIC INHERITANCES...

  ## Smite redundant nuscos of predicative noun phrases out of Subs...
  for xHi,l in Subs.items():
    for xLo in l:
      if 'r' in Inhs.get(Inhs.get(xLo,[]).get('r',''),[]):
        print( 'Smiting ' + xLo + ' out of Subs, for being redundant.' )
        Subs[xHi].remove(xLo)
        if len(Subs[xHi])==0: del Subs[xHi]
  ## Propagate scopes down inheritance chains...
  active = True
  while active:
    active = False
#    for xLo,lxHi in Inhs.items():
#      for l,xHi in lxHi.items():
#        if xHi in Scopes and xLo not in Scopes and l!='w':
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
    for q,e,r,xHi,n in Quants[:]:
      for xLo in Subs.get(xHi,[]):
        if xLo not in [s for _,_,_,s,_ in Quants]:
          if VERBOSE: print( 'Inheriting quant ' + q + ' from ' + xHi + ' to ' + xLo + '.' )
          Quants.append( (q,e,r,xLo,n) )
          active = True
  ## Clean up abstract scopes...
#  for x,ly in Inhs.items():
#    for l,y in ly.items():
#      if l=='r' and 'r' in Inhs.get(y,{}): continue  ## don't delete scope with redundant predicative inheritor
#      if y in Scopes and l in 'abcdefghijklmnopqruvxyz' and y not in [s for q,e,r,s,n in Quants]:
  for xHi in Scopes.keys():
    if xHi in Subs: #for xLo in Subs.get(xHi,[]):
      if VERBOSE: print( 'Removing redundant abstract scope parent ' + Scopes[xHi] + ' from ' + xHi + ' because of inheritance at ' + str(Subs[xHi]) )
      del Scopes[xHi]
  ## Clean up abstract quants...
  for q,e,r,s,n in Quants[:]:
    if s in Subs:
      if VERBOSE: print( 'Removing redundant abstract quant ' + q + ' from ' + s + ' because of inheritance at ' + Subs[s][0] )
      Quants.remove( (q,e,r,s,n) )

  ## Report items...
  if VERBOSE: 
    print( 'P = ' + str(sorted(Preds)) )
    print( 'Q = ' + str(sorted(Quants)) )
    print( 'S = ' + str(sorted(Scopes.items())) )


  #### III. INDUCE UNANNOTATED SCOPES AND EXISTENTIAL QUANTS...

  ## Scope ceiling...
#  def getCeilingFromSup( xLo ):
#    return getCeilingInChain( Scopes[xLo] ) if xLo in Scopes else sets.Set( [ y  for l,xHi in Inhs.get(xLo,{}).items() if l!='w'  for y in getCeilingFromSup(xHi) ] )
#  def getCeilingFromSub( xHi ):
#    return getCeilingInChain( Scopes[xHi] ) if xHi in Scopes else sets.Set( [ y  for xLo in Subs.get(xHi,[])  for y in getCeilingFromSub(xLo) ] )
#  def getCeilingInChain( x ):
#    out = getCeilingFromSup( x ) | getCeilingFromSub( x )
#    return out if len(out)>0 else sets.Set( [x] )
  def ceiling( x ):
    y = sorted( getBossesInChain(x) )[0]
    return y if y in NuscoValues or y not in Nuscos else Nuscos[y][0]

  ## List of referents that are or participate in elementary predications...
  Referents = sets.Set([ x for pred in Preds for x in pred[1:] ])

  ## List of referents that participate in elementary predications (which does not include the elementary predication itself)...
  Participants = sets.Set([ x for pred in Preds for x in pred[2:] ])

  ## Deterministically add scopes...
  active = True
  while active:
    ## Calculate ceilings of scoped refts...
    AnnotatedCeilings = sets.Set([ ceiling(x) for x in Scopes.keys() ])
    ## List of original (dominant) refts...
    HighAnnotated = sets.Set([ x for x in Referents if ceiling(x) in AnnotatedCeilings ])  # | sets.Set([ ceiling(x) for x in Scopes.values() ])
    print( 'HighAnnotated = ' + str(HighAnnotated) )

    active = False
    for pred in Preds:
      if pred[1] not in Participants:
        for xAnn in pred[2:]:
          if xAnn in HighAnnotated:
            if len(pred) == 3 and pred[1] not in Scopes:
              print( 'Unconstrained elementary predicate ' + pred[0] + ' ' + pred[1] + ' deterministically binding self to annotated participant ' + xAnn )
              Scopes[ pred[1] ] = xAnn
              active = True
            if len(pred) == 4:
              for xNotAnn in pred[2:]:
                if xAnn != xNotAnn and xNotAnn not in HighAnnotated and pred[1] not in Scopes:
                  print( 'Unconstrained elementary predicate ' + pred[0] + ' ' + pred[1] + ' deterministically binding self to non-annotated participant ' + xNotAnn + ' given other participant ' + xAnn + ' is annotated.' )
                  Scopes[ pred[1] ] = xNotAnn
                  active = True

#    if len(pred) == 3:
#      print( 'Deterministically scoping unary elementary predicate ' + pred[0] + ' ' + pred[1] + ' to participant ' + pred[2] )
#      Scopes[ pred[1] ] = pred[2]
#    elif pred[2] in PorQs:
#      print( 'Deterministically scoping elementary predicate ' + pred[0] + ' ' + pred[1] + ' to elementary predicate participant ' + pred[2] )
#      Scopes[ pred[1] ] = pred[2]
#    elif pred[3] in PorQs:
#      print( 'Deterministically scoping elementary predicate ' + pred[0] + ' ' + pred[1] + ' to elementary predicate participant ' + pred[3] )
#      Scopes[ pred[1] ] = pred[3]

  ## Recursive function to search space of scopings...
  def tryScope( HypScopes, nest=1 ):

    if VERBOSE: print( '  '*nest + str(sorted(HypScopes.items())) )

#    if nest > 8: exit(0)

    ## Helper function to determine if one ref state outscopes another
    def reachesFromSup( xLo, xHi ):
      #print( '  '*nest + 'reachesFromSup ' + xLo + ' ' + xHi )
      return True if xLo == xHi else reachesInChain( HypScopes[xLo], xHi ) if xLo in HypScopes else any( [ reachesFromSup(xSup,xHi) for l,xSup in Inhs.get(xLo,{}).items() if l!='w' ] )
    def reachesFromSub( xLo, xHi ):
      #print( '  '*nest + 'reachesFromSub ' + xLo + ' ' + xHi )
      return True if xLo == xHi else reachesInChain( HypScopes[xLo], xHi ) if xLo in HypScopes else any( [ reachesFromSub(xSub,xHi) for xSub in Subs.get(xLo,[]) ] )
    def reachesInChain( xLo, xHi ):
      #print( '  '*nest + 'reachesInChain ' + xLo + ' ' + xHi )
      return reachesFromSup( xLo, xHi ) or reachesFromSub( xLo, xHi )

    '''
    ## Helper functions to explore inheritance chain...
    def outscopingFromSup( xLo ):
      return True if xLo in HypScopes.values() else any( [ outscopingFromSup(xHi) for l,xHi in Inhs.get(xLo,{}).items() if l!='w' ] )
    def outscopingFromSub( xHi ):
      return True if xHi in HypScopes.values() else any( [ outscopingFromSub(xLo) for xLo in Subs.get(xHi,[]) ] )
    def outscopingInChain( x ):
      return outscopingFromSup( x ) or outscopingFromSub( x )
    '''

    ## Scope Ceiling...
    def getCeilingFromSup( xLo ):
      return getCeilingInChain( HypScopes[xLo] ) if xLo in HypScopes else sets.Set( [ y  for l,xHi in Inhs.get(xLo,{}).items() if l!='w'  for y in getCeilingFromSup(xHi) ] )
    def getCeilingFromSub( xHi ):
      return getCeilingInChain( HypScopes[xHi] ) if xHi in HypScopes else sets.Set( [ y  for xLo in Subs.get(xHi,[])  for y in getCeilingFromSub(xLo) ] )
    def getCeilingInChain( x ):
      out = getCeilingFromSup( x ) | getCeilingFromSub( x )
      return out if len(out)>0 else sets.Set( [x] )
    def ceiling( x ):
      y = sorted( getCeilingInChain(x) )[0]
      return y if y in NuscoValues or y not in Nuscos else Nuscos[y][0]

    unsatisfied = False
    for pred in Preds:  ## Note: for speed, try unary first
#      if len(pred) > 3:
        for xHi in pred[2:]:
          if reachesInChain( pred[1], xHi ):
            if VERBOSE: print( '  '*nest + str(nest) + ': ' + pred[0] + ' ' + pred[1] + ' bound by ' + xHi + '.' )
          else:
            unsatisfied = True
            if xHi in HighAnnotated and any([not reachesInChain(pred[1],x) and x not in HighAnnotated for x in pred[2:] if x!=xHi]):
              if VERBOSE: print( '  '*nest + str(nest) + ': ' + pred[0] + ' ' + pred[1] + ' dispreferred to annotated ' + xHi + ' when other participants not bound: ' + ','.join([x for x in pred[2:] if x!=xHi and not reachesInChain(pred[1],x) and x not in HighAnnotated ]) + '.' )
            else:
              if ceiling(pred[1]) == ceiling(xHi):
                print( '  '*nest + str(nest) + ': dead end -- ' + pred[1] + ' coapical with but not bound by ' + xHi + '.' )
                return None
              else:
                if VERBOSE: print( '  '*nest + str(nest) + ': ' + pred[0] + ' ' + pred[1] + ' not bound by ' + xHi + ', try ceiling ' + pred[1] + ' = ' + ceiling(pred[1]) + ' to ' + xHi + '...' )
                AppendedHypScopes = HypScopes.copy()
                AppendedHypScopes[ ceiling(pred[1]) ] = xHi
                OutputScopes = tryScope( AppendedHypScopes, nest+1 )
                if OutputScopes != None: return OutputScopes
    if not unsatisfied:
      if VERBOSE: print( 'Found scoping:' )
      if VERBOSE: print( HypScopes )
      return HypScopes
    return None

  if VERBOSE: print( 'running tryScope...' )
  Scopes = tryScope( Scopes )
  if VERBOSE: print( Scopes )

  ## Induce low existential quants when only scope annotated...
#  for xCh in sorted([x if x in NuscoValues else Nuscos[x] for x in Scopes.keys()] + [x for x in Scopes.values() if x in NuscoValues]):  #sorted([ s for s in NuscoValues if 'r' not in Inhs.get(Inhs.get(s,{}).get('r',''),{}) ]): #Scopes:
#  ScopeyNuscos = [ x for x in NuscoValues if 'r' not in Inhs.get(Inhs.get(x,{}).get('r',''),{}) and (x in Scopes.keys()+Scopes.values() or Inhs.get(x,{}).get('r','') in Scopes.keys()+Scopes.values()) ]
  ScopeyNuscos = [ x for x in Referents | sets.Set(Inhs.keys()) if (x not in Nuscos or x in NuscoValues) and 'r' not in Inhs.get(Inhs.get(x,{}).get('r',''),{}) and (x in Scopes.keys()+Scopes.values() or Inhs.get(x,{}).get('r','') in Scopes.keys()+Scopes.values()) ]
  print( 'ScopeyNuscos = ' + str(ScopeyNuscos) )
  print( 'Referents = ' + str(Referents) )
  print( 'Nuscos = ' + str(Nuscos) )
  for xCh in ScopeyNuscos:
    if xCh not in [s for _,_,_,s,_ in Quants]: # + [r for q,e,r,s,n in Quants]:
      if Inhs[xCh].get('r','') == '': Inhs[xCh]['r'] = xCh+'r'
      if VERBOSE: print( 'Inducing existential quantifier: ' + str([ 'D:someQ', xCh+'P', Inhs[xCh]['r'], xCh, '_' ]) )
      Quants.append( ( 'D:someQ', xCh+'P', Inhs[xCh]['r'], xCh, '_' ) )


  #### IV. ENFORCE NORMAL FORM (QUANTS AND SCOPE PARENTS AT MOST SPECIFIC INHERITANCES...

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
    for q,e,r,xHi,n in Quants[:]:
      for xLo in Subs.get(xHi,[]):
        if xLo not in [s for _,_,_,s,_ in Quants]:
          if VERBOSE: print( 'Inheriting quant ' + q + ' from ' + xHi + ' to ' + xLo + '.' )
          Quants.append( (q,e,r,xLo,n) )
          active = True
  ## Clean up abstract scopes...
  for xHi in Scopes.keys():
    if xHi in Subs: #for xLo in Subs.get(xHi,[]):
      if VERBOSE: print( 'Removing redundant abstract scope parent ' + Scopes[xHi] + ' from ' + xHi + ' because of inheritance at ' + str(Subs[xHi]) )
      del Scopes[xHi]
  ## Clean up abstract quants...
  print( Subs )
  for q,e,r,s,n in Quants[:]:
    if s in Subs:
      if VERBOSE: print( 'Removing redundant abstract quant ' + q + ' from ' + s + ' because of inheritance at ' + Subs[s][0] )
      Quants.remove( (q,e,r,s,n) )
  print( 'Quants = ' + str(Quants) )


  #### V. TRANSLATE TO LAMBDA CALCULUS...

  Translations = [ ]
  Abstractions = collections.defaultdict( list )  ## Key is lambda.
  Expressions  = collections.defaultdict( list )  ## Key is lambda.

  ## Iterations...
  i = 0
  active = True
  while active: #Preds != [] or Quants != []:
    i += 1
    active = False

    if VERBOSE: 
      print( '---- ITERATION ' + str(i) + ' ----' )
      print( 'P = ' + str(sorted(Preds)) )
      print( 'Q = ' + str(sorted(Quants)) )
      print( 'S = ' + str(sorted(Scopes.items())) )
      print( 't = ' + str(sorted(Traces.items())) )
      print( 'I = ' + str(sorted(Inhs.items())) )
      print( 'T =  ' + str(sorted(Translations)) )
      print( 'A = ' + str(sorted(Abstractions.items())) )
      print( 'E = ' + str(sorted(Expressions.items())) )

    ## P rule...
    for Participants in list(Preds):
      for particip in Participants[1:]:
        if particip not in Scopes.values() and particip not in Inhs:
          if VERBOSE: print( 'applying P to make \\' + particip + '. ' + lambdaFormat(Participants) )
          Abstractions[ particip ].append( Participants )
          if Participants in Preds: Preds.remove( Participants )
          active = True

    ## C rule...
    for var,Structs in Abstractions.items():
      if len(Structs) > 1:
        if VERBOSE: print( 'applying C to make \\' + var + '. ' + lambdaFormat( tuple( ['and'] + Structs ) ) )
        Abstractions[var] = [ tuple( ['and'] + Structs ) ]
        active = True

    ## M rule...
    for var,Structs in Abstractions.items():
      if len(Structs) == 1 and var not in Scopes.values() and var not in Inhs:
        if VERBOSE: print( 'applying M to make \\' + var + '. ' + lambdaFormat(Structs[0]) )
        Expressions[var] = Structs[0]
        del Abstractions[var]
        active = True
 
    ## Q rule...
    for q,e,r,s,n in list(Quants):
      if r in Expressions and s in Expressions:
        if VERBOSE: print( 'applying Q to make ' + lambdaFormat( ( q, n, ( 'lambda', r, Expressions[r] ), ( 'lambda', s, Expressions[s] ) ) ) )   ## (' + q + ' (\\' + r + '. ' + str(Expressions[r]) + ') (\\' + s + '. ' + str(Expressions[s]) + '))' )
        Translations.append( ( q, n, ( 'lambda', r, Expressions[r] ), ( 'lambda', s, Expressions[s] ) ) )
        Quants.remove( (q, e, r, s, n) )
        active = True

    ## I1 rule...
    for src,lbldst in Inhs.items():
      for lbl,dst in lbldst.items():
        if dst not in Inhs and dst not in Abstractions and dst not in Expressions and dst not in Scopes.values() and dst not in [ x for Particips in Preds for x in Particips ]:
          if VERBOSE: print( 'applying I1 to make \\' + dst + ' True' )
          Abstractions[ dst ].append( () )
          active = True

    ## I2,I3,I4 rule...
    for src,lbldst in Inhs.items():
      for lbl,dst in lbldst.items():
        if dst in Expressions:
          if src in Scopes and dst in Traces and Scopes[src] in Scopes and Traces[dst] in Traces:
            Abstractions[ src ].append( replaceVarName( replaceVarName( replaceVarName( Expressions[dst], dst, src ), Traces[dst], Scopes[src] ), Traces[Traces[dst]], Scopes[Scopes[src]] ) )    ## I4 rule.
            if VERBOSE: print( 'applying I4 to replace ' + dst + ' with ' + src + ' and ' + Traces[dst] + ' with ' + Scopes[src] + ' and ' + Traces[Traces[dst]] + ' with ' + Scopes[Scopes[src]] + ' to make \\' + src + ' ' + lambdaFormat(Abstractions[src][-1]) )
#            del Traces[dst]
          elif src in Scopes and dst in Traces:
            Abstractions[ src ].append( replaceVarName( replaceVarName( Expressions[dst], dst, src ), Traces[dst], Scopes[src] ) )    ## I4 rule.
            if VERBOSE: print( 'applying I4 to replace ' + dst + ' with ' + src + ' and ' + Traces[dst] + ' with ' + Scopes[src] + ' to make \\' + src + ' ' + lambdaFormat(Abstractions[src][-1]) )
#            del Traces[dst]
          else:
            if VERBOSE: print( 'applying I2/I3 to replace ' + dst + ' with ' + src + ' to make \\' + src + ' ' + lambdaFormat(replaceVarName( Expressions[dst], dst, src )) )   #' in ' + str(Expressions[dst]) )
            Abstractions[ src ].append( replaceVarName( Expressions[dst], dst, src ) )
            if dst in Scopes and src in [s for q,e,r,s,n in Quants] + [r for q,e,r,s,n in Quants]:  Scopes[src if src in NuscoValues else Nuscos[src][0]] = Scopes[dst]     ## I3 rule.
          del Inhs[src][lbl]
          if len(Inhs[src])==0: del Inhs[src]
          active = True

    ## S1 rule...
    for q,n,R,S in list(Translations):
      if S[1] in Scopes:
        if VERBOSE: print( 'applying S1 to make (\\' + Scopes[ S[1] ] + ' ' + q + ' ' + str(R) + ' ' + str(S) + ')' )
        Abstractions[ Scopes[ S[1] ] ].append( (q, n, R, S) )
        del Scopes[ S[1] ]
#        if R[1] in Scopes: del Scopes[ R[1] ]   ## Should use 't' trace assoc.
        print( 'size of Translations before = ' + str(len(Translations)) )
        Translations.remove( (q, n, R, S) )
        print( 'size of Translations after = ' + str(len(Translations)) )
        active = True

#print( Translations )
for expr in Translations:
  print( lambdaFormat(expr) )
  findUnboundVars( expr )
  checkConstsUsed( expr, OrigConsts )
for k in OrigConsts:
  print( 'WARNING: const does not appear in translations: ' + k )

