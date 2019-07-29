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
    if expr not in bound: print( 'ERROR: unbound var: ' + expr )
  elif expr[0] == 'lambda':
    for subexpr in expr[2:]:
      findUnboundVars( subexpr, bound + [ expr[1] ] )
  else:
    for subexpr in expr[1:]:
      findUnboundVars( subexpr, bound               )

################################################################################

## For each discourse graph...
for line in sys.stdin:

  line = line.rstrip()
  if VERBOSE: print( 'GRAPH: ' + line )

  ## Initialize associations...
  PorQs  = collections.defaultdict( list )                                     ## Key is elem pred.
  Scopes = { }                                                                 ## Key is outscoped.
  Inhs   = collections.defaultdict( lambda : collections.defaultdict(float) )  ## Key is inheritor.
  Nuscos = { }
 
  ## For each assoc...
  for assoc in sorted( line.split(' ') ):
    src,lbl,dst = assoc.split(',')
    if lbl.isdigit():  PorQs  [src].insert( int(lbl), dst )   ## Add preds and quants.
    elif lbl == 's':   Scopes [src]      = dst                ## Add scopes.
    else:              Inhs   [src][lbl] = dst                ## Add inheritances.
    if lbl == 'r':     Nuscos [dst]      = src                ## Index nusco of each restr.

  Preds  = [ ]
  Quants = [ ] 

  ## Distinguish preds and quants...
  for elempred,Particips in PorQs.items():
    ## If three participants and last restriction-inherits from previous, it's a quant...
    if len( Particips ) == 3 and Inhs.get(Particips[2],{}).get('r','') == Particips[1]:  Quants.append( tuple( [ Particips[0] ] + [ elempred ] + Particips[1:] ) )
    else:                                                                                Preds.append ( tuple( [ Particips[0] ] + [ elempred ] + Particips[1:] ) )

  ## helper function to get most outscoping referent in scope chain, sharing vars Nuscos,Scopes
  def ceiling( x ):
    if x in Nuscos and x not in Nuscos.values(): return ceiling( Nuscos[x] )
    if x in Scopes: return ceiling( Scopes[x] )
    return x

  '''
  ## Copy outgoing Scopes up to 'e' inheritances...
  for inheritor in Scopes.keys():
    inherited = Inhs.get(inheritor,{}).get('e','')
    if inherited == '': inherited = Inhs.get( Inhs.get(inheritor,{}).get('r',''), {} ).get('e','')
    if inherited != '':
      Scopes[inherited] = Scopes[inheritor]
      if VERBOSE: print( 'X0: copying scope up extraction-inheritance ' + inherited + ' to ' + Scopes[inherited] )
  '''

  ## Induce scopes upward to pred args...
  for Args in Preds:
    for a in Args[2:]:
      if a not in Scopes.values() and a not in Scopes and Nuscos.get(a,'') not in Scopes and ( a in PorQs or Inhs.get(a,{}).get('r','') in PorQs ):
        if VERBOSE: print( 'X1: inducing scope ' + ceiling( Args[1] ) + ' to ' + a )
        Scopes[ ceiling( Args[1] ) ] = a
  ## Induce scopes upward to non-pred args...
  for Args in Preds:
    for a in Args[2:]:
      if a not in Scopes.values() and a not in Scopes and Nuscos.get(a,'') not in Scopes:
        if VERBOSE: print( 'X2: inducing scope ' + ceiling( Args[1] ) + ' to ' + a )
        Scopes[ ceiling( Args[1] ) ] = a
  ## Induce scopes upward to anything else not in chain...
  for Args in Preds:
    for a in Args[2:]:
      if a not in Scopes.values() and ceiling( a ) != ceiling( Args[1] ):
        if VERBOSE: print( 'X3: inducing scope ' + ceiling( Args[1] ) + ' to ' + a )
        Scopes[ ceiling( Args[1] ) ] = a
  ## Induce scopes upward to anything else not in chain...
  for Args in Preds:
    for a in Args[2:]:
      if ceiling( a ) != ceiling( Args[1] ):
        if VERBOSE: print( 'X4: pred ' + Args[1] + ' inducing scope ' + ceiling( Args[1] ) + ' to ' + a )
        Scopes[ ceiling( Args[1] ) ] = a

  ## Induce low existential quants when only scope annotated...
  for Args in Preds:
    for a in Args[1:]:
      nusco = a if a in Nuscos.values() else Nuscos.get(a,a)
      if nusco not in [s for q,e,r,s in Quants]:
        if Inhs[nusco].get('r','') == '': Inhs[nusco]['r'] = nusco+'r'
        Quants.append( ( 'D:some', nusco+'P', Inhs[nusco]['r'], nusco ) )

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
      print( 'I = ' + str(sorted(Inhs.items())) )
      print( 'T = ' + str(sorted(Translations)) )
      print( 'A = ' + str(sorted(Abstractions.items())) )
      print( 'E = ' + str(sorted(Expressions.items())) )

    ## P rule...
    for Participants in list(Preds):
      for particip in Participants[1:]:
        if particip not in Scopes.values() and particip not in Inhs:
          if VERBOSE: print( 'applying P to make \\' + particip + '. ' + str(Participants) )
          Abstractions[ particip ].append( Participants )
          if Participants in Preds: Preds.remove( Participants )
          active = True

    ## C rule...
    for var,Structs in Abstractions.items():
      if len(Structs) > 1:
        if VERBOSE: print( 'applying C to make \\' + var + '. ' + str( tuple( ['and'] + Structs ) ) )
        Abstractions[var] = [ tuple( ['and'] + Structs ) ]
        active = True

    ## M rule...
    for var,Structs in Abstractions.items():
      if len(Structs) == 1 and var not in Scopes.values() and var not in Inhs:
        if VERBOSE: print( 'applying M to make \\' + var + '. ' + str(Structs) )
        Expressions[var] = Structs[0]
        del Abstractions[var]
        active = True
 
    ## Q rule...
    for q,e,r,s in list(Quants):
      if r in Expressions and s in Expressions:
        if VERBOSE: print( 'applying Q to make (' + q + ' (\\' + r + '. ' + str(Expressions[r]) + ') (\\' + s + '. ' + str(Expressions[s]) + '))' )
        Translations.append( ( q, ( 'lambda', r, Expressions[r] ), ( 'lambda', s, Expressions[s] ) ) )
        Quants.remove( (q, e, r, s) )
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
          if src in Scopes and dst in Scopes:
            Abstractions[ src ].append( replaceVarName( replaceVarName( Expressions[dst], dst, src ), Scopes[dst], Scopes[src] ) )    ## I4 rule.
            if VERBOSE: print( 'applying I4 to replace ' + dst + ' with ' + src + ' and ' + Scopes[dst] + ' with ' + Scopes[src] + ' to make ' + str(Abstractions[src][-1]) )
          else:
            if VERBOSE: print( 'applying I2/I3 to replace ' + dst + ' with ' + src + ' to make ' + str(replaceVarName( Expressions[dst], dst, src )) )   #' in ' + str(Expressions[dst]) )
            Abstractions[ src ].append( replaceVarName( Expressions[dst], dst, src ) )
            if dst in Scopes and src in [s for q,e,r,s in Quants] + [r for q,e,r,s in Quants]:  Scopes[src if src in Nuscos.values() else Nuscos[src]] = Scopes[dst]     ## I3 rule.
          del Inhs[src][lbl]
          if len(Inhs[src])==0: del Inhs[src]
          active = True

    ## S1 rule...
    for q,R,S in list(Translations):
      if S[1] in Scopes:
        if VERBOSE: print( 'applying S1 to make (\\' + Scopes[ S[1] ] + ' ' + q + ' ' + str(R) + ' ' + str(S) + ')' )
        Abstractions[ Scopes[ S[1] ] ].append( (q, R, S) )
        del Scopes[ S[1] ]
        if R[1] in Scopes: del Scopes[ R[1] ]   ## Should use 't' trace assoc.
        Translations.remove( (q, R, S) )
        active = True

#print( Translations )
for expr in Translations:
  print( lambdaFormat(expr) )
  findUnboundVars( expr )



