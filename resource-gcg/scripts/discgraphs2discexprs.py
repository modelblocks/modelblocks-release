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
      findUnboundVars( subexpr, bound               )

################################################################################

## For each discourse graph...
for line in sys.stdin:

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

  '''
  ## helper function to get most outscoping referent in scope chain, sharing vars Nuscos,Scopes
  def ceiling( x, recursed = False ):
    ## Choose scope if there is one...
    if x in Scopes: return ceiling( Scopes[x] )
    ## Try nuclear scope if in restrictor...
    if not recursed and x in Nuscos and x not in NuscoValues:
      return ceiling( Nuscos[x][0] )
#      for y in Nuscos[x]:
#        best = ceiling( y )
#        if best != y: return best
    ## Try each inheritance...
    for l in Inhs.get(x,{}):
      best = ceiling( Inhs[x][l], True )
      if best != Inhs[x][l]: return best
#    if x in Inhs and 'c' in Inhs[x]: return ceiling( Inhs[x]['c'] )
#    if x in Inhs and 'r' in Inhs[x] and Inhs[x]['r'] in Inhs and 'n' in Inhs[Inhs[x]['r']]: return ceiling( Inhs[ Inhs[x]['r'] ]['n'] )
#    if x in Inhs and 'n' in Inhs[x]: return ceiling( Inhs[x]['n'] )
    return x
  '''

  ## Construct list of inheriting refstates...
  Subs = collections.defaultdict( list )
  for xLo,lxHi in Inhs.items():
    for l,xHi in lxHi.items():
      if l!='w':
        Subs[ xHi ].append( xLo )
  print( 'Subs = ' + str(Subs) )
  OrigQuant = { }
  for q,e,r,s,n in Quants:
    OrigQuant[r] = True
    OrigQuant[s] = True
  print( 'OrigQuant = ' + str(OrigQuant) )
  ## Return true if superstate outscoped...
  def outscopedAbove( xLo ):
    if xLo in Scopes: return True
    return any( [ outscopedAbove(xHi) for l,xHi in Inhs.get(xLo,{}).items() if l!='w'] )
  ## Return true if substate outscoped...
  def outscopedBelow( xHi ):
    if xHi in Scopes: return True
    return any( [ outscopedBelow(xLo) for xLo in Subs.get(xHi,[]) ] )
  ## Return true if sub/superstate outscoped...
  def outscopedInChain( x ):
    return outscopedAbove(x) or outscopedBelow(x)
  ## Return true if superstate outscoping...
  def outscopingAbove( xLo ):
    if xLo in Scopes.values(): return True
    return any( [ outscopingAbove(xHi) for l,xHi in Inhs.get(xLo,{}).items() if l!='w' ] )
  ## Return true if substate outscoping...
  def outscopingBelow( xHi ):
    if xHi in Scopes.values(): return True
    return any( [ outscopingBelow(xLo) for xLo in Subs.get(xHi,[]) ] )
  ## return true if sub/superstate outscoping...
  def outscopingInChain( x ):
    return outscopingAbove(x) or outscopingBelow(x)
  ## Return true if superstate has orig quant...
  def hasOrigQuantAbove( xLo ):
    if xLo in OrigQuant: return True
    return any( [ hasOrigQuantAbove(xHi) for l,xHi in Inhs.get(xLo,{}).items() if l!='w' ] )
  ## Return true if substate has orig quant...
  def hasOrigQuantBelow( xHi ):
    if xHi in OrigQuant: return True
    return any( [ hasOrigQuantBelow(xLo) for xLo in Subs.get(xHi,[]) ] )
  ## Return true if sub/superstate has orig quant...
  def hasOrigQuantInChain( x ):
    return hasOrigQuantAbove(x) or hasOrigQuantBelow(x)

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
      if len( getBossesInChain(x) ) > 1: sys.stderr.write( 'WARNING: multiple outscopings in inheritance chain: ' + str( getBossesInChain(x) ) + '\n' )
      print( x, getBossesInChain(x) )
  def ceiling( x ):
    y = sorted( getBossesInChain(x) )[0]
    return y if y in NuscoValues or y not in Nuscos else Nuscos[y][0]

  '''
  Sepocs = dict( (v,k) for k,v in Scopes.items() )
  def getMinionsFromSup( xLo ):
    if xLo in Sepocs: return getMinionsInChain( Sepocs[xLo] )
    return sets.Set( [ y for l,xHi in Inhs.get(xLo,{}).items() if l!='w' for y in getMinionsFromSup(xHi) ] )
  def getMinionsFromSub( xHi ):
    if xHi in Sepocs: return getMinionsInChain( Sepocs[xHi] )
    return sets.Set( [ y for xLo in Subs.get(xHi,[]) for y in getMinionsFromSub(xLo) ] )
  def getMinionsInChain( x ):
    out = getMinionsFromSup(x) | getMinionsFromSub(x)
    return out if len(out)>0 else sets.Set( [x] )
  def floor( x ):
    return sorted( getMinionsInChain(x) )[0]
  '''

  ## Find depth of annotated scope associaitons...
  Depths = { }
  def getDepthFromAbove( xLo ):
    if xLo in Scopes: return getDepthInChain( Scopes[xLo] ) + 1
    return max( [ 0 ] + [ getDepthFromAbove(xHi) for l,xHi in Inhs.get(xLo,{}).items() if l!='w' ] )
  def getDepthFromBelow( xHi ):
    if xHi in Scopes: return getDepthInChain( Scopes[xHi] ) + 1
    return max( [ 0 ] + [ getDepthFromBelow(xLo) for xLo in Subs.get(xHi,[]) ] )
  def getDepthInChain( x ):
    return max( [ getDepthFromAbove(x), getDepthFromBelow(x) ] )
  '''
  active = True
  while active:
    active = False
    for pred in Preds:
      for x in pred[1:]:
        d = getDepthInChain( x )
        if d > Depths.get(x,0):
          Depths[x] = d
          active = True
  '''
  for pred in Preds:
    for x in pred[1:]:
      Depths[x] = getDepthInChain( x )
  print( 'Depths = ' + str( Depths ) )


  active = True
  while active:
    active = False

    ## Attach predicates to predicate arguments...
    for pred in Preds:
        xLo = pred[1]
        if getDepthInChain( xLo ) == 0:
          l = [ xHi for xHi in pred[2:] if xHi in PorQs and ceiling(xLo) != ceiling(xHi) ]
#          print( pred[1], xLo, l )
          if len( l ) == 1:
            if VERBOSE: print( 'X1: pred ' + pred[1] + ' adding scope from ceiling of ' + xLo + ' = ' + ceiling(xLo) + ' to ' + l[0] ) #floor( max(l)[1] ) )
            Scopes[ ceiling(xLo) ] = l[0] #floor( max(l)[1] )
            active = True

    ## Attach predicates to non-predicate arguments...
    for pred in Preds:
        xLo = pred[1]
        if getDepthInChain( xLo ) == 0:
          l = [ xHi for xHi in pred[2:] if xHi.endswith('\'') and getDepthInChain(xHi)==0 and not hasOrigQuantInChain(xHi) and ceiling(xLo) != ceiling(xHi) ]
#          print( pred[1], xLo, l )
          if len( l ) == 1:
            if VERBOSE: print( 'X2: pred ' + pred[1] + ' adding scope from ceiling of ' + xLo + ' = ' + ceiling(xLo) + ' to ' + l[0] ) #floor( max(l)[1] ) )
            Scopes[ ceiling(xLo) ] = l[0] #floor( max(l)[1] )
            active = True

    ## Attach arguments to arguments...
#    for pred in Preds:
#      for xLo in pred[2:]:
#        for xHi in pred[2:]:

    l = sorted( [ (getDepthInChain(xHi),xLo not in PorQs,pred[1],xLo,xHi)  for pred in Preds  for xLo in pred[1:]  for xHi in pred[2:]  if xLo != xHi ],  reverse=True )   ## and xLo not in PorQs
    print(l)
    for _,_,pd,xLo,xHi in l:
      if ceiling(xLo) != ceiling(xHi):
#        print( 'Scos = ' + str(Scopes) )
#        print( 'Subs = ' + str(Subs) )
#        print( 'Boss = ' + str(getBossesInChain(xLo)) )
        if VERBOSE: print( 'X3: pred ' + pd + ' adding scope from ceiling of ' + xLo + ' = ' + ceiling(xLo) + ' to ' + xHi )
        Scopes[ ceiling(xLo) ] = xHi
        active = True
        break

    '''
  active = True
  while active:
    active = False

    l = sorted( [ (getDepthInChain(xHi),pred[1],xLo,xHi)  for pred in Preds  for xLo in pred[1:]  for xHi in pred[2:]  if xLo != xHi ],  reverse=True )   ## and xLo in PorQs
    print(l)
    for _,pd,xLo,xHi in l:
      if ceiling(xLo) != ceiling(xHi):
#        print( 'Scos = ' + str(Scopes) )
#        print( 'Subs = ' + str(Subs) )
#        print( 'Boss = ' + str(getBossesInChain(xLo)) )
        if VERBOSE: print( 'X4: pred ' + pd + ' adding scope from ceiling of ' + xLo + ' = ' + ceiling(xLo) + ' to ' + xHi )
        Scopes[ ceiling(xLo) ] = xHi
        active = True
        break
    '''

    '''
    ## Induce scope from argument to argument...
    l = sorted( [ (getDepthInChain(xHi),pred[1],xLo,xHi)  for pred in Preds  for xLo in pred[1:]  for xHi in pred[2:]  if xLo != xHi ],  reverse=True )
    print(l)
    for d,_,xLo,xHi in l:
      if (d > 0 or ) and ceiling(xLo) != ceiling(xHi):
        if VERBOSE: print( 'X3: pred ' + pred[1] + ' adding scope from ' + ceiling(xLo) + ' to ' + xHi )
        Scopes[ ceiling(xLo) ] = xHi
        active = True


    ## Induce scope from argument to argument...
    for pred in Preds:
      for xLo in reversed( pred[2:] ):
        if getDepthInChain( xLo ) == 0:
          l = [ (getDepthInChain(xHi),xHi) for xHi in pred[2:] if ceiling(xHi) != ceiling(xLo) ]
          print( pred[1], xLo, l )
          if len( l ) > 0:
            if VERBOSE: print( 'X1: pred ' + pred[1] + ' adding scope from ' + ceiling(xLo) + ' to ' + max(l)[1] ) #floor( max(l)[1] ) )
#            a = ceiling(xLo)
#            b = floor( max(l)[1] )
#            Scopes[ a ] = b
#            Sepocs[ b ] = a
            Scopes[ ceiling(xLo) ] = max(l)[1] #floor( max(l)[1] )
#            Sepocs[ floor( max(l)[1] ) ] = ceiling(xLo)
            active = True
    ## Induce scope from predicate to argument...
    for pred in Preds:
#      for xLo in reversed( pred[1:] ):
          xLo = pred[1]
#        if getDepthInChain( xLo ) == 0:
          l = [ (getDepthInChain(xHi),xHi) for xHi in pred[2:] if ceiling(xHi) != ceiling(xLo) ]
          print( pred[1], xLo, l )
          if len( l ) > 0:
            if VERBOSE: print( 'X2: pred ' + pred[1] + ' adding scope from ' + ceiling(xLo) + ' to ' + max(l)[1] ) #floor( max(l)[1] ) )
#            a = ceiling(xLo)
#            b = floor( max(l)[1] )
#            Scopes[ a ] = b
#            Sepocs[ b ] = a
            Scopes[ ceiling(xLo) ] = max(l)[1] #floor( max(l)[1] )
#            Sepocs[ floor( max(l)[1] ) ] = ceiling(xLo)
            active = True
    '''

    '''
    ## Induce scope from argument to bot argument...
    for pred in Preds:
     if len(pred) > 3:
      for xHi in pred[2:]:
        if ( outscopedInChain( xHi ) or hasOrigQuantInChain( xHi ) ) and not outscopingInChain( xHi ):
#        if xHi in Scopes and xHi not in Scopes.values():
          for xLo in pred[2:]:
            if not outscopedInChain( xLo ) and ceiling( xLo ) != ceiling( xHi ):    ##  and not hasOrigQuantInChain( xLo )
#            if xLo not in Scopes and xHi not in Inheriteds and not any( [x in Scopes for x in Nuscos.get(xLo,[])] ) and ceiling( xLo ) != ceiling( xHi ):
              if VERBOSE: print( 'X1: pred ' + pred[1] + ' adding scope from ' + ceiling( xLo ) + ' to ' + xHi )
              Scopes[ ceiling( xLo ) ] = xHi
              active = True
    ## Induce scope from predicate to bot argument...
    if not active:
      for pred in Preds:
       if len(pred) > 3:
        for xHi in pred[2:]:
          if ( outscopedInChain( xHi ) or hasOrigQuantInChain( xHi ) ) and not outscopingInChain( xHi ):
#          if xHi in Scopes and xHi not in Scopes.values():
            xLo = pred[1]
            if not outscopedInChain( xLo ) and ceiling( xLo ) != ceiling( xHi ):
#            if pred[1] not in Scopes and not any( [x in Scopes for x in Nuscos.get(pred[1],[])] ) and ceiling( pred[1] ) != ceiling( xHi ):
              if VERBOSE: print( 'X2: pred ' + pred[1] + ' adding scope from ' + ceiling( pred[1] ) + ' to ' + xHi )
              Scopes[ ceiling( pred[1] ) ] = xHi
              active = True
    ## Induce scope from argument to argument...
    if not active:
      for pred in Preds:
        for xHi in pred[2:]:
         if ( outscopedInChain( xHi ) or hasOrigQuantInChain( xHi ) ):
          for xLo in pred[2:]:
            if not outscopedInChain( xLo ) and not hasOrigQuantInChain( xLo ) and ceiling( xLo ) != ceiling( xHi ):
#            if xLo not in Scopes and xHi not in Inheriteds and not any( [x in Scopes for x in Nuscos.get(xLo,[])] ) and ceiling( xLo ) != ceiling( xHi ):
              if VERBOSE: print( 'X3: pred ' + pred[1] + ' adding scope from ' + ceiling( xLo ) + ' to ' + xHi )
              Scopes[ ceiling( xLo ) ] = xHi
              active = True
    ## Induce scope from predicate to argument...
    if not active:
      for pred in Preds:
        for xHi in pred[2:]:
         if ( outscopedInChain( xHi ) or hasOrigQuantInChain( xHi ) ):
          xLo = pred[1]
          if not outscopedInChain( xLo ) and not hasOrigQuantInChain( xLo ) and ceiling( xLo ) != ceiling( xHi ):
#          if pred[1] not in Scopes and not any( [x in Scopes for x in Nuscos.get(pred[1],[])] ) and ceiling( pred[1] ) != ceiling( xHi ):
            if VERBOSE: print( 'X4: pred ' + pred[1] + ' adding scope from ' + ceiling( pred[1] ) + ' to ' + xHi )
            Scopes[ ceiling( pred[1] ) ] = xHi
            active = True
    '''

  ## Clean up abstract scopes...
  for x,ly in Inhs.items():
    for l,y in ly.items():
      if l=='r' and 'r' in Inhs.get(y,{}): continue  ## don't delete scope with redundant predicative inheritor
      if y in Scopes and l in 'abcdefghijklmnopqruvxyz' and y not in [s for q,e,r,s,n in Quants]:  del Scopes[y]


  ## Induce low existential quants when only scope annotated...
  for xCh,xPt in Scopes.items():
    if xCh not in [s for q,e,r,s,n in Quants]:
      if Inhs[xCh].get('r','') == '': Inhs[xCh]['r'] = xCh+'r'
      Quants.append( ( 'D:someQ', xCh+'P', Inhs[xCh]['r'], xCh, '_' ) )


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
      print( 'T = ' + str(sorted(Traces.items())) )
      print( 'I = ' + str(sorted(Inhs.items())) )
      print( 'T = ' + str(sorted(Translations)) )
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
            if VERBOSE: print( 'applying I4 to replace ' + dst + ' with ' + src + ' and ' + Traces[dst] + ' with ' + Scopes[src] + ' and ' + Traces[Traces[dst]] + ' with ' + Scopes[Scopes[src]] + ' to make ' + lambdaFormat(Abstractions[src][-1]) )
#            del Traces[dst]
          elif src in Scopes and dst in Traces:
            Abstractions[ src ].append( replaceVarName( replaceVarName( Expressions[dst], dst, src ), Traces[dst], Scopes[src] ) )    ## I4 rule.
            if VERBOSE: print( 'applying I4 to replace ' + dst + ' with ' + src + ' and ' + Traces[dst] + ' with ' + Scopes[src] + ' to make ' + lambdaFormat(Abstractions[src][-1]) )
#            del Traces[dst]
          else:
            if VERBOSE: print( 'applying I2/I3 to replace ' + dst + ' with ' + src + ' to make ' + lambdaFormat(replaceVarName( Expressions[dst], dst, src )) )   #' in ' + str(Expressions[dst]) )
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
        Translations.remove( (q, n, R, S) )
        active = True

#print( Translations )
for expr in Translations:
  print( lambdaFormat(expr) )
  findUnboundVars( expr )



