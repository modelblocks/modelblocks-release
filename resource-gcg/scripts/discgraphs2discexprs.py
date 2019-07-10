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

## For each discourse graph...
for line in sys.stdin:

  line = line.rstrip()
  if VERBOSE: print( 'GRAPH: ' + line )

  ## Initialize associations...
  PorQs  = collections.defaultdict( list )                                     ## Key is elem pred.
  Scopes = { }                                                                 ## Key is outscoped.
  Inhs   = collections.defaultdict( lambda : collections.defaultdict(float) )  ## Key is inheritor.
  
  ## For each assoc...
  for assoc in sorted( line.split(' ') ):
    src,lbl,dst = assoc.split(',')
    if lbl.isdigit():  PorQs  [src].insert( int(lbl), dst )   ## Add preds and quants.
    elif lbl == 's':   Scopes [src]      = dst                ## Add scopes.
    else:              Inhs   [src][lbl] = dst                ## Add inheritances.

  Preds  = [ ]
  Quants = [ ] 

  ## Distinguish preds and quants...
  for elempred,Particips in PorQs.items():
    ## If three participants and last restriction-inherits from previous, it's a quant...
    if len( Particips ) == 3 and Inhs.get(Particips[2],{}).get('r','') == Particips[1]:  Quants.append( tuple( [ Particips[0] ] + [ elempred ] + Particips[1:] ) )
    else:                                                                                Preds.append ( tuple( [ Particips[0] ] + [ elempred ] + Particips[1:] ) )

  ## Induce low existential quants for predicates occurring as arguments, to form chain...
  for Particips in Preds:
    for x in Particips[2:]:
      if x in PorQs and x not in Scopes:
        Outscopers = []
        for y in Particips[2:]:
          if y in Scopes:
            projection = y
            while projection in Scopes:
              projection = Scopes[projection]
              if projection in Particips: Outscopers += [projection]
        Scopes[x] = [y for y in Particips[2:] if y != x and y not in Outscopers][0]
  print( 'S = ' + str(sorted(Scopes.items())) )
  ## Induce low existential quants when only scope annotated...
  for nusco in Scopes:
    if nusco not in [s for q,e,r,s in Quants]:
      if Inhs[nusco].get('r','') == '': Inhs[nusco]['r'] = nusco+'r'
      Quants.append( ( 'D:some', nusco+'P', Inhs[nusco]['r'], nusco ) )
  ## Induce low existential quants for predicates (add to lowest arg)...
  for Particips in Preds:
    if len( Particips ) > 2:
      e = Particips[1]
      if e not in [s for q,_,r,s in Quants] and e not in [r for q,_,r,s in Quants]:
        lowest = Particips[2] if Particips[2] in Scopes else Particips[-1]
        for x in Particips[2:-1]:
          if x != lowest:
            outscoper = x
            while outscoper in Scopes:
              if outscoper == lowest: lowest = x
              outscoper = Scopes[outscoper]
        if Inhs[e].get('r','') == '': Inhs[e]['r'] = e+'r'
        Quants.append( ( 'D:some', e+'Q', Inhs[e]['r'], e ) )
        Scopes[e] = lowest
        #y = [s for s in Inhs if Inhs.get(s,{}).get('r','') == e][0]
#        e,y = (e,e[:-1]+'s') if e[-1]=='r' else (e+'r',e)
#        if Inhs[y].get('r','') == '': Inhs[y]['r'] = e
#        Quants.append( ( 'D:some', y+'P', e, y ) )
#        Scopes[y] = lowest
  ## Induce low existential quants for missing arguments...
  for Particips in Preds:
    lowest = Particips[1]
    for x in Particips[2:]:
      if x not in [Inhs[s]['r'] for s in Inhs if 'r' in Inhs[s]] and x not in [s for q,e,r,s in Quants] and x not in [r for q,e,r,s in Quants]:
        if Inhs[x].get('r','') == '': Inhs[x]['r'] = x+'r'
        Quants.append( ( 'D:some', x+'Q', Inhs[x]['r'], x ) )
        Scopes[x] = lowest
        lowest = x

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

    ## S1 rule...
    for q,R,S in list(Translations):
      if S[1] in Scopes:
        if VERBOSE: print( 'applying S1 to make (\\' + Scopes[ S[1] ] + ' ' + q + ' ' + str(R) + ' ' + str(S) + ')' )
        Abstractions[ Scopes[ S[1] ] ].append( (q, R, S) )
        del Scopes[ S[1] ]
        Translations.remove( (q, R, S) )
        active = True

    ## I1 rule...
    for src,lbldst in Inhs.items():
      for lbl,dst in lbldst.items():
        if dst not in Inhs and dst not in Abstractions and dst not in Expressions and dst not in Scopes.values() and dst not in [ x for Particips in Preds for x in Particips ]:
          if VERBOSE: print( 'applying I1 to make \\' + dst + ' True' )
          Abstractions[ dst ].append( () )
          active = True

    ## I2 rule...
    for src,lbldst in Inhs.items():
      for lbl,dst in lbldst.items():
        if dst in Expressions:
          if VERBOSE: print( 'applying I2 to replace ' + dst + ' with ' + src + ' to make ' + str(replaceVarName( Expressions[dst], dst, src )) )   #' in ' + str(Expressions[dst]) )
          Abstractions[ src ].append( replaceVarName( Expressions[dst], dst, src ) )
          del Inhs[src][lbl]
          if len(Inhs[src])==0: del Inhs[src]
          active = True

print( Translations )
