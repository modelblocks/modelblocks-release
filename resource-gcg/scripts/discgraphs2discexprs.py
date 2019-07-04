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
  PorQs  = collections.defaultdict( list )    ## Key is elem pred.
  Scopes = { }                                ## Key is outscoped.
  Inhs   = { }                                ## Key is inheritor.
  
  ## For each assoc...
  for assoc in sorted( line.split(' ') ):
    src,lbl,dst = assoc.split(',')
    if lbl.isdigit():  PorQs  [src].insert( int(lbl), dst )   ## Add preds and quants.
    elif lbl == 's':   Scopes [src]     = dst                 ## Add scopes.
    else:              Inhs   [src,lbl] = dst                 ## Add inheritances.

  Preds  = [ ]
  Quants = [ ] 

  ## Distinguish preds and quants:
  for elempred,Particips in PorQs.items():
    ## If three participants and last restriction-inherits from previous, it's a quant...
    if len( Particips ) == 3 and (Particips[2],'r') in Inhs and Inhs[ Particips[2], 'r' ] == Particips[1]:  Quants.append( tuple( [ Particips[0] ] + [ elempred ] + Particips[1:] ) )
    else:                                                                                                   Preds.append ( tuple( [ Particips[0] ] + [ elempred ] + Particips[1:] ) )

  Translations = [ ]
  Abstractions = collections.defaultdict( list )  ## Key is lambda.
  Expressions  = collections.defaultdict( list )  ## Key is lambda.

  ## Iterations...
  i = 0
  while Preds != [] or Quants != []:
    i += 1

    if VERBOSE: 
      print( '---- ITERATION ' + str(i) + ' ----' )
      print( 'P = ' + str(Preds) )
      print( 'Q = ' + str(Quants) )
      print( 'S = ' + str(Scopes) )
      print( 'I = ' + str(Inhs) )
      print( 'T = ' + str(Translations) )
      print( 'A = ' + str(Abstractions) )
      print( 'E = ' + str(Expressions) )

    ## P rule...
    for Participants in list(Preds):
      for particip in Participants[1:]:
        if particip not in Scopes.values() and (particip,'r') not in Inhs and (particip,'c') not in Inhs and (particip,'e') not in Inhs:
          if VERBOSE: print( 'applying P to make \\' + particip + '. ' + str(Participants) )
          Abstractions[ particip ].append( Participants )
          if Participants in Preds: Preds.remove( Participants )

    ## C rule...
    for var,Structs in Abstractions.items():
      if len(Structs) > 1:
        if VERBOSE: print( 'applying C to make \\' + var + '. ' + str( tuple( ['and'] + Structs ) ) )
        Abstractions[var] = [ tuple( ['and'] + Structs ) ]

    ## M rule...
    for var,Structs in Abstractions.items():
      if len(Structs) == 1 and var not in Scopes.values() and (var,'r') not in Inhs and (var,'c') not in Inhs and (var,'e') not in Inhs:
        if VERBOSE: print( 'applying M to make \\' + var + '. ' + str(Structs) )
        Expressions[var] = Structs[0]
        del Abstractions[var]

    ## Q rule...
    for q,e,r,s in list(Quants):
      if r in Expressions and s in Expressions:
        if VERBOSE: print( 'applying Q to make (' + q + ' (\\' + r + '. ' + str(Expressions[r]) + ') (\\' + s + '. ' + str(Expressions[s]) + '))' )
        Translations.append( ( q, ( 'lambda', r, Expressions[r] ), ( 'lambda', s, Expressions[s] ) ) )
        Quants.remove( (q, e, r, s) )

    ## S1 rule...
    for q,R,S in list(Translations):
      if S[1] in Scopes:
        if VERBOSE: print( 'applying S1 to make (\\' + Scopes[ S[1] ] + ' ' + q + ' ' + str(R) + ' ' + str(S) + ')' )
        Abstractions[ Scopes[ S[1] ] ].append( (q, R, S) )
        del Scopes[ S[1] ]
        Translations.remove( (q, R, S) )

    ## I1 rule...
    for srclbl,dst in Inhs.items():
      if dst not in Abstractions and dst not in Expressions and dst not in [ x for Particips in Preds for x in Particips ]:
        if VERBOSE: print( 'applying I1 to make \\' + dst + ' True' )
        Abstractions[ dst ].append( () )

    ## I2 rule...
    for srclbl,dst in Inhs.items():
      if dst in Expressions:
        if VERBOSE: print( 'applying I2 to replace ' + dst + ' with ' + srclbl[0] + ' to make ' + str(replaceVarName( Expressions[dst], dst, srclbl[0] )) )   #' in ' + str(Expressions[dst]) )
        Abstractions[ srclbl[0] ].append( replaceVarName( Expressions[dst], dst, srclbl[0] ) )
        del Inhs[srclbl]

print( Translations )
