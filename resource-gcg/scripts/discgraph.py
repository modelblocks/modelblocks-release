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

import sys, collections, sets

VERBOSE = False
for a in sys.argv:
  if a=='-d':
    VERBOSE = True

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
      src,lbl,dst = assoc.split( ',', 2 )
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
    if VERBOSE: print( 'Subs = ' + str(D.Subs) )

    ## List of referents that are or participate in elementary predications...
    D.Referents = sets.Set( [ x for pred in D.PredTuples for x in pred[1:] ] + D.Inhs.keys() )


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
      if xLo in L:
        sys.stderr.write( 'ERROR: inheritance cycle: ' + str(L+[xLo]) + '\n' )
        print(           '#ERROR: inheritance cycle: ' + str(L+[xLo]) )
        return True
      return any([ checkInhCycles( xHi, L + [xLo] )  for l,xHi in D.Inhs.get(xLo,{}).items() ])
    ## Check for scope cycles...
    def checkScopeCyclesFromSup( xLo, L=[] ):
      if xLo in L:
        sys.stderr.write( 'ERROR: scope cycle: ' + str(L+[xLo]) + '\n' )
        print(           '#ERROR: scope cycle: ' + str(L+[xLo]) )
        return True
      return checkScopeCyclesInChain(D.Scopes[xLo],L+[xLo]) if xLo in D.Scopes else any([ checkScopeCyclesFromSup(xHi,L+[xLo]) for l,xHi in D.Inhs.get(xLo,{}).items() if l!='w' and l!='o' ])
    def checkScopeCyclesFromSub( xHi, L=[] ):
      if xHi in L:
        sys.stderr.write( 'ERROR: scope cycle: ' + str(L+[xHi]) + '\n' )
        print(           '#ERROR: scope cycle: ' + str(L+[xHi]) )
        return True
      return checkScopeCyclesInChain(D.Scopes[xHi],L+[xHi]) if xHi in D.Scopes else any([ checkScopeCyclesFromSub(xLo,L+[xHi]) for xLo in D.Subs.get(xHi,[]) ])
    def checkScopeCyclesInChain( x, L=[] ):
      return checkScopeCyclesFromSup( x, L ) or checkScopeCyclesFromSub( x, L )
    ## Check for inheritance cycles...
    for x in D.Referents:
      if checkInhCycles( x ): return
    ## Check for scopecycles...
    for x in D.Referents:
      if checkScopeCyclesInChain( x ): return
    ## Check for multiple outscopings...
    for x in D.Referents:
      if len( D.getBossesInChain(x) ) > 1:
        sys.stderr.write( 'WARNING: ' + x + ' has multiple outscopings in inheritance chain: ' + str( D.getBossesInChain(x) ) + '\n' )
        print(           '#WARNING: ' + x + ' has multiple outscopings in inheritance chain: ' + str( D.getBossesInChain(x) ) )
      if VERBOSE: print( 'Bosses of ' + x + ': ' + str(D.getBossesInChain(x)) )


  def normForm( D ):
    ## Smite redundant nuscos of predicative noun phrases out of Subs...
    for xHi,l in D.Subs.items():
      for xLo in l:
        if 'r' in D.Inhs.get(D.Inhs.get(xLo,[]).get('r',''),[]):
          if VERBOSE: print( 'Smiting ' + xLo + ' out of Subs, for being redundant.' )
          D.Subs[xHi].remove(xLo)
          if len(D.Subs[xHi])==0: del D.Subs[xHi]
          if xLo in D.Scopes:
            sys.stderr.write( 'ERROR: scope should not be annotated on redundant predicative referent: ' + xLo + '\n' )
            print(           '#ERROR: scope should not be annotated on redundant predicative referent: ' + xLo )
          if xLo in D.Subs:
            sys.stderr.write( 'ERROR: inheritance should not be annotated from ' + str(D.Subs[xLo]) + ' to redundant predicative referent: ' + xLo + '\n' )
            print(           '#ERROR: inheritance should not be annotated from ' + str(D.Subs[xLo]) + ' to redundant predicative referent: ' + xLo )

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



