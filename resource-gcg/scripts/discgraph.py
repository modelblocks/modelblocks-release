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

  #### Translate string representation into dict representation...
  def __init__( D, line ):

    ## Initialize associations...
    D.PorQs      = collections.defaultdict( list )                                     ## Key is elem pred.
    D.Scopes     = { }                                                                 ## Key is outscoped.
    D.Traces     = { }                                                                 ## Key is outscoped.
    D.Inhs       = collections.defaultdict( lambda : collections.defaultdict(float) )  ## Key is inheritor.
    D.Inheriteds = { }
    D.DiscInhs   = { }
    D.Referents  = [ ]
 
    ## For each assoc...
    for assoc in sorted( line.split(' ') ):
      src,lbl,dst = assoc.split( ',', 2 )
      if dst.startswith('N-bO:') or dst.startswith('N-bN:') or dst.startswith('N-b{N-aD}:') or dst.startswith('N-aD-b{N-aD}:'): dst += 'Q'
      D.Referents += [ src ] if lbl=='0' else [ src, dst ]
      if lbl.isdigit():  D.PorQs    [src].insert( int(lbl), dst )   ## Add preds and quants.
      elif lbl == 's':   D.Scopes   [src]      = dst                ## Add scopes.
      elif lbl == 't':   D.Traces   [src]      = dst                ## Add traces.
      elif lbl == 'm':   D.DiscInhs [src]      = dst                ## Add discource anaphor.
      else:              D.Inhs     [src][lbl] = dst                ## Add inheritances.
#      if lbl == 'r':     D.Nuscos [dst].append( src )             ## Index nusco of each restr.
#      if lbl == 'r':     D.NuscoValues[src]  = True
      if lbl == 'e':     D.Inheriteds[dst]   = True

    D.PredTuples  = [ ]
    D.QuantTuples = [ ] 

    ## Distinguish preds and quants...
    for elempred,Particips in D.PorQs.items():
      ## If three participants and last restriction-inherits from previous, it's a quant...
  #    if len( Particips ) == 3 and Inhs.get(Particips[2],{}).get('r','') == Particips[1]:  QuantTuples.append( tuple( [ Particips[0] ] + [ elempred ] + Particips[1:] ) )
      if Particips[0].endswith('Q'):  D.QuantTuples.append( tuple( [ Particips[0] ] + [ elempred ] + Particips[1:] + (['_'] if len(Particips)<4 else []) ) )
      else:                           D.PredTuples.append ( tuple( [ Particips[0] ] + [ elempred ] + Particips[1:] ) )

    D.OrigConsts = [ (ep[0],ep[1]) for ep in D.PredTuples ] + [ (q[0],'Q') for q in D.QuantTuples ]

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
    D.smite()  # only does smiting now.
    if VERBOSE: print( 'Subs = ' + str(D.Subs) )

    D.Nuscos = collections.defaultdict( list )                                     ## Key is restrictor.
    D.NuscoValues = { }

    ## Define nuscos after smiting...
    for xLo,lxHi in D.Inhs.items():
      for lbl,xHi in lxHi.items():
        if lbl=='r':
          D.Nuscos[xHi].append( xLo )
          D.NuscoValues[xLo] = True

#    ## List of referents that are or participate in elementary predications...
#    D.Referents = sorted( sets.Set( [ x for pred in D.PredTuples for x in pred[1:] ] + D.Inhs.keys() ) )
    D.Referents = sorted( sets.Set( D.Referents ) )


  #### Translate dict representation back into string representation...
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
    ## List discourse anaphor inheritancess...
    for xLo,xHi in D.DiscInhs.items():
      G.append( xLo + ',m,' + xHi )
    ## List scopes...
    for xLo,xHi in HypScopes.items():
      G.append( xLo + ',s,' + xHi )
    ## print out...
    return ' '.join( sorted( G ) )


  def smiteRecursively( D, x ):
    ## Recurse to all conjuncts...
    for xConjunct in D.Subs.get(x,[])[:]:
      if D.Inhs.get(xConjunct,{}).get('c','') == x:
        D.smiteRecursively( xConjunct )
    ## Remove all inheritances and subs immediately above smitten...
    for l,xHi in D.Inhs[x].items():
      D.Subs[xHi].remove(x)
      if len(D.Subs[xHi])==0: del D.Subs[xHi]
    del D.Inhs[x]
    ## Complain about subs immediately below smitten...
    for xLo in D.Subs[x]:
      sys.stderr.write( 'WARNING: inheritance -n should not be annotated from ' + xLo + ' to redundant predicative referent: ' + x + '\n' )
      print(           '#WARNING: inheritance -n should not be annotated from ' + xLo + ' to redundant predicative referent: ' + x )
    ## Complain about scopes immediately above smitten...
    if x in D.Scopes:
      sys.stderr.write( 'WARNING: scope -s should not be annotated on redundant predicative referent: ' + x + '\n' )
      print(           '#WARNING: scope -s should not be annotated on redundant predicative referent: ' + x )
    ## Complain about scopes immediately below smitten...
    if x in D.Scopes.values():
      sys.stderr.write( 'WARNING: scope -s should not be annotated *to* redundant predicative referent: ' + x + '\n' )
      print(           '#WARNING: scope -s should not be annotated *to* redundant predicative referent: ' + x )
    ## Complain about quants/preds connected to smitten...
    for tup in D.PredTuples:
      if x in tup:
        sys.stderr.write( 'WARNING: predicate ' + tup[0] + ' ' + tup[1] + ' should not include redundant predicative referent: ' + x + '\n' )
        print(           '#WARNING: predicate ' + tup[0] + ' ' + tup[1] + ' should not include redundant predicative referent: ' + x )
    ## Complain about quants/preds connected to smitten...
    for tup in D.QuantTuples:
      if x in tup:
        sys.stderr.write( 'WARNING: quantifier ' + tup[0] + ' ' + tup[1] + ' should not include redundant predicative referent: ' + x + '\n' )
        print(           '#WARNING: quantifier ' + tup[0] + ' ' + tup[1] + ' should not include redundant predicative referent: ' + x )
 

  '''
        del D.Inhs[xConjunct]['c']
        if len(D.Inhs[xConjunct])==0: del D.Inhs[xConjunct]
        D.Subs[x].remove(xConjunct)
        if len(D.Subs[x])==0: del D.Subs[x]
    for xHi,lxLo in D.Subs.items():
      for l,xLo in lxLo.items():
        if xLo == x:
          del lxLo[l]
          if xHi
    del D.Subs[x]
  '''

  #### Remove redundant nuclear scopes of predicative noun phrases from Subs and Inhs...
  def smite( D ):
    for xHi,L in D.Subs.items():
      for xLo in L:
        ## Diagnose as redundant if reft xLo has rin which also has rin...
        if 'r' in D.Inhs.get(D.Inhs.get(xLo,[]).get('r',''),[]):
          if VERBOSE: print( 'Smiting ' + xLo + ' out of Subs, for being redundant.' )
          D.smiteRecursively( xLo )
#          D.Subs[xHi].remove(xLo)
          if len(D.Subs[xHi])==0: del D.Subs[xHi]
#          del D.Inhs[xLo]['r']
          if len(D.Inhs[xLo])==0: del D.Inhs[xLo]
          '''
          if xLo in D.Scopes:
            sys.stderr.write( 'WARNING: scope (-s) should not be annotated on redundant predicative referent: ' + xLo + '\n' )
            print(           '#WARNING: scope (-s) should not be annotated on redundant predicative referent: ' + xLo )
          if xLo in D.Scopes.values():
            sys.stderr.write( 'WARNING: scope (-s) should not be annotated *to* redundant predicative referent: ' + xLo + '\n' )
            print(           '#WARNING: scope (-s) should not be annotated *to* redundant predicative referent: ' + xLo )
          BadSubs = [ x  for x in D.Subs.get(xLo,[])  for l,y in D.Inhs.get(x,{}).items()  if y == xLo and l != 'c' ]
          if BadSubs != []:
            sys.stderr.write( 'WARNING: inheritance (-n) should not be annotated from ' + ' '.join(BadSubs) + ' to redundant predicative referent: ' + xLo + '\n' )
            print(           '#WARNING: inheritance (-n) should not be annotated from ' + ' '.join(BadSubs) + ' to redundant predicative referent: ' + xLo )
          '''
          '''
            ## Modify bad subs to point to ...
            for x in BadSubs:
              for l,y in D.Inhs.get(x,{}).items():
                if y == xLo and l != 'c':
                  D.Inhs[x][l] = D.Inhs[xLo]['r']
                  #D.Subs[xLo].remove(x)
                  D.Subs[ D.Inhs[xLo]['r'] ].append( x )
                  if VERBOSE: print( '#NOTE: moving ' + l + ' inheritance of ' + x + ' from ' + xLo + ' to ' + D.Inhs[x][l] )
          '''
          if xLo in [ s  for q,e,r,s,n in D.QuantTuples ]:
            sys.stderr.write( 'WARNING: quantifier should not be annotated on redundant predicative referent: ' + xLo + '\n' )
            print(           '#WARNING: quantifier should not be annotated on redundant predicative referent: ' + xLo )


  '''
  ## Check that no reft has multiple outscopers...
  def getBossesFromSup( D, xLo ):
#      print( 'now getting sup ' + xLo )
    L = sets.Set([ xBoss  for xNusco in D.Nuscos.get(xLo,[])  if xNusco in D.Scopes  for xBoss in D.getBossesInChain( D.Scopes[xNusco] ) ])  ## Outscoper of nusco is outscoper of restrictor.
    if L != []: return L
    if xLo in D.Scopes: return D.getBossesInChain( D.Scopes[xLo] )
    return sets.Set( [ y  for l,xHi in D.Inhs.get(xLo,{}).items() if l!='w' and l!='o'  for y in D.getBossesFromSup(xHi) ] )
  def getBossesFromSub( D, xHi ):
#      print( 'now getting sub ' + xHi )
    if xHi in D.Scopes: return D.getBossesInChain( D.Scopes[xHi] )
    return sets.Set( [ y  for xLo in D.Subs.get(xHi,[])  for y in D.getBossesFromSub(xLo) ] )
  def getBossesInChain( D, x ):
    out = D.getBossesFromSup(x) | D.getBossesFromSub(x)
    return out if len(out)>0 else sets.Set( [x] )
  '''


  #### Validate disc graph against inheritance and scope cycles...
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
#      print( 'checking sup ' + xLo + ' with L=' + ' '.join(L) )
      if xLo in L:
        sys.stderr.write( 'ERROR: scope cycle: ' + str(L+[xLo]) + '\n' )
        print(           '#ERROR: scope cycle: ' + str(L+[xLo]) )
        return True
      if any([ checkScopeCyclesInChain( D.Scopes[xNusco], L+[xLo] )  for xNusco in D.Nuscos.get(xLo,[])  if xNusco in D.Scopes ]): return True  ## Outscoper of nusco is outscoper of restrictor.
      return ( checkScopeCyclesInChain(D.Scopes[xLo],L+[xLo]) if xLo in D.Scopes else False ) or any([ checkScopeCyclesFromSup(xHi,L+[xLo]) for l,xHi in D.Inhs.get(xLo,{}).items() if l!='w' and l!='o' ])
    def checkScopeCyclesFromSub( xHi, L=[] ):
#      print( 'checking sub ' + xHi + ' with L=' + ' '.join(L) )
      if xHi in L:
        sys.stderr.write( 'ERROR: scope cycle: ' + str(L+[xHi]) + '\n' )
        print(           '#ERROR: scope cycle: ' + str(L+[xHi]) )
        return True
#      return ( checkScopeCyclesInChain(D.Scopes[xHi],L+[xHi]) if xHi in D.Scopes else False ) or any([ checkScopeCyclesFromSub(xLo,L+[xHi]) for xLo in D.Subs.get(xHi,[]) ])
      return ( checkScopeCyclesFromSub(D.Scopes[xHi],L+[xHi]) if xHi in D.Scopes else False ) or any([ checkScopeCyclesFromSub(xLo,L+[xHi]) for xLo in D.Subs.get(xHi,[]) ])
    def checkScopeCyclesInChain( x, L=[] ):
#      print( 'checking ch ' + x + ' with L=' + ' '.join(L) )
      return checkScopeCyclesFromSup( x, L ) or checkScopeCyclesFromSub( x, L )

    ## Check for inheritance cycles...
    for x in D.Referents:
      if checkInhCycles( x ): return False
    ## Check for scopecycles...
    for x in D.Scopes.values(): #D.Referents:
#      if checkScopeCyclesInChain( x ): return False
      if checkScopeCyclesFromSub( x ): return False

    return True


  #### Validate discgraph against mulitple outscopers...
  def checkMultipleOutscopers( D ):
    def getScopersFromSup( xLo ):
      return ( [ (xLo,D.Scopes[xLo]) ] if xLo in D.Scopes else [] ) + [ x for l,xHi in D.Inhs.get(xLo,{}).items() if l!='w' and l!='o' for x in getScopersFromSup(xHi) ]
    def getScopersFromSub( xHi ):
      return ( [ (xHi,D.Scopes[xHi]) ] if xHi in D.Scopes else [] ) + [ x for xLo in D.Subs.get(xHi,[]) for x in getScopersFromSub(xLo) ]
    ## Obtain inheritance chain for each reft...
    Scopers = { x : sets.Set( getScopersFromSup(x) ) for x in D.Referents }   #+ getScopersFromSub(x)
    ## Check for multiple outscopings...
    for x in D.Referents:
      if len( Scopers[x] ) > 1:
        sys.stderr.write( 'WARNING: multiple outscopings found in same inheritance chain: ' + str( sorted(Scopers.get(x,[])) ) + '\n' )
        print(           '#WARNING: multiple outscopings found in same inheritance chain: ' + str( sorted(Scopers.get(x,[])) ) )
#        sys.stderr.write( 'WARNING: chain ' + str( sorted(Chains.get(x,[])) ) + ' has multiple outscopings: ' + str( D.getBossesInChain(x) ) + '\n' )
#        print(           '#WARNING: chain ' + str( sorted(Chains.get(x,[])) ) + ' has multiple outscopings: ' + str( D.getBossesInChain(x) ) )
#      if VERBOSE: print( 'Bosses of ' + x + ': ' + str(D.getBossesInChain(x)) )


#  def normForm( D ):


    '''
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
    '''
    '''
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
    '''


  ## Scope ceiling...
#  def getCeilingFromSup( xLo ):
#    return getCeilingInChain( Scopes[xLo] ) if xLo in Scopes else sets.Set( [ y  for l,xHi in Inhs.get(xLo,{}).items() if l!='w'  for y in getCeilingFromSup(xHi) ] )
#  def getCeilingFromSub( xHi ):
#    return getCeilingInChain( Scopes[xHi] ) if xHi in Scopes else sets.Set( [ y  for xLo in Subs.get(xHi,[])  for y in getCeilingFromSub(xLo) ] )
#  def getCeilingInChain( x ):
#    out = getCeilingFromSup( x ) | getCeilingFromSub( x )
#    return out if len(out)>0 else sets.Set( [x] )



