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

import sys, os, re, collections
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
import gcgtree
import cuegraph

VERBOSE = False
EQN_DEFAULTS = False

################################################################################

class StoreStateCueGraph( cuegraph.CueGraph ):

  def dump( G ):
    print( 'G.a=' + G.a + ' G.b=' + G.b + ' ' + str(G) )


  def getArity( G, cat ):
    if re.findall( '\{[^{}]*-x[^{}]*\}', cat ):
      sys.stderr.write( 'WARNING -- category should not have \'-x\' delimiter inside braces: ' + cat + ' -- truncating!\n' )
      cat = re.sub( '\{([^{}]*)-x[^{}]*\}', '\\1', cat )
    cat = re.sub( '-x.*', '', cat )
    while '{' in cat:
      cat = re.sub('\{[^\{\}]*\}','X',cat)
    return len(re.findall('-[ab]',cat))

  ## match nolos back to front in cat and bottom up on store...
  def findNolos( G, nolos, n ):
    while True:
      if (n,'0') in G and G[n,'0']==nolos[-1]: nolos.pop()                                                        ## remove last nolo if n matches it
      if (n,'0') in G and not G[n,'0'].startswith('-') and len(nolos)>0 and nolos[-1] not in G[n,'0']: return ''  ## fail if reach full apex or base that does not contain last nolo
      if nolos == []: return n                                                                                    ## if removed nolo was leftmost, report n
      if   (n,'A') in G: n = G[n,'A']  ## advance n if A is next on store
      elif (n,'B') in G: n = G[n,'B']  ## advance n if B is next on store
      else: return ''

  def findNolo( G, sN, n ):
    while True:
      if (n,'0') in G and G[n,'0']==sN: return n
      if (n,'0') in G and not G[n,'0'].startswith('-') and sN not in G[n,'0']: return ''
      if   (n,'A') in G: n = G[n,'A']
      elif (n,'B') in G: n = G[n,'B']
      else: return ''

  def isNoloNeeded( G, n, d ):
    if n==d or (d,'0') not in G: return False
    if G[n,'0'] in G[d,'0']: return True
    if   (d,'A') in G: return G.isNoloNeeded( n, G[d,'A'] )
    elif (d,'B') in G: return G.isNoloNeeded( n, G[d,'B'] )
    return False


  def updateLex( G, f, sD, w, id ):

    if VERBOSE:
      G.dump( )
      print( 'f', f, sD, w, id )

    if f==0:
      ## rename 'S' and 'r' nodes...
      if (G.result('S',G.b),'r') in G:  G.rename( id+'r', G.result('r',G.result('S',G.b)) )
      G.rename( id+'s', G.result('S',G.b) )
      G.rename( id,     G.b )
      G.b = id
      G[G.b,'0'] = sD
      G.equate( w, 'X', G.b )
      G.a = G.result( 'A', G.b )
      while (G.a,'A') in G: G.a = G[G.a,'A']      ## traverse all non-local dependencies on A
    if f==1:
      G.a = id
      G.equate( sD, '0', G.a )
      G.equate( w,  'X', G.a )
      ## add all nonlocal dependencies with no nolo on store...
      b = G.a
      for sN in reversed( gcgtree.deps(sD) ):
        if sN[1] in 'ghirv' and not G.findNolo( sN, G.b ):
          b = G.result( 'B', b )
          G.equate( sN, '0', b )
      G.equate( G.b, 'B', b )
      ## rename 'r' nodes...
      if (G.result('S',G.a),'r') in G:  G.equate( id+'r', 'r', G.result('S',G.a) )

    ## attach rel pro / interrog pro antecedent...
    for i,psi in enumerate( gcgtree.deps(sD) ):
      if psi[1] in 'ir':
        G.equate( G.result('S',G.findNolo(psi,id)), str(G.getArity(sD)+1)+'\'',G.result('S',id) )
      '''
      if psi[1] in 'ir' and sD[0] in 'AR':
## should be on restr:        G.equate( G.result('S',G.findNolo(psi,id)), 'e', G.result('2\'',G.result('S',id)) )                ## adverbial relpro
## no 'e':         G.equate( G.result('r',G.result('S',G.findNolo(psi,id))), 'e', G.result('2\'',G.result('S',id)) )                ## adverbial relpro
# no 'r':       G.equate( G.result('r',G.result('S',G.findNolo(psi,id))), '2\'',G.result('S',id) )                ## adverbial relpro
        G.equate( G.result('S',G.findNolo(psi,id)), '2\'',G.result('S',id) )                ## adverbial relpro
      elif psi[1] in 'ir':
### no 'e':        G.equate( G.result('r',G.result('S',G.findNolo(psi,id))), 'e', G.result('r',G.result('S',id)) )    ## restrictive nominal relpro
## no 'r':        G.equate( G.result('r',G.result('S',G.findNolo(psi,id))), '1\'',G.result('S',id) )    ## restrictive nominal relpro
        G.equate( G.result('S',G.findNolo(psi,id)), '1\'',G.result('S',id) )    ## restrictive nominal relpro
# with 1':        G.equate( G.result('S',G.findNolo(psi,id)), 'S', id )    ## restrictive nominal relpro
### chg 'r' to 1':       G.equate( G.result('r',G.result('S',G.findNolo(psi,id))), 'r',G.result('S',id) )    ## restrictive nominal relpro
#### don't directly identify:       G.equate( G.result('r',G.result('S',G.findNolo(psi,id))), 'S', id )    ## restrictive nominal relpro
#      if VERBOSE: print( 'relpro')
#      if VERBOSE: G.dump()
      '''
  def updateUna( G, s, sC, sD, id ):

    if VERBOSE:
      G.dump( )
      print( 'S', s, sC, sD, id )

    n = ''
    l,d = ('B',G.a) if s==0 else ('A',G.b)
    dUpper,dLower = (G.a+'u',G.a) if s==0 else (G.b,G.b+'u')  ## bottom-up on left child, top-down on right child

    if '-lV' in sD:                                          ## V
      sN = re.findall('-v(?:[^-\{\}]|\{[^\{\}]*\})',sD)[-1]
      n = G.findNolo( sN, d )
      if n=='':
        n = G.result( l, d+'u' )
        G.equate( sN, '0', n )
        G.equate( G.result(l,d), l, n )
      else: G.equate( G.result(l,d), l, d+'u' )
      G.equate( G.result('S',n), '1\'', G.result('S',dUpper) )
      G.equate( G.result('S',dUpper), 'e', G.result('r',G.result('S',dLower)) )
    elif '-lF' in sD:                                        ## F (I?)
      sN = re.findall('-i(?:[^-\{\}]|\{[^\{\}]*\})',sD)[-1]
      n = G.findNolo( sN, d )
      if n=='':
        n = G.result( l, d+'u' )
        G.equate( sN, '0', n )
        G.equate( G.result(l,d), l, n )
      else: G.equate( G.result(l,d), l, d+'u' )
      G.equate( G.result('r',G.result('S',dUpper)), 'S', n )
    elif '-lQ' in sD:                                        ## Q
      G.equate( G.result(l,d), l, d+'u' )
      G.equate( G.result('1\'',G.result('S',d)), '2\'', G.result('S',d+'u') )  ## switch 1' & 2' arguments (same process top-down as bottom-up)
      G.equate( G.result('2\'',G.result('S',d)), '1\'', G.result('S',d+'u') )
      G.equate( G.result('S',dUpper), 'e', G.result('r',G.result('S',dLower)) )
    elif '-lZ' in sD and sC.startswith('N-aD-b{N-aD}'):      ## Zd (cardinal quant)
      G.equate( G.result(l,d), l, d+'u' )
      G.equate( G.result('S',dLower),                               '3', G.result('r',G.result('S',dUpper)) )
      G.equate( G.result('2\'',G.result('S',dUpper)),               '2', G.result('r',G.result('S',dUpper)) )
      G.equate( G.result('r',G.result('2\'',G.result('S',dUpper))), '1', G.result('r',G.result('S',dUpper)) )
      G.equate( 'D:eqQ',                                            '0', G.result('r',G.result('S',dUpper)) )
    elif '-lZ' in sD and sC.startswith('A-aN-x'):            ## Zc
      G.equate( G.result(l,d), l, d+'u' )
      G.equate( G.result('S',dLower),                 '2', G.result('r',G.result('S',dUpper)) )
      G.equate( G.result('1\'',G.result('S',dUpper)), '1', G.result('r',G.result('S',dUpper)) )
      G.equate( 'A-aN-bN:~',                          '0', G.result('r',G.result('S',dUpper)) )
    elif '-lZ' in sD and sC.startswith('A-a'):               ## Za
      if VERBOSE: print( 'applying Za' )
      G.equate( G.result(l,d), l, d+'u' )
#      G.equate( G.result('r',G.result('r',G.result('S',dLower))), '1\'', G.result('S',dUpper) )
      G.equate( G.result('r',G.result('S',dLower)), '1\'', G.result('S',dUpper) )
#      G.equate( G.result('S',dUpper), 'h', G.result('S',dLower) )              ## hypothetical world inheritance -- should be implemented throughout
      G.equate( G.result('S',dUpper), 'H', G.result('S',dLower) )    ## hypothetical world inheritance
# after 'S'      G.equate( G.result('S',dUpper), 'H', dLower )    ## hypothetical world inheritance
#      G.equate( G.result('S',dUpper), 'h', G.result('r',G.result('E',G.result('S',dLower))) )    ## hypothetical world inheritance
    elif '-lZ' in sD and sC.startswith('R-a'):               ## Zb (during)
      G.equate( G.result(l,d), l, d+'u' )
      G.equate( G.result('S',dLower),                 '2', G.result('r',G.result('S',dUpper)) )
      G.equate( G.result('1\'',G.result('S',dUpper)), '1', G.result('r',G.result('S',dUpper)) )
      G.equate( 'A-aN-bN:during',                     '0', G.result('r',G.result('S',dUpper)) )
    elif '-lz' in sD:                                        ## Zb (contain)
      G.equate( G.result(l,d), l, d+'u' )
      G.equate( G.result('S',dLower),                 '2', G.result('r',G.result('S',dUpper)) )
      G.equate( G.result('1\'',G.result('S',dUpper)), '1', G.result('r',G.result('S',dUpper)) )
      G.equate( 'B-aN-bN:contain',                    '0', G.result('r',G.result('S',dUpper)) )
    elif '-lE' in sD:
      nolos = gcgtree.deps( sC, 'ghirv' )
      if len(nolos)<1:
        sys.stderr.write( 'ERROR: opearator E not supported in absence of non-local dependencies: ' + sC + ' -> ' + sD + '\n' )
        print(            'ERROR: opearator E not supported in absence of non-local dependencies: ' + sC + ' -> ' + sD )
        exit( 1 )
      sN = nolos[0]
      n = G.findNolos( nolos, G[d,l] )
      if n=='':
        n = G.result( l, d+'u' )
        G.equate( sN, '0', n )
        G.equate( G.result(l,d), l, n )
      else: G.equate( G.result(l,d), l, d+'u' )
      if len( gcgtree.deps(sC) ) > len( gcgtree.deps(sD) ) and sN.endswith('-rN}'):  ## Ee
        if VERBOSE: print( 'using rule Ee' )
        G.equate( G.result('S',n), 'e', G.result('r',G.result('S',dLower)) )
      elif len( gcgtree.deps(sC) ) > len( gcgtree.deps(sD) ) and sN == '-g{V-aN}':  ## Ef
        if VERBOSE: print( 'using rule Ef' )
#      if sN.endswith('-aN}') or sN.endswith('-iN}') or sN.endswith('-rN}'):  ## Eb,Ed
        G.equate( G.result('1\'',G.result('S',n)), 'e', G.result('S',dLower) )
#        G.equate( G.result('s',G.result('S',dLower)), 's', G.result('1\'',G.result('S',n)) )    ## sent7
#        G.equate( G.result('r',G.result('S',dLower)), '1\'', id+'y' )
#        G.equate( G.result('S',n), 'e', id+'y' )
      elif len( gcgtree.deps(sC) ) > len( gcgtree.deps(sD) ) and sN == '-g{V-gN}':  ## Eg
        if VERBOSE: print( 'using rule Eg' )
        G.equate( G.result('S',dLower), 's', G.result('S',n) )
        G.equate( 'D:someQ',                     '0', G.result('Q',G.result('S',n)) )
        G.equate( G.result('r',G.result('S',n)), '1', G.result('Q',G.result('S',n)) )
        G.equate( G.result('S',n),               '2', G.result('Q',G.result('S',n)) )
      elif len( gcgtree.deps(sC) ) > len( gcgtree.deps(sD) ):  ## Ec,Ed
        if VERBOSE: print( 'using rule Ec/Ed' )
#      if sN.endswith('-aN}') or sN.endswith('-iN}') or sN.endswith('-rN}'):  ## Eb,Ed
        G.equate( G.result('1\'',G.result('S',n)), 'e', G.result('r',G.result('S',dLower)) )
#        G.equate( G.result('s',G.result('S',dLower)), 's', G.result('1\'',G.result('S',n)) )    ## sent7
#        G.equate( G.result('r',G.result('S',dLower)), '1\'', id+'y' )
#        G.equate( G.result('S',n), 'e', id+'y' )
      else:                                                    ## Ea,Eb
        if VERBOSE: print( 'using rule Ea/Eb' )
## should not add 'e'       G.equate( G.result('S',n), 'e', G.result( str(G.getArity(sD))+'\'', G.result('S',dLower) ) )
        G.equate( G.result('S',n), str(G.getArity(sD))+'\'', G.result('S',dLower) )
#        if '-g' in sN:                                       ## Ea,Eb
#          G.equate( G.result('r',G.result('S',n)), str(G.getArity(sD))+'\'', G.result('S',dLower) )
#        else:                                                ## Eh?,Ei? don't get restrictor for passive
#          G.equate( G.result('S',n), str(G.getArity(sD))+'\'', G.result('S',dLower) )
      G.equate( G.result('S',d), 'S', d+'u' )
    elif '-l' not in sD:                                     ## T
      ## update category of every nonlocal sign on store that changes with type change...
      hideps = gcgtree.deps( sC )
      lodeps = gcgtree.deps( sD )
      a = d+'u'
      for i in range( len(hideps) ):
        if i<len(lodeps) and hideps[i][1]!=lodeps[i][1] and hideps[i][1] in 'ghirv' and lodeps[i][1] in 'ghirv':
          a = G.result( l, a )
          G.equate( lodeps[i], '0', a )   ## note below only for s==1
          G.equate( G.result('S',G.findNolo(hideps[i],d)), 'S', a )
#          if s==0: G[ G.findNolo(lodeps[i],d), '0' ] = hideps[i]    ## bottom up on left child
#          else:    G[ G.findNolo(hideps[i],d), '0' ] = lodeps[i]    ## top down on right child
      if a == d+'u': return  ## don't add 'u' node
      G.equate( G[d,l], l, a )
    else:
      sys.stderr.write( 'ERROR: No analysis for annotated unnary expansion ' + sC + ' -> ' + sD + ' at ' + str(id) + '.\n' )
      print(            'ERROR: No analysis for annotated unnary expansion ' + sC + ' -> ' + sD + ' at ' + str(id) )
      return  ## don't add 'u' node

    ## add 'u' node
    if s==0:
      G.a = d+'u'
      G.equate( sC, '0', d+'u' )
    if s==1:
      G.b = d+'u'
      G.equate( sD, '0', d+'u' )

    if n!='' and not G.isNoloNeeded(n,d+'u'): G[n,'0']+='-closed'


  def updateBin( G, j, sC, sD, sE, id ):

    if VERBOSE:
      G.dump( )
      print( 'j', j, sC, sD, sE, id )

    G.b = id + 'b'
    G.equate( sE, '0', G.b )
    if j==0:
      c = id + 'a'
#      G.equate( c, 'A', G.result('A',G.b) if '-lG' in sD or '-lI' in sE or '-lR' in sE or re.match('.*-[ri]N-lH$',sE)!=None or re.match('.*-g{.-aN}-lH$',sE)!=None or sE.endswith('-g{V-gN}-lC') else G.b )
      G.equate( c, 'A', G.result('A',G.b) if '-lG' in sD or '-lI' in sE or '-lR' in sE or ('-lH' in sE and re.match('.*-h\{[^\}]*-[ri]N\}$',sD)!=None) or re.match('.*-g\{.-aN\}-lH$',sE)!=None or sE.endswith('-g{V-gN}-lC') else G.b )
      G.equate( sC, '0', c )
      ## add all nonlocal dependencies with no nolo on store...
      b = c
      for sN in reversed( gcgtree.deps(sC) ):
        if sN[1] in 'ghirv' and G.findNolo( sN, G[G.a,'B'] )=='':
          b = G.result( 'B', b )
          G.equate( sN, '0', b )
      G.equate( G.result('B',G.a), 'B', b )
    if j==1:
      c = G.result( 'B', G.a )
      while (c,'B') in G: c = G[c,'B']            ## if there are non-local dependencies on B
#      G.equate( G.result('A',c), 'A', G.result('A',G.b) if '-lG' in sD or '-lI' in sE or '-lR' in sE or re.match('.*-[ri]N-lH$',sE)!=None or re.match('.*{.-aN}-lH$',sE)!=None or sE.endswith('-g{V-gN}-lC') else G.b )
      G.equate( G.result('A',c), 'A', G.result('A',G.b) if '-lG' in sD or '-lI' in sE or '-lR' in sE or ('-lH' in sE and re.match('.*-h\{[^\}]*-[ri]N\}$',sD)!=None) or re.match('.*\{.-aN\}-lH$',sE)!=None or sE.endswith('-g{V-gN}-lC') else G.b )

    d,e = G.a,G.b
    if   '-lD' in sD:                               ## Da
      G.equate( G.result('S',c), 'S', e )
    elif '-lD' in sE:                               ## Db
      G.equate( G.result('S',d), 'S', c )
    elif '-lA' in sD:                               ## Aa
      G.equate( G.result('S',c), 'S', e )
      G.equate( G.result('S',d), str(G.getArity(sE))+'\'', G.result('S',e) )
    elif '-lA' in sE:                               ## Ab
      G.equate( G.result('S',d), 'S', c )
#      G.equate( G.result('r',G.result('S',d)), 'r', G.result('S',c) )   ## rename 'r' node as well as 'S'
      G.equate( G.result('S',e), str(G.getArity(sD))+'\'', G.result('S',d) )
    elif '-lU' in sD:                               ## Ua
      G.equate( G.result('S',d), 'S', c )
      G.equate( G.result('S',d), str(G.getArity(sE))+'\'', G.result('S',e) )
    elif '-lU' in sE:                               ## Ub
      G.equate( G.result('S',c), 'S', e )
      G.equate( G.result('S',e), str(G.getArity(sD))+'\'', G.result('S',d) )
    elif '-lM' in sD:                               ## Ma
      G.equate( G.result('S',c), 'S', e )
      G.equate( G.result('r',G.result('S',e)), '1\'', G.result('S',d) )
    elif '-lM' in sE:                               ## Mb
      G.equate( G.result('S',d), 'S', c )
      G.equate( G.result('r',G.result('S',d)), '1\'', G.result('S',e) )
    elif '-lC' in sD and '-lC' in sE:               ## Cd
      G.equate( G.result('S',c), 'c', G.result('S',d) )
      G.equate( G.result('S',c), 'c', G.result('S',e) )
      G.equate( G.result('r',G.result('S',c)), 'c', G.result('r',G.result('S',d)) )
      G.equate( G.result('r',G.result('S',c)), 'c', G.result('r',G.result('S',e)) )
      G.equate( G.result('H',G.result('S',d)), 'H', G.result('S',c) )
      G.equate( G.result('H',G.result('S',c)), 'H', G.result('S',e) )
      for i in range( 1, len(gcgtree.deps(sC,'ab'))+1 ): #G.getArity(sC)+1 ):
        G.equate( G.result(str(i)+'\'',G.result('S',c)), str(i)+'\'', G.result('S',d) )
        G.equate( G.result(str(i)+'\'',G.result('S',c)), str(i)+'\'', G.result('S',e) )
    elif '-lC' in sD:                               ## Ca,Cb
      G.equate( G.result('S',c), 'S', e )
      G.equate( G.result('r',G.result('S',e)), 'c', G.result('r',G.result('S',d)) )
      G.equate( G.result('S',e), 'c', G.result('S',d) )
      G.equate( G.result('H',G.result('S',d)), 'H', G.result('S',c) )
      G.equate( G.result('H',G.result('S',c)), 'H', G.result('S',e) )
      for i in range( 1, len(gcgtree.deps(sC,'ab'))+1 ): #G.getArity(sC)+1 ):
        G.equate( G.result(str(i)+'\'',G.result('S',c)), str(i)+'\'', G.result('S',d) )
    elif '-lC' in sE:                               ## Cc
      G.equate( G.result('S',d), 'S', c )
      G.equate( G.result('r',G.result('S',d)), 'c', G.result('r',G.result('S',e)) )
      G.equate( G.result('S',d), 'c', G.result('S',e) )
      G.equate( G.result('H',G.result('S',d)), 'H', G.result('S',c) )
      G.equate( G.result('H',G.result('S',c)), 'H', G.result('S',e) )
      for i in range( 1, len(gcgtree.deps(sC,'ab'))+1 ):  #G.getArity(sC)+1 ):
        G.equate( G.result(str(i)+'\'',G.result('S',c)), str(i)+'\'', G.result('S',e) )
      if sE.endswith('-g{V-gN}-lC'):                ## Ce
        G.equate( gcgtree.lastdep(sE), '0', G.result('A',e) )
#        G.equate( G.result('W',G.result('S',d)), 'w', G.result('S',G.result('A',e)) )
#        G.equate( G.result('s',G.result('W',G.result('S',d))), 't', G.result('W',G.result('S',d)) )
        G.equate( G.result('W',d), 'w', G.result('S',G.result('A',e)) )
        G.equate( G.result('s',G.result('W',d)), 't', G.result('W',d) )
    elif '-lG' in sD:                               ## G
      G.equate( G.result('S',c), 'S', e )
      G.equate( G.result('S',d), 'S', G.result('A',e) )  # grab nusco
#      G.equate( G.result('r',G.result('S',d)), 'S', G.result('A',e) )  # grab restrictor
      G.equate( gcgtree.lastdep(sE), '0', G.result('A',e) )
    elif '-lH' in sE:                               ## H
      G.equate( G.result('S',d), 'S', c )
      n = G.findNolo( gcgtree.lastdep(sD), d )
      if n!='':
#        if re.match('.*-[ri]N-lH$',sE)!=None:       ## Hb
        if '-lH' in sE and re.match('.*-h\{[^\}]*-[ri]N\}$',sD)!=None:       ## Hb
          G.equate( G.result('r',G.result('S',n)), 'S', G.result('A',e) )
          G.equate( gcgtree.lastdep(sE), '0', G.result('A',e) )
        elif re.match('.*-g\{.-aN\}-lH$',sE)!=None:   ## Hc
          G.equate( G.result('S',n), 'S', G.result('A',e) )
          G.equate( gcgtree.lastdep(sE), '0', G.result('A',e) )
        else:                                       ## Ha
          G.equate( G.result('S',e), 'S', n )
        G[n,'0']+='-closed'  ## close off nolo
    elif '-lI' in sE:                               ## I
      G.equate( G.result('S',d), 'S', c )
      if sD.startswith('N-b{V-g{') or sD.startswith('N-b{I-aN-g{'):                               ## nominal clause with relation extraction
        G.equate( G.result('S',d), 'S', G.result('A',e) )
      elif sD.startswith('N-b{V-g') or sD.startswith('N-b{I-aN-g'):                               ## nominal clause with nominal extraction
        G.equate( G.result('r',G.result('S',d)), 'S', G.result('A',e) )
      elif '-b{I-aN-gN}' in sD:                                                                   ## tough construction
        G.equate( G.result('1\'',G.result('S',d)), 'S', G.result('A',e) )
        G.equate( G.result('S',e), str(G.getArity(sD))+'\'', G.result('S',d) )
      else: G.equate( G.result('S',G.result('A',e)), str(G.getArity(sD))+'\'', G.result('S',d) )  ## embedded question
      G.equate( gcgtree.lastdep(sE), '0', G.result('A',e) )
    elif '-lR' in sD:                               ## Ra (off-spec)
      G.equate( G.result('S',c), 'S', e )
#      G.equate( G.result('S',e), 'S', G.result('B',d) )
      G.equate( G.result('r',G.result('S',e)), 'S', G.result('B',d) )  # grab restrictor, tho 'which' gets put on restrictor of nolo
      G.equate( gcgtree.lastdep(sD), '0', G.result('B',d) )
    elif 'I-aN-g{R-aN}-lR' == sE:                   ## Rc (off-spec)
      G.equate( G.result('S',d),     'S', c )
      G.equate( 'A-aN-bN:support',                             '0', G.result('r',G.result('S',G.result('A',e))) )
      G.equate( G.result('1\'',G.result('S',G.result('A',e))), '1', G.result('r',G.result('S',G.result('A',e))) )
      G.equate( G.result('r',G.result('S',d)),                 '2', G.result('r',G.result('S',G.result('A',e))) )
      G.equate( gcgtree.lastdep(sE), '0', G.result('A',e) )
    elif sE.endswith('-g{R-aN}-lR'):                ## Rd (off-spec: "the only/furthest north I can go")
      G.equate( G.result('S',d), 'S', c )
      G.equate( G.result('S',d), 'S', G.result('A',e) )  # grab nusco
      G.equate( gcgtree.lastdep(sE), '0', G.result('A',e) )
    elif '-lR' in sE:                               ## R
      G.equate( G.result('S',d), 'S', c )
#      G.equate( G.result('S',d), 'S', G.result('A',e) )
      G.equate( G.result('r',G.result('S',d)), 'S', G.result('A',e) )  # grab restrictor, tho 'which' gets put on restrictor of nolo
      G.equate( gcgtree.lastdep(sE), '0', G.result('A',e) )
    elif sD.startswith(',') and sE.endswith('-pPc') or sD.startswith(';') and sE.endswith('-pPs'):
      G.equate( G.result('S',c), 'S', e )
    elif sD=='U' and sE=='U':
      G.equate( G.result('S',d), 'S', c )
      G.equate( G.result('S',c), 'S', e )
    else:
      if sC != 'FAIL':   #sC != sD != sE != 'FAIL':
        sys.stderr.write( 'WARNING: No analysis for annotated binary expansion ' + sC + ' -> ' + sD + ' ' + sE + ' at ' + str(id) + '.\n' )
    if ( '-lE' in sD or '-lE' in sE or
         '-lZ' in sD or '-lZ' in sE or
         '-lz' in sD or '-lz' in sE or
         '-lQ' in sD or '-lQ' in sE or
         '-lF' in sD or '-lF' in sE or
         '-lV' in sD or '-lV' in sE ):
      sys.stderr.write( 'WARNING: Illegal unary operator in binary expansion ' + sC + ' -> ' + sD + ' ' + sE + ' at ' + str(id) + '.\n' )


  def convert( G, t, sentnumprefix='', s=0, i=0 ):

    ## terminal...
    if len(t.ch)==1 and len(t.ch[0].ch)==0:
      i += 1
      G.updateLex( 1-s, t.c, t.ch[0].c, sentnumprefix+('0' if i<10 else '')+str(i) )

    ## nonterminal unary...
    elif len(t.ch)==1:
      if s==1: G.updateUna( s, t.c, t.ch[0].c, sentnumprefix+('0' if i<10 else '')+str(i) )  ## update storestate preorder if right child (to build on G.b)
      i = G.convert( t.ch[0], sentnumprefix, s, i )
      if s==0: G.updateUna( s, t.c, t.ch[0].c, sentnumprefix+('0' if i<10 else '')+str(i) )  ## update storestate postorder if left child (to build on G.a)

    ## nonterminal binary...
    elif len(t.ch)==2:
      i = G.convert( t.ch[0], sentnumprefix, 0, i )
      G.updateBin( s, t.c, t.ch[0].c, t.ch[1].c, sentnumprefix+('0' if i<10 else '')+str(i) )
      i = G.convert( t.ch[1], sentnumprefix, 1, i )

    ## unsupported n-ary branch...
    else:
      sys.stderr.write( 'ERROR: No analysis for super-binary branch: ' + str(t) + '\n' )

    return i


#  def __init__( G, t, sentnumprefix='' ):
  def add( G, t, sentnumprefix='' ):
    
#    gcgtree.relabel( t )
    if VERBOSE: print( t )

    G.a = 'top'
    G.b = 'bot'
    G.convert( t, sentnumprefix )

    ## for each word...
    for x,l in sorted(G):
      ## add predicates by applying lex rules...
      if l=='X':
#        print('AAAAA')
#        G.dump()
        ## rename 'S' and 'r' nodes...
        if (x,    'S') in G: G.rename( x+'s', G[x,    'S'] )
        if (x+'s','r') in G: G.rename( x+'r', G[x+'s','r'] )
        ## apply -n, -m, and -s tags...
          # TODO this repetitious warning code could use an abstraction
        if len( re.findall( '-n',G[x,'0'] ) ) > 1:
          sys.stderr.write( 'ERROR: multiple -n tags in category ' + G[x,'0'] + ' -- these will be unified, which is probably not desired!\n' )
        if len( re.findall( '-nn',G[x,'0'] ) ) > 0:
          sys.stderr.write( 'ERROR: -nn tag in category ' + G[x,'0'] + ' -- not implemented!\n' )
        if len( re.findall( '-s',G[x,'0'] ) ) > 1:
          sys.stderr.write( 'ERROR: multiple -s tags in category ' + G[x,'0'] + ' -- these will be unified, which is probably not desired!\n' )
        if len( re.findall( '-t',G[x,'0'] ) ) > 1:
          sys.stderr.write( 'ERROR: multiple -t tags in category ' + G[x,'0'] + ' -- these will be unified, which is probably not desired!\n' )
        if len( re.findall( '-u',G[x,'0'] ) ) > 1:
          sys.stderr.write( 'ERROR: multiple -u tags in category ' + G[x,'0'] + ' -- these will be unified, which is probably not desired!\n' )
        if len( re.findall( '-uu',G[x,'0'] ) ) > 1:
          sys.stderr.write( 'ERROR: multiple -uu tags in category ' + G[x,'0'] + ' -- these will be unified, which is probably not desired!\n' )
        if len( re.findall( '-w',G[x,'0'] ) ) > 1:
          sys.stderr.write( 'ERROR: multiple -w tags in category ' + G[x,'0'] + ' -- these will be unified, which is probably not desired!\n' )
        if len( re.findall( '-ww',G[x,'0'] ) ) > 1:
          sys.stderr.write( 'ERROR: multiple -ww tags in category ' + G[x,'0'] + ' -- these will be unified, which is probably not desired!\n' )
        for tag,dest,r in re.findall( '-([mnstuw]+)([0-9]+)(r?)', G[x,'0'] ):
          dest = sentnumprefix+dest if len(dest) == 2 else dest
          if   tag=='m' and r=='r': G.equate( dest+'r', 'm', x+'r' )  ## discourse anaphor
          elif tag=='m':            G.equate( dest+'s', 'm', x+'r' )  ## discourse anaphor
          elif tag=='n' and r=='r': G.equate( dest+'r', 'n', x+'r' )
          elif tag=='n':            G.equate( dest+'s', 'n', x+'r' )
          elif tag=='t' and r=='r':  ## scotch tape (suggested scope for lambda expressions, but not to be used for scoring)
                                    G.equate( dest+'r', 's', x+'s' )
                                    G.equate( dest+'r', 'tt', x+'s' ) # 'tainted' not truth-conditionally necessary
          elif tag=='t':  ## scotch tape (suggested scope for lambda expressions, but not to be used for scoring)
                                    G.equate( dest+'s', 's', x+'s' )
                                    G.equate( dest+'s', 'tt', x+'s' ) # 'tainted' not truth-conditionally necessary
          elif tag=='s' and r=='r': G.equate( dest+'r', 's', x+'s' )
          elif tag=='s':            G.equate( dest+'s', 's', x+'s' )
          elif tag=='u':            G.equate( dest+'s', 'u', x+'s' ) # 'upward' outscoper but not necessarily immediate outscoper
          elif tag=='uu':           G.equate( dest+'s', 'uu', x+'s' ) # ditto
          # as of October 2020 nothing becomes of -w but at least they'll be in the graph
          elif tag=='w':            G.equate( dest+'s', 'W', x )
          elif tag=='ww':           G.equate( dest+'s', 'WW', x )
          else: sys.stderr.write( 'ERROR: unknown tag ' + tag + ' in category ' + G[x,'0'] + '\n' )
        G[x,'0'] = re.sub( '-[mnstuw]+[0-9]+r?', '', G[x,'0'] ) # delete the tags
        ## obtain pred by applying lex rules to word token...
        s = re.sub('-l.','',G[x,'0']) + ':' + G[x,'X'].lower()
        eqns = re.sub( '-x.*:', ':', s )
        for xrule in re.split( '-x', G[x,'0'] )[1:] :   #re.findall( '(-x(?:(?!-x).)*)', s ):
          ## replace macro lex rules with compositional rules...
          if   xrule == 'COPU'  :  xrule = '%|21=1'
          elif xrule == 'MDL'   :  xrule = '%|Qr0=%Q^Qr1=r^Qr2=^Er0=B:contain^Er1=^Er2=2r'
          elif xrule == 'CONDL' :  xrule = '%|Qr0=%Q^Qr1=r^Qr2=^Er0=B:contain^Er1=^Er2=2r^Fr0=B:contain^Fr1=r^Fr2=1'
          elif xrule == 'DPOSS' :  xrule = '%|Er0=B-aN-bN:have^Er1=2^Er2=1r'
          elif xrule == 'NOTHAVE' : xrule = '%|r0=D:noneQ^r1=Er^r2=E^Er0=B-aN-bN:have^Er1=1^Er2=2'
          elif xrule == 'NGEN'  :  xrule = '%|Qr0=D:genQ^Qr1=r^Qr2=^Er0=%^Er1=r' + ''.join( [ '^Er'+str(i  )+'='+str(i) for i in range(2,G.getArity(G[x,'0'])+1) ] )
          elif xrule == 'NEXI'  :  xrule = '%|Qr0=D:someQ^Qr1=r^Qr2=^Er0=%^Er1=r' + ''.join( [ '^Er'+str(i  )+'='+str(i) for i in range(2,G.getArity(G[x,'0'])+1) ] )
          elif xrule == 'NUNI'  :  xrule = '%|Qr0=D:allQ^Qr1=r^Qr2=^Er0=%^Er1=r' + ''.join( [ '^Er'+str(i  )+'='+str(i) for i in range(2,G.getArity(G[x,'0'])+1) ] )
          elif xrule == 'NORD'  :  xrule = '%|Qr0=%DecOneQ^Qr1=2r^Qr2=2^ro=2r^Rr0=A:prec^Rr1=2^Rr2=r^Rrh=SH'
          elif xrule == 'NORDEXI' :  xrule = '%|Rr0=D:someQ^Rr1=r^Rr2=^Qr0=%DecOneQ^Qr1=2r^Qr2=2^ro=2r^Pr0=A:prec^Pr1=2^Pr2=r^Prh=SH'
          elif xrule == 'NORDSUP' :  xrule = '%|Qr0=%DecOneQ^Qr1=2r^Qr2=2^ro=2r^31=2^3r1h=SH^Pr0=3r0^Pr1=r^Rr0=A:gt^Rr1=3r2^Rr2=Pr2'
          elif xrule == 'NSUP'  :  xrule = '%|Qr0=D:moreQ^Er0=%ness^Er1=r^Er2=Qr1^Fr0=Er0^Fr1=2^Fr2=Qr2^Pr0=D:allQ^Pr1=2r^Pr2=2' #^Qr2s=2^Qr1s=Qr2'  ## should induce this; should not crash if added; why not?
          elif xrule == 'NSUP3' :  xrule = '%|Qr0=D:moreQ^Er0=3r0^Er1=r^Er2=Qr1^Fr0=Er0^Fr1=2^Fr2=Qr2^Pr0=D:allQ^Pr1=2r^Pr2=2'
          elif xrule == 'NSUPEXI' :  xrule = '%|Rr0=D:someQ^Rr1=r^Rr2=^Qr0=D:moreQ^Er0=%ness^Er1=r^Er2=Qr1^Fr0=Er0^Fr1=2^Fr2=Qr2^Pr0=D:allQ^Pr1=2r^Pr2=2'
          elif xrule == 'NCOMP' :  xrule = '%|Er0=%^Er1=r^Er2=2^2w=^t=s' #^Q0=D:someDummyQ^Q1=31r^Q2=31'
          elif xrule == 'NCOMP3' :  xrule = '%|Er0=%^Er1=2^Er2=3^3w=^t=s'
          elif xrule == 'NOUN'  :  xrule = '%|Er0=%^Er1=r' + ''.join( [ '^Er' +str(i  )+'='+str(i) for i in range(2,G.getArity(G[x,'0'])+1) ] ) + '^Erh=SH'
          elif xrule == 'NQUANT' :  xrule = '%|Qr0=%Q^Qr1=r^Qr2='
          elif xrule == 'NRELEXI'  :  xrule = '%|Qr0=D:someQ^Qr1=r^Qr2=^Er0=%^Er1=r' + ''.join( [ '^Er' +str(i+1)+'='+str(i) for i in range(1,G.getArity(G[x,'0'])+1) ] ) + '^Erh=SH'
          elif xrule == 'NREL'  :  xrule = '%|Er0=%^Er1=r' + ''.join( [ '^Er' +str(i+1)+'='+str(i) for i in range(1,G.getArity(G[x,'0'])+1) ] ) + '^Erh=SH'
          elif xrule == 'NREL2' :  xrule = '%|Er0=%^Er1=r^Er2=2'
          elif xrule == 'PGEN'  :  xrule = '%|Qr0=D:genQ^Qr1=r^Qr2=^r0=%' + ''.join( [ '^r' +str(i  )+'='+str(i) for i in range(1,G.getArity(G[x,'0'])+1) ] )
          elif xrule == 'PRED1' :  xrule = '%|r0=%^r1=1'
          elif xrule == 'PRED2' :  xrule = '%|r0=%^r1=2'  ## 'the form of a mountain'
          elif xrule == 'PRED12':  xrule = '%|r0=%^r1=1^r2=2'
          elif xrule == 'PRED13':  xrule = '%|r0=%^r1=1^r2=3'
          elif xrule == 'PRED'  :  xrule = '%|r0=%' + ''.join( [ '^r' +str(i  )+'='+str(i) for i in range(1,G.getArity(G[x,'0'])+1) ] )
          elif xrule == 'PRAIS' :  xrule = '%|21=1^r0=%' + ''.join( [ '^r' +str(i  )+'='+str(i) for i in range(1,G.getArity(G[x,'0'])+1) ] )
          elif xrule == 'QGEN'  :  xrule = '%|r0=D:genQ^r1=1r^r2=1'
          elif xrule == 'QUANT' :  xrule = '%|r0=%Q^r1=1r^r2=1'
          elif xrule == 'QUANT2' :  xrule = '%|r0=%Q^r1=2r^r2=2'
          elif xrule == 'QUANT12' :  xrule = '%|r0=%Q^r1=1r^r2=1' #^r3=1'  ## number should be connected, but not sure if lambdafier can accommodate
          elif xrule == 'QUANT23' :  xrule = '%|r0=%Q^r1=2r^r2=2' #^r3=1'  ## number should be connected, but not sure if lambdafier can accommodate
          elif xrule == 'WEAK2' :  xrule = '%|%^2=W'
          elif xrule == 'XEXI'  :  xrule = '%|Qr0=D:someQ^Qr1=r^Qr2='
          elif xrule == 'XGEN'  :  xrule = '%|Qr0=D:genQ^Qr1=r^Qr2='
          ## apply compositional lex rules...
          m = re.search( '(.*)%(.*)%(.*)\|(.*)%(.*)%(.*)', xrule )
          if m is not None:
            eqns = re.sub( '^'+m.group(1)+'(.*)'+m.group(2)+'(.*)'+m.group(3)+'$', m.group(4)+'\\1'+m.group(5)+'\\2'+m.group(6), eqns )
            continue
          m = re.search( '(.*)%(.*)\|(.*)%(.*)', xrule )
          if m is not None:
            eqns = re.sub( '^'+m.group(1)+'(.*)'+m.group(2)+'$', m.group(3)+'\\1'+m.group(4), eqns )
            continue
          m = re.search( '.*%.*\|(.*)', xrule )
          if m is not None: eqns = m.group(1)
#        s = eqns

        ## apply default lex sems...
        if EQN_DEFAULTS and ':' in eqns and '=' not in eqns:
          ## quantifier semantics...
          if   eqns.startswith('N-b{N-aD}:'):    eqns = 'r0='  + eqns + '^r1=1r^r2=1'
          elif eqns.startswith('N-aD-b{N-aD}:'): eqns = 'r0='  + eqns + '^r1=2r^r2=2'
          elif eqns.startswith('N-aD-b{N-aD}-bN:'): eqns = 'r0='  + eqns + '^r1=2r^r2=2^r3=3'
          elif eqns.startswith('N-bN:'):         eqns = 'r0='  + eqns + '^r1=1r^r2=1'
          elif eqns.startswith('N-bO:'):         eqns = 'r0='  + eqns + '^r1=1r^r2=1'
          ## relative/interrogative pronoun semantics...
          elif eqns.startswith('A-aN-iN'):       eqns = 'r0='  + eqns +            ''.join( [ '^r' +str(i  )+'='+str(i) for i in range(1,3) ] )
          elif eqns.startswith('A-aN-rN'):       eqns = 'r0='  + eqns +            ''.join( [ '^r' +str(i  )+'='+str(i) for i in range(1,3) ] )
          elif eqns.startswith('R-aN-iN'):       eqns = 'r0='  + eqns +            ''.join( [ '^r' +str(i  )+'='+str(i) for i in range(1,3) ] )
          elif eqns.startswith('R-aN-rN'):       eqns = 'r0='  + eqns +            ''.join( [ '^r' +str(i  )+'='+str(i) for i in range(1,3) ] )
          ## elementary predications...
          elif eqns.startswith('A'):             eqns = 'r0='  + eqns +            ''.join( [ '^r' +str(i  )+'='+str(i) for i in range(1,G.getArity(G[x,'0'])+1) ] )
          elif eqns.startswith('R'):             eqns = 'r0='  + eqns +            ''.join( [ '^r' +str(i  )+'='+str(i) for i in range(1,G.getArity(G[x,'0'])+1) ] )
          elif eqns.startswith('B'):             eqns = 'r0='  + eqns +            ''.join( [ '^r' +str(i  )+'='+str(i) for i in range(1,G.getArity(G[x,'0'])+1) ] ) + '^rh=SH'
          ## relative/interrogative pronoun semantics...
          elif eqns.startswith('N-iN'):          eqns = 'Er0=' + eqns + '^Er1=^1='
          elif eqns.startswith('N-rN'):          eqns = 'Er0=' + eqns + '^Er1=^1='   ## NOTE: FIRST SYNARG IS ITSELF -- PROBLEMATIC FOR RELCLAUSE ON DEVERBAL NOUN
          ## possessive pronouns (not in discgraph spec, but handled if extended morph not used)...
          elif eqns.startswith('D'):             eqns = 'Er0=' + eqns + '^Er1=r' + ''.join( [ '^Er'+str(i  )+'='+str(i) for i in range(2,G.getArity(G[x,'0'])+1) ] ) + '^Erh=SH'
          ## relational/nonrelational nouns...
          elif eqns.startswith('N'):             eqns = 'Er0=' + eqns + '^Er1=r' + ''.join( [ '^Er'+str(i  )+'='+str(i) for i in range(2,G.getArity(G[x,'0'])+1) ] ) + '^Erh=SH'
          ## unanalyzed labels...
          elif eqns.startswith('U'):             eqns = 'Er0=' + eqns + '^Er1=r' + ''.join( [ '^Er'+str(i  )+'='+str(i) for i in range(2,G.getArity(G[x,'0'])+1) ] ) + '^Erh=SH'
          ## other function words, treating as predicate (not in discgraph spec, but handled if extended morph not used)...
          else:                                  eqns = 'r0='  + eqns +            ''.join( [ '^r' +str(i  )+'='+str(i) for i in range(1,G.getArity(G[x,'0'])+1) ] )
#          G.dump()
          if VERBOSE: print( 'Inducing default equation: ' + eqns )

        if '-x' in G[x,'0'] and eqns != '' and '=' not in eqns:
          sys.stderr.write( 'WARNING: rewrite rules in: ' + G[x,'0'] + ' specify no graph equations: "' + eqns + '" -- will have no effect at ' + x + '\n' )

        ## if lexical rules produce equations, build appropriate graph...
        if '=' in eqns:
          ## translate eqn into graph...
#          print( ' -> ' + eqns )
#          G.dump()
          for eqn in eqns.split( '^' ):
            if VERBOSE: G.dump()
            if VERBOSE: print( ' applying eqn ' + eqn + '...' )
            lhs,rhs = eqn.split( '=' )
            xlhs = xrhs = G.result('S',x)
            if len(lhs)>0 and lhs[0]>='A' and lhs[0]<='Z': xlhs = x
            if len(rhs)>0 and rhs[0]>='A' and rhs[0]<='Z': xrhs = x
#            for num,lbl in enumerate(lhs[:-1]):
#              xlhs = G.result( lbl+'\'' if lbl.isdigit() and num==0 else lbl, xlhs )
            for lbl in lhs[:-1]:
              xlhs = G.result( lbl+'\'' if lbl.isdigit() and xlhs[-1] in 'sSH\'' else lbl, xlhs )
            if ':' in rhs: G.equate( rhs, lhs[-1]+'\'' if lhs[-1].isdigit() and xlhs[-1] in 'sSH\'' else lhs[-1], xlhs )
            else:
#              for num,lbl in enumerate(rhs):
#                xrhs = G.result( lbl+'\'' if lbl.isdigit() and num==0 else lbl, xrhs )
              for lbl in rhs:
                xrhs = G.result( lbl+'\'' if lbl.isdigit() and xrhs[-1] in 'sSH\'' else lbl, xrhs )
              G.equate( xrhs, lhs[-1]+'\'' if lhs[-1].isdigit() and xlhs[-1] in 'sSH\'' else lhs[-1], xlhs )
            if VERBOSE: G.dump()
        if VERBOSE:
          print( x, l, G[x,l] )
          G.dump( )
        '''
#        while( '-x' in s ):
#          s1           = re.sub( '^.((?:(?!-x).)*)-x.%:(\\S*)%(\\S*)\|(\\S*)%(\\S*):([^% ]*)%([^-: ]*)([^: ]*):\\2(\\S*)\\3', '\\4\\1\\5\\8:\\6\\9\\7', s )
#          if s1==s: s1 = re.sub( '^.((?:(?!-x).)*)-x.%(\\S*)\|([^% ]*)%([^-: ]*)([^: ]*):(\\S*)\\2', '\\3\\1\\5:\\6\\4', s )
#          if s1==s: s1 = re.sub( '-x', '', s )
#          s = s1
        ## if lexical rules produce non-equations...
        else:
          s = re.sub( 'BNOM-aD-bO', 'B-aN-bN', s )
          s = re.sub( 'BNOM-aD', 'B-aN', s )
          s = re.sub( 'BNOM', 'B-aN', s )
          ## place pred in ##e or ##r node, depending on category...
          if s[0]=='N' and not s.startswith('N-b{N-aD}') and not s.startswith('N-b{V-g{R') and not s.startswith('N-b{I-aN-g{R'):
            G.equate( s, '0', x+'e' )
            if (x+'s','h') in G: G.equate( G.result('h',x+'s'), 'h', x+'e' )      ## inherit possible/hypothetical world from s node.
            G.equate( G.result('r',G.result('S',x)), '1', x+'e' )
          else: G.equate( s, '0', G.result('r',G.result('S',x)) )
          ## coindex subject of raising construction with subject of direct object...
          if re.match( '^.-aN-b{.-aN}:', s ) != None:
            G.equate( G.result('1\'',G.result('S',x)), '1\'', G.result('2\'',G.result('S',x)) )
#          G.equate( G.result('1\'',G.result('2\'',G.result('S',x))), '1\'', G.result('S',x) )
        '''
    ## for each word...
    for x,l in sorted(G):
      if l=='X':
        ## rename 'S' node again, in case it changed in above raising attachments...
        if (x,    'S') in G: G.rename( x+'s', G[x    ,'S'] )
        if (x+'s','r') in G: G.rename( x+'r', G[x+'s','r'] )
    '''
    ## for each syntactic dependency...
    for x,l in sorted(G):
      if l[-1]=='\'':
        if VERBOSE:
          sys.stderr.write( 'Creating semantic dependencies for ' + x + ',' + l + '...\n' )
        try:
          ## add semantic dependencies...
          if   (x,'r') in G and (G[x,'r'],'0') in G:                                         G.equate( G[x,l], l[:-1], G.result('r',x) )        ## predicate
          elif (x,'e') in G and (G[x,'e'],'r') in G and (G[G[x,'e'],'r'],'0') in G:          G.equate( G[x,l], l[:-1], x )                      ## extraction of predicate
          elif (x,'e') in G and (G[x,'e'],'r') in G and (G[G[x,'e'],'r'][:-1]+'e','0') in G: G.equate( G[x,l], str(int(l[:-1])+1), x )          ## extraction of nominal
          elif (x[:-1]+'e','0') in G:                                                        G.equate( G[x,l], str(int(l[:-1])+1), x[:-1]+'e' ) ## nominal
#BAD;NON-PRED      elif (x,'e') in G and (G[x,'e'],'r') in G and (G[G[x,'e'],'r'][:-1]+'e','0') in G: G.equate( G[x,l], str(int(l[:-1])+1), x )           ## extraction of nominal
          if VERBOSE:
            sys.stderr.write( str(G) + '\n' )
        except KeyError as e:
          sys.stderr.write( 'KeyError ' + str(e) + ' in ' + str(t) + '\n' )
    '''

################################################################################

#class SemCueGraph( cuegraph.CueGraph ):
class SemCueGraph( StoreStateCueGraph ):

  def __init__( H, t=None ):
    if t is not None:
      G = StoreStateCueGraph( t )
      for x,l in sorted( G.keys() ):
        if l!='A' and l!='B' and l!='S' and l!='X' and l not in 'ABCDEFGHIJKLMNOPQRSTUVWWXYZ' and (l!='0' or x[-1] in 'ersABCDEFGHIJKLMNOPQRSTUVWXYZ') and l[-1]!='\'':
          H[x,l] = G[x,l]

#  def add( H, t, sentnumprefix ):
#    H.add( t, sentnumprefix )
##    G = StoreStateCueGraph( t, sentnumprefix )
##    for x,l in sorted( G.keys() ):
##      if l not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' and (l!='0' or x[-1] in 'erCDEFGHIJKLMNOPQRSTUVWXYZ') and l[-1]!='\'':
##        H[x,l] = G[x,l]

  def finalize( G ):
    for x,l in sorted( G.keys() ):
      if (l in 'mnsuuw' or l == 'tt') and ( (G[x,l][0:4],'S') not in G or (G[G[x,l][0:4],'S'],'r') not in G ):
        sys.stderr.write( 'WARNING: destination ' + G[x,l] + ' of dependency ' + l + ' in ' + x + ' not complete referential state in graph!\n' )
    for x,l in sorted( G.keys() ):
      if l in 'ABCDEFGHIJKLMNOPQRSTUVWWXYZ' or l[-1]=='\'' or l=='0' and x[-1] not in 'ersCDEFGHIJKLMNOPQRSTUVWXYZ':
        del G[x,l]

      '''
      if l in 's' and (G[x,l],'r') not in G:
        del G[x,l]   ## remove spurious scopes that result from coindexation
        sys.stderr.write( 'removing ' + x + ',' + l + ' because not real\n' )
      '''
    '''
    ## inherit all scopes...
    active = True
    while active:
      active = False
      for (x,l),y in G.items():
        if l=='r' and (y,'r') in G: continue  ## skip redundant predicative nusco
        if l in 'abcdefghijklmnopqruvxyz' and (y,'s') in G:  ## no s or t or w
          #if (x,'s') in G:  sys.stderr.write( 'WARNING: ' + x + ' has departing scope and inherits departing scope from ' + y + '\n' )
          if (x,'s') not in G:
            G[x,'s'] = G[y,'s']
            active = True
    '''

################################################################################

def last_inh( z, G ):
  if (z,'r') in G and (z,'e') in G: sys.stderr.write( 'ERROR MULTIPLE INHERITANCES: ' + z + ' in ' + str(G) + '\n' )
#' '.join( [ x+','+l+','+G[x,l] for x,l in sorted(G) ] ) )
  if (z,'r') in G: return last_inh( G[z,'r'], G )
  if (z,'e') in G: return last_inh( G[z,'e'], G )
  if (z,'h') in G: return last_inh( G[z,'h'], G )
  return z


class SimpleCueGraph( cuegraph.CueGraph ):

  def __init__( H, G ):
    for x,l in G:
      if '0'<=l and l<='9':
#        print( 'trying to add '+x+','+l+','+G[x,l] +' as ' + last_inh(x,G) + ',' + l + ',' + last_inh( G[x,l], G ) )
        H.equate( last_inh(G[x,l],G), l, last_inh(x,G) )
#        H[ last_inh(x,G), l ] = last_inh( G[x,l], G )
#        print( 'adding '+x+','+l+','+G[x,l] +' to ' + last_inh(x,G) + ',' + l + ',' + last_inh( G[x,l], G ) )
    for x,l in G:
      if l=='0':
        H.rename( x, last_inh(x,G) )



