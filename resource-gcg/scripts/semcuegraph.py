import sys, os, re, collections
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
import gcgtree
import cuegraph

VERBOSE = False


################################################################################

class StoreStateCueGraph( cuegraph.CueGraph ):

  def dump( G ):
    print( 'G.a=' + G.a + ' G.b=' + G.b + ' ' + str(G) )


  def getArity( G, cat ):
    while '{' in cat:
      cat = re.sub('\{[^\{\}]*\}','X',cat)
    return len(re.findall('-[ab]',cat))


  def findNolos( G, nolos, n ):
    while True:
      if (n,'0') in G and G[n,'0']==nolos[-1]: nolos.pop()
      if (n,'0') in G and not G[n,'0'].startswith('-') and len(nolos)>0 and nolos[-1] not in G[n,'0']: return ''
      if nolos == []: return n
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
      ## rename 's' and 'r' nodes...
      G.rename( id+'r', G.result('r',G.result('s',G.b)) )
      G.rename( id+'s', G.result('s',G.b) )
      G.rename( id,     G.b )
      G.b = id
      G[G.b,'0'] = sD
      G.equate( w, 'w', G.b )
      G.a = G.result( 'A', G.b )
      while (G.a,'A') in G: G.a = G[G.a,'A']      ## traverse all non-local dependencies on A
    if f==1:
      G.a = id
      G.equate( sD, '0', G.a )
      G.equate( w,  'w', G.a )
      ## add all nonlocal dependencies with no nolo on store...
      b = G.a
      for sN in reversed( gcgtree.deps(sD) ):
        if sN[1] in 'ghirv' and not G.findNolo( sN, G.b ):
          b = G.result( 'B', b )
          G.equate( sN, '0', b )
      G.equate( G.b, 'B', b )
      ## rename 'r' nodes...
      G.equate( id+'r', 'r', G.result('s',G.a) )

    ## attach rel pro / interrog pro antecedent...
    for i,psi in enumerate( gcgtree.deps(sD) ):
      if psi[1] in 'ir':
        G.equate( G.result('s',G.findNolo(psi,id)), 'e', G.result('r',G.result('s',id)) )    ## restrictive relpro


  def updateUna( G, s, sC, sD, id ):

    if VERBOSE:
      G.dump( )
      print( 's', s, sC, sD, id )

    n = ''
    l,d = ('B',G.a) if s==0 else ('A',G.b)
    dUpper,dLower = (G.a+'u',G.a) if s==0 else (G.b,G.b+'u')  ## bottom-up on left child, top-down on right child

    if '-lV' in sD:                               ## V
      sN = re.findall('-v(?:[^-{}]|{[^{}]*})',sD)[-1]
      n = G.findNolo( sN, d )
      if n=='':
        n = G.result( l, d+'u' )
        G.equate( sN, '0', n )
        G.equate( G.result(l,d), l, n )
      else: G.equate( G.result(l,d), l, d+'u' )
      G.equate( G.result('s',n), '1\'', G.result('s',dUpper) )
      G.equate( G.result('s',dUpper), 'e', G.result('r',G.result('s',dLower)) )
    elif '-lQ' in sD:                             ## Q
      G.equate( G.result(l,d), l, d+'u' )
      G.equate( G.result('1\'',G.result('s',d)), '2\'', G.result('s',d+'u') )  ## switch 1' & 2' arguments (same process top-down as bottom-up)
      G.equate( G.result('2\'',G.result('s',d)), '1\'', G.result('s',d+'u') )
      G.equate( G.result('s',dUpper), 'e', G.result('r',G.result('s',dLower)) )
    elif '-lZ' in sD and sC.startswith('A-aN-x'): ## Zc
      G.equate( G.result(l,d), l, d+'u' )
      G.equate( G.result('s',dLower),                 '2', G.result('r',G.result('s',dUpper)) )
      G.equate( G.result('1\'',G.result('s',dUpper)), '1', G.result('r',G.result('s',dUpper)) )
      G.equate( 'A-aN-bN:~',                          '0', G.result('r',G.result('s',dUpper)) )
    elif '-lZ' in sD and sC.startswith('A-a'):    ## Za
      G.equate( G.result(l,d), l, d+'u' )
      G.equate( G.result('r',G.result('r',G.result('s',dLower))), '1\'', G.result('s',dUpper) )
      G.equate( G.result('s',dUpper), 'h', G.result('s',dLower) )              ## hypothetical world inheritance -- should be implemented throughout
    elif '-lZ' in sD and sC.startswith('R-a'):    ## Zb
      G.equate( G.result(l,d), l, d+'u' )
      G.equate( G.result('s',dLower),                 '2', G.result('r',G.result('s',dUpper)) )
      G.equate( G.result('1\'',G.result('s',dUpper)), '1', G.result('r',G.result('s',dUpper)) )
      G.equate( 'A-aN-bN:during',                     '0', G.result('r',G.result('s',dUpper)) )
    elif '-lE' in sD:
      nolos = gcgtree.deps( sC, 'ghirv' )
      sN = nolos[0]
      n = G.findNolos( nolos, G[d,l] )
      if n=='':
        n = G.result( l, d+'u' )
        G.equate( sN, '0', n )
        G.equate( G.result(l,d), l, n )
      else: G.equate( G.result(l,d), l, d+'u' )
      if sN.endswith('-aN}'):                     ## Eb,Ed
        G.equate( G.result('r',G.result('s',dLower)), '1\'', id+'y' )
        G.equate( G.result('s',n), 'e', id+'y' )
      else:                                       ## Ea,Ec
        G.equate( G.result('s',n), 'e', G.result( str(G.getArity(sD))+'\'', G.result('s',dLower) ) )
      G.equate( G.result('s',d), 's', d+'u' )
    elif '-l' not in sD:                          ## T
      ## update category of every nonlocal sign on store that changes with type change...
      hideps = gcgtree.deps( sC )
      lodeps = gcgtree.deps( sD )
      a = d+'u'
      for i in range( len(hideps) ):
        if i<len(lodeps) and hideps[i][1]!=lodeps[i][1] and hideps[i][1] in 'ghirv' and lodeps[i][1] in 'ghirv':
          a = G.result( l, a )
          G.equate( lodeps[i], '0', a )   ## note below only for s==1
          G.equate( G.result('s',G.findNolo(hideps[i],d)), 's', a )
#          if s==0: G[ G.findNolo(lodeps[i],d), '0' ] = hideps[i]    ## bottom up on left child
#          else:    G[ G.findNolo(hideps[i],d), '0' ] = lodeps[i]    ## top down on right child
      if a == d+'u': return  ## don't add 'u' node
      G.equate( G[d,l], l, a )
    else: return  ## don't add 'u' node

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
      G.equate( c, 'A', G.result('A',G.b) if '-lG' in sD or '-lI' in sE or '-lR' in sE else G.b )
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
      G.equate( G.result('A',c), 'A', G.result('A',G.b) if '-lG' in sD or '-lI' in sE or '-lR' in sE else G.b )

    d,e = G.a,G.b
    if   '-lA' in sD:                               ## Aa
      G.equate( G.result('s',c), 's', e )
      G.equate( G.result('s',d), str(G.getArity(sE))+'\'', G.result('s',e) )
    elif '-lA' in sE:                               ## Ab
      G.equate( G.result('s',d), 's', c )
#      G.equate( G.result('r',G.result('s',d)), 'r', G.result('s',c) )   ## rename 'r' node as well as 's'
      G.equate( G.result('s',e), str(G.getArity(sD))+'\'', G.result('s',d) )
    elif '-lU' in sD:                               ## Ua
      G.equate( G.result('s',d), 's', c )
      G.equate( G.result('s',d), str(G.getArity(sE))+'\'', G.result('s',e) )
    elif '-lU' in sE:                               ## Ub
      G.equate( G.result('s',c), 's', e )
      G.equate( G.result('s',e), str(G.getArity(sD))+'\'', G.result('s',d) )
    elif '-lM' in sD:                               ## Ma
      G.equate( G.result('s',c), 's', e )
      G.equate( G.result('r',G.result('s',e)), '1\'', G.result('s',d) )
    elif '-lM' in sE:                               ## Mb
      G.equate( G.result('s',d), 's', c )
      G.equate( G.result('r',G.result('s',d)), '1\'', G.result('s',e) )
    elif '-lC' in sD:                               ## Ca,Cb
      G.equate( G.result('s',c), 's', e )
      G.equate( G.result('r',G.result('s',e)), 'c', G.result('r',G.result('s',d)) )
      G.equate( G.result('s',e), 'c', G.result('s',d) )
      for i in range( 1, len(gcgtree.deps(sC,'ab'))+1 ): #G.getArity(sC)+1 ):
        G.equate( G.result(str(i)+'\'',G.result('s',c)), str(i)+'\'', G.result('s',d) )
    elif '-lC' in sE:                               ## Cc
      G.equate( G.result('s',d), 's', c )
      G.equate( G.result('r',G.result('s',d)), 'c', G.result('r',G.result('s',e)) )
      G.equate( G.result('s',d), 'c', G.result('s',e) )
      for i in range( 1, len(gcgtree.deps(sC,'ab'))+1 ):  #G.getArity(sC)+1 ):
        G.equate( G.result(str(i)+'\'',G.result('s',c)), str(i)+'\'', G.result('s',e) )
    elif '-lG' in sD:                               ## G
      G.equate( G.result('s',c), 's', e )
      G.equate( G.result('s',d), 's', G.result('A',e) )
      G.equate( gcgtree.lastdep(sE), '0', G.result('A',e) )
    elif '-lH' in sE:                               ## H
      G.equate( G.result('s',d), 's', c )
      n = G.findNolo( gcgtree.lastdep(sD), d )
      if n!='': G.equate( G.result('s',e), 's', n )
      if n!='': G[n,'0']+='-closed'  ## close off nolo
    elif '-lI' in sE:                               ## I
      G.equate( G.result('s',d), 's', c )
      if sD.startswith('N-b{V-g') or sD.startswith('N-b{I-aN-g'):                                 ## nominal clause
        G.equate( G.result('s',d), 's', G.result('A',e) )
      elif '-b{I-aN-gN}' in sD:                                                                   ## tough construction
        G.equate( G.result('1\'',G.result('s',d)), 's', G.result('A',e) )
        G.equate( G.result('s',e), str(G.getArity(sD))+'\'', G.result('s',d) )
      else: G.equate( G.result('s',G.result('A',e)), str(G.getArity(sD))+'\'', G.result('s',d) )  ## embedded question
      G.equate( gcgtree.lastdep(sE), '0', G.result('A',e) )
    elif '-lR' in sD:                               ## Ra (off-spec)
      G.equate( G.result('s',c), 's', e )
      G.equate( G.result('s',e), 's', G.result('B',d) )
      G.equate( gcgtree.lastdep(sD), '0', G.result('B',d) )
    elif '-lR' in sE:                               ## R
      G.equate( G.result('s',d), 's', c )
      G.equate( G.result('s',d), 's', G.result('A',e) )
      G.equate( gcgtree.lastdep(sE), '0', G.result('A',e) )
    else:
      sys.stderr.write( 'WARNING: No analysis for annotated binary expansion ' + sC + ' -> ' + sD + ' ' + sE + '.\n' )


  def convert( G, t, s=0, i=0 ):

    ## terminal...
    if len(t.ch)==1 and len(t.ch[0].ch)==0:
      i += 1
      G.updateLex( 1-s, t.c, t.ch[0].c, ('0' if i<10 else '')+str(i) )

    ## nonterminal unary...
    elif len(t.ch)==1:
      if s==1: G.updateUna( s, t.c, t.ch[0].c, ('0' if i<10 else '')+str(i) )  ## update storestate preorder if right child (to build on G.b)
      i = G.convert( t.ch[0], s, i )
      if s==0: G.updateUna( s, t.c, t.ch[0].c, ('0' if i<10 else '')+str(i) )  ## update storestate postorder if left child (to build on G.a)

    ## nonterminal binary...
    elif len(t.ch)==2:
      i = G.convert( t.ch[0], 0, i )
      G.updateBin( s, t.c, t.ch[0].c, t.ch[1].c, ('0' if i<10 else '')+str(i) )
      i = G.convert( t.ch[1], 1, i )

    return i


  def __init__( G, t ):
    
#    gcgtree.relabel( t )
    if VERBOSE: print( t )

    G.a = 'top'
    G.b = 'bot'
    G.convert( t )

    ## for each word...
    for x,l in sorted(G):
      ## add predicates by applying morph rules...
      if l=='w':
        ## rename 's' and 'r' nodes...
        if (x,    's') in G: G.rename( x+'s', G[x,    's'] )
        if (x+'s','r') in G: G.rename( x+'r', G[x+'s','r'] )
        if VERBOSE:
          G.dump( )
          print( x, l, G[x,l] )
        ## obtain pred by applying morph rules to word token...
        s = re.sub('-l.','',G[x,'0']) + ':' + G[x,'w'].lower()
        while( '-x' in s ):
          s1           = re.sub( '^.((?:(?!-x).)*)-x.%:(\\S*)%(\\S*)\|(\\S*)%(\\S*):([^% ]*)%([^-: ]*)([^: ]*):\\2(\\S*)\\3', '\\4\\1\\5\\8:\\6\\9\\7', s )
          if s1==s: s1 = re.sub( '^.((?:(?!-x).)*)-x.%(\\S*)\|([^% ]*)%([^-: ]*)([^: ]*):(\\S*)\\2', '\\3\\1\\5:\\6\\4', s )
          if s1==s: s1 = re.sub( '-x', '', s )
          s = s1
        s = re.sub( 'BNOM-aD-bO', 'B-aN-bN', s )
        s = re.sub( 'BNOM-aD', 'B-aN', s )
        s = re.sub( 'BNOM', 'B-aN', s )
        ## place pred in ##e or ##r node, depending on category...
        if s[0]=='N' and not s.startswith('N-b{N-aD}') and not s.startswith('N-b{V-g{R') and not s.startswith('N-b{I-aN-g{R'):
          G.equate( s, '0', x+'e' )
          G.equate( G.result('r',G.result('s',x)), '1', x+'e' )
        else: G.equate( s, '0', G.result('r',G.result('s',x)) )
        ## coindex subject of raising construction with subject of direct object...
        if re.match( '^.-aN-b{.-aN}:', s ) != None:
          G.equate( G.result('1\'',G.result('s',x)), '1\'', G.result('2\'',G.result('s',x)) )
#          G.equate( G.result('1\'',G.result('2\'',G.result('s',x))), '1\'', G.result('s',x) )
    ## for each word...
    for x,l in sorted(G):
      if l=='w':
        ## rename 's' node again, in case it changed in above raising attachments...
        if (x,    's') in G: G.rename( x+'s', G[x    ,'s'] )
        if (x+'s','r') in G: G.rename( x+'r', G[x+'s','r'] )
    ## for each syntactic dependency...
    for x,l in sorted(G):
      if l[-1]=='\'':
        ## add semantic dependencies...
        if   (x,'r') in G and (G[x,'r'],'0') in G:                                 G.equate( G[x,l], l[:-1], G.result('r',x) )         ## predicate
        elif (x,'e') in G and (G[x,'e'],'r') in G and (G[G[x,'e'],'r'],'0') in G:  G.equate( G[x,l], l[:-1], x )                       ## extraction of predicate
        elif (x[:-1]+'e','0') in G:                                                G.equate( G[x,l], str(int(l[:-1])+1), x[:-1]+'e' )  ## nominal
#BAD;NON-PRED      elif (x,'e') in G and (G[x,'e'],'r') in G and (G[G[x,'e'],'r'][:-1]+'e','0') in G: G.equate( G[x,l], str(int(l[:-1])+1), x )           ## extraction of nominal


################################################################################

class SemCueGraph( cuegraph.CueGraph ):

  def __init__( H, t ):
    G = StoreStateCueGraph( t )
    for x,l in sorted(G.keys()):
      if l!='A' and l!='B' and l!='s' and l!='w' and (l!='0' or x[-1] in 'er') and l[-1]!='\'':
        H[x,l] = G[x,l]


################################################################################

def last_inh( z, G ):
  if (z,'r') in G and (z,'e') in G: print( 'ERROR MULTIPLE INHERITANCES', z, str(G) )
#' '.join( [ x+','+l+','+G[x,l] for x,l in sorted(G) ] ) )
  if (z,'r') in G: return last_inh( G[z,'r'], G )
  if (z,'e') in G: return last_inh( G[z,'e'], G )
  return z


class SimpleCueGraph( cuegraph.CueGraph ):

  def __init__( H, G ):
    for x,l in G:
      if '0'<=l and l<='9':
        H[ last_inh(x,G), l ] = last_inh( G[x,l], G )




