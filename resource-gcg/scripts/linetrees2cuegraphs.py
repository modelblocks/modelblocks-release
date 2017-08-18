import sys, os, re, collections
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
import tree

VERBOSE = False

for a in sys.argv:
  if a=='-d': VERBOSE = True

################################################################################

class cuegraph( dict ):

  def rename( G, xNew, xOld ):
    for z,l in G.keys():
      if G[z,l] == xOld: G[z,l] = xNew       ## replace old destination with new
      if z == xOld:                          ## replace old source with new
        if (xNew,l) not in G:
          G[xNew,l] = G[xOld,l]
          del G[xOld,l]
    for z,l in G.keys():
      if z == xOld and (xNew,l) in G and G[xNew,l]!=G[xOld,l]: G.rename( G[xNew,l], G[xOld,l] )

  def result( G, l, x ):                     ## (f_l x)
    if (x,l) not in G:  G[x,l] = x+l         ## if dest is new, name it after source and label
    return G[x,l]

  def equate( G, y, l, x ):                  ## y = (f_l x)
    if (x,l) in G: G.rename( y, G[x,l] )     ## if source and label exist, rename
    else:          G[x,l] = y                ## otherwise, add to dict


################################################################################

def dump( G ):
  sys.stdout.write( 'G.a=' + G.a + ' G.b=' + G.b )
  for x,l in sorted( G ):
    sys.stdout.write( ' ' + x + ',' + l + ',' + G[x,l] )
  print( '' )


################################################################################

def deps( s, ops='abcdghirv' ):
  lst = [ ]
  d,h = 0,0
  for i in range( len(s) ):
    if s[i]=='{': d+=1
    if s[i]=='}': d-=1
    if d==0 and s[i]=='-' and h+1<len(s) and s[h+1] in ops: lst += [ s[h:i] ]
    if d==0 and s[i]=='-': h = i
  if h+1<len(s) and s[h+1] in ops: lst += [ s[h:] ]
  return lst


def lastdep( s ):
  d = deps( s )
  return d[-1] if len(d)>0 else ''


def firstnolo( s ):
  d = deps( s, 'ghirv' )
  return d[0] if len(d)>0 else ''


################################################################################

def relabel( t ):
#  print( t.c, lastdep(t.c) )
  p = ''

  ## for binary branches...
  if len(t.ch)==2:
    ## adjust tags...
    if lastdep(t.ch[1].c).startswith('-g') and '-lN' in t.ch[0].c:                                         t.ch[0].c = re.sub( '-lN', '-lG', t.ch[0].c )   ## G
    if lastdep(t.ch[0].c).startswith('-r') and '-lN' in t.ch[0].c:                                         t.ch[0].c = re.sub( '-lN', '-lR', t.ch[0].c )   ## Ra (off-spec)
    if lastdep(t.ch[1].c).startswith('-r') and '-lN' in t.ch[1].c:                                         t.ch[1].c = re.sub( '-lN', '-lR', t.ch[1].c )   ## R
    if lastdep(t.ch[0].c).startswith('-h') and '-lN' in t.ch[1].c:                                         t.ch[1].c = re.sub( '-lN', '-lH', t.ch[1].c )   ## H
    if re.match('-b{.*-[ghirv].*}',lastdep(t.ch[0].c))!=None and '-lA' in t.ch[1].c:                       t.ch[1].c = re.sub( '-lA', '-lI', t.ch[1].c )   ## I
    if '-lA' in t.ch[1].c and re.match( 'C-bV|E-bB|F-bI|O-bN|.-aN-b{.-aN}|N-b{N-aD}', t.ch[0].c ) != None: t.ch[1].c = re.sub( '-lA', '-lU', t.ch[1].c )   ## U
    if '-lA' in t.ch[0].c and t.ch[1].c=='D-aN':                                                           t.ch[0].c = re.sub( '-lA', '-lU', t.ch[0].c )
    ## fix gcg reannotation hacks ('said Kim' inversion, ':' as X-cX-dX, ':' as A-aN-bN)...
    if   '-lA' in t.ch[1].c and len( deps(t.ch[0].c,'b') )==0 and t.ch[0].c==':': t.ch[0] = tree.Tree( 'A-aN-bN', [ t.ch[0] ] ) 
    elif '-lA' in t.ch[1].c and len( deps(t.ch[0].c,'b') )==0: t.ch[0] = tree.Tree( re.sub('(V)-a('+deps(t.ch[0].c,'a')[-1][2:]+')','\\1-b\\2',t.ch[0].c), [ t.ch[0] ] )
    if   '-lC' in t.ch[1].c and len( deps(t.ch[0].c,'d') )==0: t.ch[0] = tree.Tree( 'X-cX-dX', [ t.ch[0] ] )
    ## calc parent given child types and ops...
    lcpsi = ''.join( deps(t.ch[0].c,'ghirv') )
    rcpsi = ''.join( deps(t.ch[1].c,'ghirv') )
    p = t.c
    if '-lA' in t.ch[0].c or '-lU' in t.ch[0].c:  p = re.findall('^[^-]*',t.ch[1].c)[0] + ''.join(deps(t.ch[1].c,'abcd')[:-1]) + lcpsi + rcpsi  ## Aa,Ua
    if '-lA' in t.ch[1].c or '-lU' in t.ch[1].c:  p = re.findall('^[^-]*',t.ch[0].c)[0] + ''.join(deps(t.ch[0].c,'abcd')[:-1]) + lcpsi + rcpsi  ## Ab,Ub
    if '-lC' in t.ch[0].c:                        p = re.findall('^[^-]*',t.ch[1].c)[0] + ''.join(deps(t.ch[1].c,'abcd')[:-1]) + lcpsi + rcpsi  ## Ca,Cb
    if '-lC' in t.ch[1].c:                        p = re.findall('^[^-]*',t.ch[0].c)[0] + ''.join(deps(t.ch[0].c,'abcd')[:-1]) + lcpsi + rcpsi  ## Cc
    if '-lM' in t.ch[0].c:                        p = re.findall('^[^-]*',t.ch[1].c)[0] + ''.join(deps(t.ch[1].c,'abcd')) + lcpsi + rcpsi       ## Ma
    if '-lM' in t.ch[1].c:                        p = re.findall('^[^-]*',t.ch[0].c)[0] + ''.join(deps(t.ch[0].c,'abcd')) + lcpsi + rcpsi       ## Mb
    if '-lG' in t.ch[0].c:                        p = re.sub( '(.*)'+deps(t.ch[1].c,'g')[-1], '\\1', t.ch[1].c, 1 ) + lcpsi                     ## G
    if '-lH' in t.ch[1].c:                        p = re.sub( '(.*)'+deps(t.ch[0].c,'h')[-1], '\\1', t.ch[0].c, 1 ) + rcpsi                     ## H
    if '-lI' in t.ch[1].c:                        p = re.findall('^[^-]*',t.ch[0].c)[0] + ''.join(deps(t.ch[0].c,'abcd')[:-1]) + lcpsi + ''.join( deps(t.ch[1].c,'ghirv')[:-1] )  ## I
    if '-lR' in t.ch[0].c:                        p = t.ch[1].c                                                                                 ## Ra (off-spec)
    if '-lR' in t.ch[1].c:                        p = t.ch[0].c                                                                                 ## R
    ## add calculated parent of children as unary branch if different from t.c...
    if re.sub('-[lx][^-]*','',p) != re.sub('-[lx][^-]*','',t.c):
      if VERBOSE: print( 'T rule: ' + t.c + ' -> ' + p )
      t.ch = [ tree.Tree( p, t.ch ) ]

  ## for (new or old) unary branches...
  if len(t.ch)==1:
    locs  = deps(t.c,'abcd')
    nolos = deps(t.c,'ghirv')
    chlocs = deps(t.ch[0].c,'abcd')
    if   len(locs)>1 and len(chlocs)>1 and locs!=chlocs and locs[0][2:]==chlocs[1][2:] and locs[1][2:]==chlocs[0][2:]: t.ch[0].c += '-lQ'       ## Q
    elif t.c.startswith('A-a') and len(t.ch)==1 and t.ch[0].c.startswith('L-aN'): t.ch = [ tree.Tree( 'L-aN-vN-lV', t.ch ) ]                    ## V
    elif t.c.startswith('A-a') and len(t.ch)==1 and t.ch[0].c.startswith('N'): t.ch[0].c += '-lZ'                                               ## Za
    elif t.c.startswith('R-a') and len(t.ch)==1 and t.ch[0].c.startswith('N'): t.ch[0].c += '-lZ'                                               ## Zb
    elif len(nolos)>0 and nolos[0][2]!='{' and len(deps(t.c))==len(deps(t.ch[0].c)) and len(chlocs)>len(locs) and nolos[0]!=chlocs[len(locs)]:  ## Ea
      t.ch = [ tree.Tree( re.sub(nolos[0],chlocs[len(locs)],re.sub('-l.','',t.c),1)+'-lE', t.ch ) ]
    elif len(nolos)>0 and nolos[0][2]=='{': t.ch = [ tree.Tree( re.sub(nolos[0],'',re.sub('-l.','',t.c),1)+'-lE', t.ch ) ]                      ## Eb

  for st in t.ch:
    relabel( st )


################################################################################

class storestate( cuegraph ):

  def getArity( G, cat ):
    while '{' in cat:
      cat = re.sub('\{[^\{\}]*\}','X',cat)
    return len(re.findall('-[ab]',cat))


  def findNolos( G, nolos, n ):
    while True:
      if (n,'0') in G and G[n,'0']==nolos[-1]: nolos.pop()
      if nolos == []: return n
      if   (n,'A') in G: n = G[n,'A']  ## advance n if A is next on store
      elif (n,'B') in G: n = G[n,'B']  ## advance n if B is next on store
      else: return ''

  def findNolo( G, sN, n ):
    while True:
      if (n,'0') in G and G[n,'0']==sN: return n
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
      dump( G )
      print( 'f', f, sD, w, id )

    if f==0:
      G.rename( id+'r', G.result('r',G.result('s',G.b)) )
      G.rename( id+'s', G.result('s',G.b) )
      G.rename( id, G.b )
      G.b = id
      G.equate( w, 'w', G.b )
      G.a = G.result( 'A', G.b )
      while (G.a,'A') in G: G.a = G[G.a,'A']      ## traverse all non-local dependencies on A
    if f==1:
      G.a = id
      G.equate( sD,  '0', G.a )
      G.equate( w,   'w', G.a )
      ## add all nonlocal dependencies with no nolo on store...
      b = G.a
      for sN in reversed( deps(sD) ):
        if sN[1] in 'ghirv' and not G.findNolo( sN, G.b ):
          b = G.result( 'B', b )
          G.equate( sN, '0', b )
      G.equate( G.b, 'B', b )
      G.equate( id+'r', 'r', G.result('s',G.a) )

    ## attach rel pro antecedent...
    for i,psi in enumerate( deps(sD) ):
      if psi[1] == 'r':
        G.equate( G.result('s',G.findNolo(psi,id)), 'e', G.result('r',G.result('s',id)) )    ## restrictive relpro


  def updateUna( G, s, sC, sD, id ):

    if VERBOSE:
      dump( G )
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
      G.equate( G.result('s',dLower), 'e', G.result('r',G.result('s',dUpper)) )
    elif '-lQ' in sD:                             ## Q
      G.equate( G.result(l,d), l, d+'u' )
      G.equate( G.result('1\'',G.result('s',d)), '2\'', G.result('s',d+'u') )  ## switch 1' & 2' arguments (same process top-down as bottom-up)
      G.equate( G.result('2\'',G.result('s',d)), '1\'', G.result('s',d+'u') )
      G.equate( G.result('s',dLower), 'e', G.result('s',dUpper) )
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
      nolos = deps( sC, 'ghirv' )
      sN = nolos[0]
      n = G.findNolos( nolos, d )
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
      hideps = deps( sC )
      lodeps = deps( sD )
      for i in range( len(hideps) ):
        if i<len(lodeps) and hideps[i][1]!=lodeps[i][1] and hideps[i][1] in 'ghirv' and lodeps[i][1] in 'ghirv':
          if s==0: G[ G.findNolo(lodeps[i],d), '0' ] = hideps[i]    ## bottom up on left child
          else:    G[ G.findNolo(hideps[i],d), '0' ] = lodeps[i]    ## top down on right child
      return      ## don't add 'u' node
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
      dump( G )
      print( 'j', j, sC, sD, sE, id )

    G.b = id + 'b'
    G.equate( sE, '0', G.b )
    if j==0:
      c = id + 'a'
      G.equate( c, 'A', G.result('A',G.b) if '-lG' in sD or '-lI' in sE or '-lR' in sE else G.b )
      G.equate( sC, '0', c )
      ## add all nonlocal dependencies with no nolo on store...
      b = c
      for sN in reversed( deps(sC) ):
        if sN[1] in 'ghirv' and not G.findNolo( sN, G.a ):
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
      G.equate( G.result('r',G.result('s',d)), 'r', G.result('s',c) )   ## rename 'r' node as well as 's'
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
      for i in range( 1, len(deps(sC,'ab'))+1 ): #G.getArity(sC)+1 ):
        G.equate( G.result(str(i)+'\'',G.result('s',c)), str(i)+'\'', G.result('s',d) )
    elif '-lC' in sE:                               ## Cc
      G.equate( G.result('s',d), 's', c )
      G.equate( G.result('r',G.result('s',d)), 'c', G.result('r',G.result('s',e)) )
      G.equate( G.result('s',d), 'c', G.result('s',e) )
      for i in range( 1, len(deps(sC,'ab'))+1 ):  #G.getArity(sC)+1 ):
        G.equate( G.result(str(i)+'\'',G.result('s',c)), str(i)+'\'', G.result('s',e) )
    elif '-lG' in sD:                               ## G
      G.equate( G.result('s',c), 's', e )
      G.equate( G.result('s',d), 's', G.result('A',e) )
      G.equate( lastdep(sE), '0', G.result('A',e) )
    elif '-lH' in sE:                               ## H
      G.equate( G.result('s',d), 's', c )
      n = G.findNolo( lastdep(sD), d )
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
      G.equate( lastdep(sE), '0', G.result('A',e) )
    elif '-lR' in sD:                               ## Ra (off-spec)
      G.equate( G.result('s',c), 's', e )
      G.equate( G.result('s',e), 's', G.result('B',d) )
      G.equate( lastdep(sD), '0', G.result('B',d) )
    elif '-lR' in sE:                               ## R
      G.equate( G.result('s',d), 's', c )
      G.equate( G.result('s',d), 's', G.result('A',e) )
      G.equate( lastdep(sE), '0', G.result('A',e) )
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


################################################################################

for line in sys.stdin:

  line = re.sub( '-iN-gN', '-gN-iN', line )  ## script puts nonlocal deps in wrong order; higher bound nolos should be buried deeper
  line = re.sub( '-rN-vN', '-vN-rN', line )

  tr = tree.Tree( )
  tr.read( line )
  relabel( tr )
  if VERBOSE: print( tr )

  G = storestate( )
  G.a = 'top'
  G.b = 'bot'
  G.convert( tr )

  ## for each word...
  for x,l in sorted(G):
    ## add predicates by applying morph rules...
    if l=='w':
      if VERBOSE:
        dump( G )
        print( x, l, G[x,l] )
      s = re.sub('-l.','',G[x,'0']) + ':' + G[x,'w'].lower()
      while( '-x' in s ):
        s1 = re.sub( '^.(\\S*?)-x.%:(\\S*)%(\\S*)\|(\\S*)%(\\S*):([^% ]*)%([^-: ]*)([^: ]*):\\2(\\S*)\\3', '\\4\\1\\5\\8:\\6\\9\\7', s )
        if s1==s: s1 = re.sub( '^.(\\S*?)-x.%(\\S*)\|([^% ]*)%([^-: ]*)([^: ]*):(\\S*)\\2', '\\3\\1\\5:\\6\\4', s )
        if s1==s: s1 = re.sub( '-x', '', s )
        s = s1
      ## place pred in ##e or ##r node, depending on category...
      if s[0]=='N' and not s.startswith('N-b{N-aD}') and not s.startswith('N-b{V-g{R') and not s.startswith('N-b{I-aN-g{R'):
        G.equate( s, '0', x+'e' )
        G.equate( G.result('r',G.result('s',x)), '1', x+'e' )
      else: G.equate( s, '0', G.result('r',G.result('s',x)) )
      ## coindex subject of raising construction with subject of direct object...
      if re.match( '^.-aN-b{.-aN}:', s ) != None:
        G.equate( G.result('1\'',G.result('2\'',G.result('s',x))), '1\'', G.result('s',x) )
  ## for each syntactic dependency...
  for x,l in sorted(G):
    if l[-1]=='\'':
      ## add semantic dependencies...
      if   (x,'r') in G and (G[x,'r'],'0') in G:                                   G.equate( G[x,l], l[:-1], G.result('r',x) )         ## predicate
      elif (x,'e') in G and (G[x,'e'],  'r') in G and (G[G[x,'e'],'r'],'0') in G:  G.equate( G[x,l], l[:-1], x )                       ## extraction of predicate
      elif (x[:-1]+'e','0') in G:                                                  G.equate( G[x,l], str(int(l[:-1])+1), x[:-1]+'e' )  ## nominal

  for x,l in sorted(G.keys()):
    if l!='A' and l!='B' and l!='s' and l!='w' and (l!='0' or x[-1] in 'er') and l[-1]!='\'':
#':' in G[x,l]) and l[-1]!='\'':
      sys.stdout.write( ' ' + x + ',' + l + ',' + G[x,l] )
  sys.stdout.write( '\n' )



