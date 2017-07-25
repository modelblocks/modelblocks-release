import sys, os, re, collections
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
import tree

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
      if z == xOld and (xNew,l) in G: G.rename( G[xNew,l], G[xOld,l] )

  def result( G, l, x ):                     ## (f_l x)
    if (x,l) not in G:  G[x,l] = x+l         ## if dest is new, name it after source and label
    return G[x,l]

  def equate( G, y, l, x ):                  ## y = (f_l x)
    if (x,l) in G: G.rename( y, G[x,l] )     ## if source and label exist, rename
    else:          G[x,l] = y                ## otherwise, add to dict


################################################################################

def dump( G ):
  for x,l in sorted(G):
    sys.stdout.write(' '+x+','+l+','+G[x,l])
  print( '' )


################################################################################

class storestate( cuegraph ):

  def getArity( G, cat ):
#    cat = G.result('0',a)
    while '{' in cat:
      cat = re.sub('\{[^\{\}]*\}','X',cat)
    return len(re.findall('-[ab]',cat))


  def findNolo( G, sN ):
    n = G.a
    while True:
      if (n,'0') in G and G[n,'0']==sN: return n
      if   (n,'A') in G: n = G[n,'A']
      elif (n,'B') in G: n = G[n,'B']
      else: return ''


  def updateLex( G, f, sD, w, id ):
#    print( 'f', f, 'at', id )
    if f==0:
      G.rename( id+'sr', G.result('r',G.result('s',G.b)) )
      G.rename( id+'s', G.result('s',G.b) )
      G.rename( id, G.b )
      G.b = id
      G.equate( w, 'w', G.b )
      G.a = G.result( 'A', G.b )
      while (G.a,'A') in G: G.a = G[G.a,'A']    ## if there are non-local dependencies on A
    if f==1:
      G.a = id
      G.equate( sD,  '0', G.a )
      G.equate( w,   'w', G.a )
      G.equate( G.b, 'B', G.a )
      G.result( 'r', G.result('s',G.a) )
#    dump( G )


  def updateUna( G, s, sC, sD, id ):

    dump( G )
    print( 's', s, 'at', id, sC, sD )

    l,d = ('B',G.a) if s==0 else ('A',G.b)
    dUpper,dLower = (G.a+'u',G.a) if s==0 else (G.b,G.b+'u')  ## bottom-up on left child, top-down on right child

    if '-lV' in sD:               ## V
      sN = re.findall('-v(?:[^-{}]|{[^{}]*})',sD)[-1]
#      n = G[d,l]
#      del G[d,l]
#      G.equate( n,  l,   G.result(l,d) )  ## insert nonlocal above complete sign
#      G.equate( sN, '0', G.result(l,d) )  ## set label of nonlocal
#      G.rename( d+'old', d )
      n = G.findNolo( sN )
      print( n )
      if n=='':
        n = G.result( l, d+'u' )
        G.equate( sN, '0', n )
        G.equate( G.result(l,d), l, n )
      else: G.equate( G.result(l,d), l, d+'u' )
      G.equate( G.result('s',n), '1\'', G.result('s',dUpper) )
      G.equate( G.result('s',dLower), 'e', G.result('s',dUpper) )
#      G.equate( G.result('s',n),      '1\'', G.result('s',dUpper) )
    elif '-lQ' in sD:               ## Q
#      G.rename( d+'old', d )      ## switch 1' & 2' arguments (same process top-down as bottom-up)
      G.equate( G.result(l,d), l, d+'u' )
      G.equate( G.result('1\'',G.result('s',d)), '2\'', G.result('s',d+'u') )
      G.equate( G.result('2\'',G.result('s',d)), '1\'', G.result('s',d+'u') )
      G.equate( G.result('s',dLower), 'e', G.result('s',dUpper) )
    elif '-lZ' in sD and sC.startswith('A-aN'):   ## Za
#      G.rename( d+'old', d )
      G.equate( G.result(l,d), l, d+'u' )
      G.equate( G.result('r',G.result('s',dLower)), '1\'', G.result('s',dUpper) )
    elif '-lZ' in sD and sC.startswith('R-aN'):   ## Zb
#      G.rename( d+'old', d )
      G.equate( G.result(l,d), l, d+'u' )
      G.equate( G.result('s',dLower),                 '2', G.result('r',G.result('s',dUpper)) )
      G.equate( G.result('1\'',G.result('s',dUpper)), '1', G.result('r',G.result('s',dUpper)) )
    elif '-lE' in sD:
      sN = re.findall('-[ghirv](?:[^-]*|{[^{}]*})',sC)[0]
      print( sN )
      n = G.findNolo( sN )
      if n=='':
        n = G.result( l, d+'u' )
        G.equate( sN, '0', n )
        G.equate( G.result(l,d), l, n )
#        ab = G[d,l]
#        del G[d,l]
#        n = G.result( l, d )
#        G.equate( ab, l,   n )    ## insert nonlocal above complete sign
#        G.equate( sN, '0', n )    ## set label of nonlocal
      else: G.equate( G.result(l,d), l, d+'u' )
      if sN.endswith('-aN}'):      ## Eb,Ed
        G.equate( G.result('r',G.result('s',dLower)), '1\'', id+'y' )
        G.equate( G.result('s',n), 'e', id+'y' )
      else:                       ## Ea,Ec
        G.equate( G.result('s',n), 'e', G.result( str(G.getArity(sD))+'\'', G.result('s',dLower) ) )
    else: return

    if s==0:
      G.a = d+'u'
      G.equate( sC, '0', d+'u' )
    if s==1:
      G.b = d+'u'
      G.equate( sD, '0', d+'u' )
#    dump( G )


  def updateBin( G, j, sC, sD, sE, id ):

    dump( G )
    print( 'j', j, 'at', id, sC, sD, sE )

    G.b = id + 'b'
    G.equate( sE, '0', G.b )
    if j==0:
      c = id + 'a'
      G.equate( c, 'A', G.result('A',G.b) if '-lG' in sD or '-lI' in sE or '-lR' in sE else G.b )
      G.equate( sC, '0', c )
      G.equate( G.result('B',G.a), 'B', c )
    if j==1:
      c = G.result( 'B', G.a )
      while (c,'B') in G: c = G[c,'B']          ## if there are non-local dependencies on B
      G.equate( G.result('A',c), 'A', G.result('A',G.b) if '-lG' in sD or '-lI' in sE or '-lR' in sE else G.b )

    d,e = G.a,G.b
    if '-lA' in sD:   ## Aa
      G.equate( G.result('s',c), 's', e )
      G.equate( G.result('s',d), str(G.getArity(sE))+'\'', G.result('s',e) )
    if '-lA' in sE:   ## Ab
      G.equate( G.result('s',d), 's', c )
      G.equate( G.result('r',G.result('s',d)), 'r', G.result('s',c) )   ## rename 'r' node as well as 's'
      G.equate( G.result('s',e), str(G.getArity(sD))+'\'', G.result('s',d) )
    if '-lM' in sD:   ## Ma
      G.equate( G.result('s',c), 's', e )
      G.equate( G.result('r',G.result('s',e)), '1\'', G.result('s',d) )
    if '-lM' in sE:   ## Mb
      G.equate( G.result('s',d), 's', c )
      G.equate( G.result('r',G.result('s',d)), '1\'', G.result('s',e) )
    if '-lC' in sD:   ## Ca,Cb
      G.equate( G.result('s',c), 's', e )
      G.equate( G.result('r',G.result('s',e)), 'c', G.result('r',G.result('s',d)) )
      G.equate( G.result('s',e), 'c', G.result('s',d) )
      for i in range( 1, G.getArity(sC)+1 ):
        G.equate( G.result(str(i)+'\'',G.result('s',d)), str(i)+'\'', G.result('s',e) )
    if '-lC' in sE:   ## Cc
      G.equate( G.result('s',d), 's', c )
      G.equate( G.result('r',G.result('s',d)), 'c', G.result('r',G.result('s',e)) )
      G.equate( G.result('s',d), 'c', G.result('s',e) )
      for i in range( 1, G.getArity(sC)+1 ):
        G.equate( G.result(str(i)+'\'',G.result('s',d)), str(i)+'\'', G.result('s',e) )
    if '-lG' in sD:    ## G
      G.equate( G.result('s',c), 's', e )
      G.equate( G.result('s',d), 's', G.result('A',e) )
      G.equate( lastdep(sE), '0', G.result('A',e) )
    if '-lH' in sE:    ## H
      G.equate( G.result('s',d), 's', c )
      n = G.findNolo( lastdep(sD) )
      if n!='': G.equate( G.result('s',e), 's', n )
    if '-lI' in sE:    ## I
      G.equate( G.result('s',d), 's', c )
      G.equate( G.result('s',G.result('A',e)), str(G.getArity(sD))+'\'', G.result('s',d) )
    if '-lR' in sE:    ## R
      G.equate( G.result('s',d), 's', c )
      G.equate( G.result('s',d), 's', G.result('A',e) )
      G.equate( lastdep(sE), '0', G.result('A',e) )
#    dump( G )


  def convert( G, t, s=0, i=0 ):

#    print( t.c )
#    dump( G )

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

def lastdep( s ):
  d = 0
  j = len(s)
  for i in range(len(s)-1,-1,-1):
    if s[i]=='{': d+=1
    if s[i]=='}': d-=1 
    if d==0 and s[i]=='-' and i+1<len(s) and s[i+1] in 'abcdghirv': return( str(s[i:j]) )
    if d==0 and s[i]=='-': j=i
  return ''
#  l = re.findall( '-[abcdghirv](?:[^-]*|{[^{}]*(?:{[^{}]*})*})', s )
#  print( s, l )
#  return l[-1] if len(l)>0 else ''


################################################################################

def firstnolo( s ):
  d = 0
  h = 0
  for i in range(len(s)):
    if s[i]=='{': d+=1
    if s[i]=='}': d-=1
    if d==0 and s[i]=='-' and h!=0 and i>h: return( s[h:i] )
    if d==0 and s[i]=='-' and i+1<len(s) and s[i+1] in 'ghv': h = i
  return '' if h==0 else s[h:]


################################################################################

def relabel( t ):
#  print( t.c, lastdep(t.c) )
  if len(t.ch)==2:
    if lastdep(t.ch[1].c).startswith('-g') and '-lN' in t.ch[0].c: t.ch[0].c = re.sub( '-lN', '-lG', t.ch[0].c )
    if lastdep(t.ch[0].c).startswith('-h') and '-lN' in t.ch[1].c: t.ch[1].c = re.sub( '-lN', '-lH', t.ch[1].c )
    if lastdep(t.ch[1].c).startswith('-i') and '-lN' in t.ch[1].c: t.ch[1].c = re.sub( '-lN', '-lI', t.ch[1].c )
    if lastdep(t.ch[1].c).startswith('-r') and '-lN' in t.ch[1].c: t.ch[1].c = re.sub( '-lN', '-lR', t.ch[1].c )  
  ## choose one tree expansion in the following order...
  fn = firstnolo(t.c)
  if   t.c.startswith('A-aN') and len(t.ch)>0 and t.ch[0].c.startswith('L-aN'): t.ch = [ tree.Tree( 'L-aN-vN-lV', t.ch ) ]
  elif t.c.startswith('A-aN') and len(t.ch)>0 and t.ch[0].c.startswith('N') and '-l' not in t.ch[0].c: t.ch = [ tree.Tree( 'N-lZ', t.ch ) ]
  elif t.c.startswith('A-aN') and len(t.ch)>1 and t.ch[1].c.startswith('N') and '-l' not in t.ch[1].c: t.ch = [ tree.Tree( 'N-lZ', t.ch ) ]
  elif t.c.startswith('R-aN') and len(t.ch)>0 and t.ch[0].c.startswith('N') and '-l' not in t.ch[0].c: t.ch = [ tree.Tree( 'N-lZ', t.ch ) ]
  elif t.c.startswith('R-aN') and len(t.ch)>1 and t.ch[1].c.startswith('N') and '-l' not in t.ch[1].c: t.ch = [ tree.Tree( 'N-lZ', t.ch ) ]
  elif fn!='' and (len(t.ch)<1 or fn not in t.ch[0].c) and (len(t.ch)<2 or fn not in t.ch[1].c): t.ch = [ tree.Tree( re.sub(fn[0:2],'-b',re.sub('-l.','',t.c),1)+'-lE', t.ch ) ]
  for st in t.ch:
    relabel( st )


################################################################################

for line in sys.stdin:

  tr = tree.Tree( )
  tr.read( line )
  relabel( tr )
  print( tr )
  G = storestate( )
  G.a = 'top'
  G.b = 'bot'
  G.convert( tr )
  for x,l in sorted(G):
    if l!='A' and l!='B' and l!='s':  #### and l[-1]!='\'':
      sys.stdout.write( ' ' + x + ',' + l + ',' + G[x,l] )
  sys.stdout.write( '\n' )



