import sys, os, re, collections
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
import tree

VERBOSE = False

################################################################################

def deps( s, ops='abcdghirv' ):
  lst = [ ]
  d,h = 0,0
  for i in range( len(s) ):
    if s[i]=='{': d+=1
    if s[i]=='}': d-=1
    if d==0 and s[i]=='-' and h+1<len(s) and s[h]=='-' and s[h+1] in ops: lst += [ s[h:i] ]
    if d==0 and s[i]=='-': h = i
  if h+1<len(s) and s[h]=='-' and s[h+1] in ops: lst += [ s[h:] ]
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
    if lastdep(t.ch[1].c).startswith('-g') and '-lN' in t.ch[0].c:                                                   t.ch[0].c = re.sub( '-lN', '-lG', t.ch[0].c )   ## G
#    if re.match('^-[rg]',lastdep(t.ch[0].c))!=None and '-lN' in t.ch[0].c:                                           t.ch[0].c = re.sub( '-lN', '-lR', t.ch[0].c )   ## Ra (off-spec)
#    if re.match('^-[rg]',lastdep(t.ch[1].c))!=None and '-lN' in t.ch[1].c:                                           t.ch[1].c = re.sub( '-lN', '-lR', t.ch[1].c )   ## R
    if lastdep(t.ch[0].c).startswith('-h') and '-lN' in t.ch[1].c:                                                   t.ch[1].c = re.sub( '-lN', '-lH', t.ch[1].c )   ## H
    if re.match('-b{.*-[ghirv].*}',lastdep(t.ch[0].c))!=None and '-lA' in t.ch[1].c:                                 t.ch[1].c = re.sub( '-lA', '-lI', t.ch[1].c )   ## I
    if re.match('^-[rg]',lastdep(t.ch[0].c))!=None and '-lN' in t.ch[0].c:                                           t.ch[0].c = re.sub( '-lN', '-lR', t.ch[0].c )   ## Ra (off-spec)
    if re.match('^-[rg]',lastdep(t.ch[1].c))!=None and '-lN' in t.ch[1].c:                                           t.ch[1].c = re.sub( '-lN', '-lR', t.ch[1].c )   ## R
    if '-lA' in t.ch[1].c and re.match( '^(C-bV|E-bB|F-bI|O-bN|.-aN-b{.-aN}(-x.*)|N-b{N-aD})$', t.ch[0].c ) != None: t.ch[1].c = re.sub( '-lA', '-lU', t.ch[1].c )   ## U
    if '-lA' in t.ch[0].c and t.ch[1].c=='D-aN':                                                                     t.ch[0].c = re.sub( '-lA', '-lU', t.ch[0].c )
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
    if '-lH' in t.ch[1].c:                        p = re.sub( '(.*)'+deps(t.ch[0].c,'h')[-1], '\\1', t.ch[0].c, 1 ) + ''.join( deps(re.sub(deps(t.ch[0].c,'h')[-1][3:-1],'',t.ch[1].c),'ghirv') )   ## H
    if '-lI' in t.ch[1].c:                        p = re.findall('^[^-]*',t.ch[0].c)[0] + ''.join(deps(t.ch[0].c,'abcd')[:-1]) + lcpsi + ''.join( deps(t.ch[1].c,'ghirv')[:-1] )  ## I
    if '-lR' in t.ch[0].c:                        p = t.ch[1].c                                                                                 ## Ra (off-spec)
    if '-lR' in t.ch[1].c:                        p = t.ch[0].c                                                                                 ## R
#    print( p + ' -> ' + t.ch[0].c + ' ' + t.ch[1].c )
    ## add calculated parent of children as unary branch if different from t.c...
    if re.sub('-[lx][^-]*','',p) != re.sub('-[lx][^-]*','',t.c) and re.sub('(.*)-c\{?\\1\}?','X-cX',t.c)!=p:
      if VERBOSE: print( 'T rule: ' + t.c + ' -> ' + p )
      t.ch = [ tree.Tree( p, t.ch ) ]

  ## for (new or old) unary branches...
  if len(t.ch)==1:
    locs  = deps(t.c,'abcd')
    nolos = deps(t.c,'ghirv')
    chlocs = deps(t.ch[0].c,'abcd')
    if   len(locs)>1 and len(chlocs)>1 and locs!=chlocs and locs[0][2:]==chlocs[1][2:] and locs[1][2:]==chlocs[0][2:]: t.ch[0].c += '-lQ'       ## Q
#    elif re.match('^[AR]-a',t.c)!=None and len(t.ch)==1 and t.ch[0].c.startswith('L-aN'): t.ch = [ tree.Tree( 'L'+(''.join(deps(t.c,'abcd')))+'-vN-lV', t.ch ) ]   ## V
    elif re.match('^[AR]-a',t.c) and len(t.ch)==1 and t.ch[0].c.startswith('L-aN'): t.ch = [ tree.Tree( 'L-aN-vN-lV', t.ch ) ]                  ## V
    elif t.c.startswith('A-a') and len(t.ch)==1 and t.ch[0].c.startswith('N') and len(t.ch[0].ch)>0: t.ch[0].c += '-lZ'                         ## Za
    elif t.c.startswith('R-a') and len(t.ch)==1 and t.ch[0].c.startswith('N') and len(t.ch[0].ch)>0: t.ch[0].c += '-lZ'                         ## Zb
#    elif re.match('^A-[ghirv]N',t.c)!=None and len(t.ch)==1 and t.ch[0].c.startswith('N'): t.ch = [ tree.Tree( 'A-aN-lE', t.ch ) ]              ## Ea
    elif len(chlocs)>len(locs) and len(nolos)>0 and chlocs[len(locs)][2:]==nolos[0][2:]:                                                        ## Ea
      t.ch = [ tree.Tree( re.sub(nolos[0],chlocs[len(locs)],re.sub('-l.','',t.c),1)+'-lE', t.ch ) ]
    elif len(chlocs)>len(locs) and len(nolos)>0 and chlocs[-1][2:]==nolos[0][2:]:                                                               ## Ea
      t.ch = [ tree.Tree( t.ch[0].c+'-lE', t.ch ) ]
    elif len(nolos)>0 and nolos[0][2]!='{' and len(deps(t.c))==len(deps(t.ch[0].c)) and len(chlocs)>len(locs) and nolos[0]!=chlocs[len(locs)]:  ## Ea
      t.ch = [ tree.Tree( re.sub(nolos[0],chlocs[len(locs)],re.sub('-l.','',t.c),1)+'-lE', t.ch ) ]
    elif t.c.startswith('A-vN') and t.ch[0].c=='N': t.ch = [ tree.Tree( 'A-aN-lE', t.ch ) ]
    elif len(locs)==len(chlocs) and len(deps(t.c))>len(deps(t.ch[0].c)) and len(nolos)>0 and nolos[0][2]=='{':                                  ## Eb
      t.ch = [ tree.Tree( re.sub(nolos[0],'',re.sub('-l.','',t.c),1)+'-lE', t.ch ) ]

  for st in t.ch:
    relabel( st )


################################################################################

class GCGTree( tree.Tree ):

  def __init__( t, s ):
    s = re.sub( '-iN-gN', '-gN-iN', s )  ## script puts nonlocal deps in wrong order; higher bound nolos should be buried deeper
    s = re.sub( '-rN-vN', '-vN-rN', s )
    tree.Tree.__init__( t )
    t.read( s )
    relabel( t )

