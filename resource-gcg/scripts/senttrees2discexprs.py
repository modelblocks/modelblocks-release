import sys
import re
import tree


def getLemma( c, w ):
  s = re.sub('-l.','',c) + ':' + w.lower()
  eqns = re.sub( '-x.*:', ':', s )
  for xrule in re.split( '-x', c )[1:] :
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
  return eqns


nSent = 0
nWord = 0


ScopesDown = {}
def getScopes( t ):
  global nWord

  ScopesDown = {}

  for st in t.ch:
    getScopes( st )

  if   len(t.ch) == 0: t.sVar = 'x' + str(nWord)
  elif len(t.ch) == 1: t.sVar = t.ch[0].sVar
  elif len(t.ch) == 2: t.sVar = t.ch[0].sVar if '-l' not in t.ch[0].c else t.ch[1].sVar if '-l' not in t.ch[1].c else None
  else: print( 'ERROR: too many children in ', t )


def translate( t, lsVars=[] ):
#  global nSent
#  global nWord

  if len(t.ch) == 1 and len(t.ch[0].ch) == 0:
#    nWord += 1
    pred = getLemma( t.c, t.ch[0].c )
#    print( nSent, nWord, pred )
    return( 'True' if pred == '' else pred + ' ' + t.sVar + ' '.join(lsVars) )
  elif len(t.ch) == 1:
    return( translate(t.ch[0],lsVars) )
  elif len(t.ch) == 2:
    if   '-lA' in t.ch[0].c:  return( '[' + translate(t.ch[0]) + '] (\\' + t.ch[0].sVar + ' ' + translate(t.ch[1],[t.ch[0].sVar]+lsVars) + ')' )
    elif '-lA' in t.ch[1].c:  return( '[' + translate(t.ch[1]) + '] (\\' + t.ch[1].sVar + ' ' + translate(t.ch[0],[t.ch[1].sVar]+lsVars) + ')' )
    elif '-lU' in t.ch[0].c:  return( '[' + translate(t.ch[0]) + '] (\\' + t.ch[0].sVar + ' ' + translate(t.ch[1],[t.ch[0].sVar]+lsVars) + ')' )
    elif '-lU' in t.ch[1].c:  return( '[' + translate(t.ch[1]) + '] (\\' + t.ch[1].sVar + ' ' + translate(t.ch[0],[t.ch[1].sVar]+lsVars) + ')' )
    elif '-lM' in t.ch[0].c:  return( translate(t.ch[0],[t.ch[1].sVar]) + ' ^ ' + translate(t.ch[1],lsVars) )
    elif '-lM' in t.ch[1].c:  return( translate(t.ch[0],lsVars) + ' ^ ' + translate(t.ch[1],[t.ch[0].sVar]) )
    elif '-lC' in t.ch[0].c:  return( translate(t.ch[0],lsVars) + ' ^ ' + translate(t.ch[1],lsVars) )
    elif '-lC' in t.ch[1].c:  return( translate(t.ch[0],lsVars) + ' ^ ' + translate(t.ch[1],lsVars) )
    else: print( 'ERROR: unhandled rule in ', t )
  else: print( 'ERROR: too many children in ', t )
#
#  for st in t.ch:
#    translate( st, lVars )

for line in sys.stdin:

  if '!ARTICLE' in line:
    nSent = 0
    print( line )

  else:
    t = tree.Tree()
    t.read( line )

    nSent += 1
    nWord = 0

    getScopes( t )
    print( translate(t) )


