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


ScopesDown = {}
def getScopes( t ):
  ScopesDown = {}


nSent = 0
nWord = 0
def translate( t, lVars ):
  global nSent
  global nWord

  if len(t.ch) == 1 and len(t.ch[0].ch) == 0:
    nWord += 1
    pred = getLemma( t.c, t.ch[0].c )
    print( nSent, nWord, pred )
#  else:
#    if '-lA' in t.ch[0]:

  for st in t.ch:
    translate( st, lVars )

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
    print( translate(t,nSent) )


