import sys
import re
import tree

################################################################################
##
##  I. HELPER FUNCTIONS
##
################################################################################

########################################
#
#  I1. get number of nonlocal arguments...
#
########################################

def getNoloArity( cat ):
  cat = re.sub( '-x.*', '', cat )
  while '{' in cat:
    cat = re.sub('\{[^\{\}]*\}','X',cat)
  return len(re.findall('-[ghirv]',cat))


########################################
#
#  I2. get predicate (lemma) from word...
#
########################################

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


########################################
#
#  I3. set scopes and variable numbers...
#
########################################

nSent = 0
nWord = 0
ScopesDown = {}
def getScopes( t ):
  global nWord

  ScopesDown = {}

  for st in t.ch:
    getScopes( st )

  if   len(t.ch) == 0:
    t.sVar = str(nWord)
    nWord += 1
  elif len(t.ch) == 1:
    t.sVar = t.ch[0].sVar
  elif len(t.ch) == 2:
    t.sVar = t.ch[0].sVar if '-l' not in t.ch[0].c else t.ch[1].sVar if '-l' not in t.ch[1].c else None
  else: print( 'ERROR: too many children in ', t )


########################################
#
#  I4. recursively translate tree to logic...
#
########################################

def translate( t, lsNolo=[] ):

  print( t )
  print( '    ', lsNolo )

  if len(t.ch) == 1 and len(t.ch[0].ch) == 0:
#    nWord += 1
    pred = getLemma( t.c, t.ch[0].c )
    return( 'Ident' if pred == '' else pred )

  elif len(t.ch) == 1:
    if   '-lE' in t.ch[0].c and len(t.ch[0].c) >= len(t.c):  return( '(' + translate( t.ch[0], lsNolo[:-1] ) + ' ' + lsNolo[-1] + ')' )
    elif '-lE' in t.ch[0].c and len(t.ch[0].c) <  len(t.c):  return( '(Mod ' + translate( t.ch[0], lsNolo[:-1] ) + ' ' + lsNolo[-1] + ')' )
    elif '-lV' in t.ch[0].c:  return( '(Pasv x' + t.sVar + ' ' + translate(t.ch[0],['(Trace x'+t.sVar+')'] + lsNolo) + ')' )
    elif '-lZ' in t.ch[0].c:  return( '(Prop ' + translate(t.ch[0],lsNolo) + ')' )
    else: return( translate(t.ch[0],lsNolo) )

  elif len(t.ch) == 2:
    m = getNoloArity(t.ch[0].c)
#    print( '********', t.ch[0].c, m, lsNolo[:m], lsNolo[m:] )
    if   '-lD' in t.ch[0].c or t.ch[0].c[0] in ',;:.!?':  return( translate(t.ch[1],lsNolo) )
    elif '-lD' in t.ch[1].c or t.ch[1].c[0] in ',;:.!?':  return( translate(t.ch[0],lsNolo) )
    elif '-lA' in t.ch[0].c or '-lU' in t.ch[0].c:  return( '(' + translate( t.ch[1], lsNolo[m:] ) + ' ' + translate( t.ch[0], lsNolo[:m] ) + ')' )
    elif '-lA' in t.ch[1].c or '-lU' in t.ch[1].c:  return( '(' + translate( t.ch[0], lsNolo[:m] ) + ' ' + translate( t.ch[1], lsNolo[m:] ) + ')' )
    elif '-lI' in t.ch[0].c:  return( '(SelfStore x' + t.ch[1].sVar + ' ' + translate( t.ch[1], lsNolo[m:] ) + ' ' + translate( t.ch[0], ['(Trace x'+t.ch[1].sVar+')'] + lsNolo[:m] ) + ')' )
    elif '-lI' in t.ch[1].c:  return( '(SelfStore x' + t.ch[0].sVar + ' ' + translate( t.ch[0], lsNolo[:m] ) + ' ' + translate( t.ch[1], ['(Trace x'+t.ch[0].sVar+')'] + lsNolo[m:] ) + ')' )
    elif '-lM' in t.ch[0].c:  return( '(Mod ' + translate( t.ch[1], lsNolo[m:] ) + ' ' + translate( t.ch[0], lsNolo[:m] ) + ')' )
    elif '-lM' in t.ch[1].c:  return( '(Mod ' + translate( t.ch[0], lsNolo[:m] ) + ' ' + translate( t.ch[1], lsNolo[m:] ) + ')' )
    elif '-lC' in t.ch[0].c:  return( '(And ' + translate( t.ch[0], lsNolo ) + ' ' + translate( t.ch[1], lsNolo ) + ')' )
    elif '-lC' in t.ch[1].c:  return( translate(t.ch[1],lsNolo) )
    elif '-lG' in t.ch[0].c:  return( '(Store x' + t.sVar + ' ' + translate(t.ch[0]) + ' ' + translate(t.ch[1],['(Trace x'+t.sVar+')'] + lsNolo) )
    elif '-lH' in t.ch[1].c:  return( '(Store x' + t.sVar + ' ' + translate(t.ch[1]) + ' ' + translate(t.ch[0],['(Trace x'+t.sVar+')'] + lsNolo) )
    elif '-lR' in t.ch[0].c:  return( '(Mod ' + translate(t.ch[1],lsNolo) + ' ' + translate(t.ch[0],['(Trace x'+t.ch[1].sVar+')']) + ')' )
    elif '-lR' in t.ch[1].c:  return( '(Mod ' + translate(t.ch[0],lsNolo) + ' ' + translate(t.ch[1],['(Trace x'+t.ch[0].sVar+')']) + ')' )
#    elif '-x%|' == t.ch[0].c[-4:]:  return( translate( t.ch[1], lsLoca, lsNolo  ) )  ## conjunction punctuation.
    else: print( 'ERROR: unhandled rule from ' + t.c + ' to ' + t.ch[0].c + ' ' + t.ch[1].c )
  else: print( 'ERROR: too many children in ', t )


'''
def translate( t, lsLoca=[], lsNolo=[] ):
#  global nSent
#  global nWord
  print( t )
  print( '    ', lsLoca, lsNolo )
  if len(t.ch) == 1 and len(t.ch[0].ch) == 0:
#    nWord += 1
    pred = getLemma( t.c, t.ch[0].c )
    pred = 'Ident' if pred == '' else pred
    return( pred if len(lsLoca)==0 else '(' + ' '.join(reversed(lsLoca + [pred])) + ')' )
  elif len(t.ch) == 1:
    if   '-lE' in t.ch[0].c  and  len(t.ch[0].c) >= len(t.c):  return( translate(t.ch[0],lsLoca+[lsNolo[-1]],lsNolo[:-1]) )
    elif '-lE' in t.ch[0].c  and  len(t.ch[0].c) <  len(t.c):  return( '(Mod ' + translate(t.ch[0],lsLoca,lsNolo[:-1]) + ' ' + lsNolo[-1] + ')' )
    elif '-lV' in t.ch[0].c:  return( translate(t.ch[0],lsLoca[1:],lsNolo+[lsLoca[0]]) )
    elif '-lZ' in t.ch[0].c:  return( '(Prop ' + translate(t.ch[0],lsLoca,lsNolo) + ')' )
    else: return( translate(t.ch[0],lsLoca,lsNolo) )
  elif len(t.ch) == 2:
    m = getNoloArity(t.ch[0].c)
#    print( '********', t.ch[0].c, m, lsNolo[:m], lsNolo[m:] )
    if   '-lD' in t.ch[0].c:  return( translate(t.ch[1],lsLoca,lsNolo) )
    elif '-lD' in t.ch[1].c:  return( translate(t.ch[0],lsLoca,lsNolo) )
    elif '-lA' in t.ch[0].c:  return( translate( t.ch[1], lsLoca + [translate(t.ch[0],[],lsNolo[:m])], lsNolo[m:] ) )
    elif '-lA' in t.ch[1].c:  return( translate( t.ch[0], lsLoca + [translate(t.ch[1],[],lsNolo[m:])], lsNolo[:m] ) )
    elif '-lU' in t.ch[0].c:  return( translate( t.ch[1], lsLoca + [translate(t.ch[0],lsLoca,lsNolo[:m])], lsNolo[m:] ) )
    elif '-lU' in t.ch[1].c:  return( translate( t.ch[0], lsLoca + [translate(t.ch[1],lsLoca,lsNolo[m:])], lsNolo[:m] ) )
    elif '-lI' in t.ch[0].c:  return( translate( t.ch[1], lsLoca + [translate(t.ch[0],[],['???'])], lsNolo ) )
    elif '-lI' in t.ch[1].c:  return( translate( t.ch[0], lsLoca + [translate(t.ch[1],[],['???'])], lsNolo ) )
    elif '-lM' in t.ch[0].c:  return( '(Mod ' + translate(t.ch[1],lsLoca,lsNolo[m:]) + ' ' + translate(t.ch[0],[],lsNolo[:m]) + ')' )
    elif '-lM' in t.ch[1].c:  return( '(Mod ' + translate(t.ch[0],lsLoca,lsNolo[:m]) + ' ' + translate(t.ch[1],[],lsNolo[m:]) + ')' )
    elif '-lC' in t.ch[0].c:  return( '(And ' + translate(t.ch[0],lsLoca,lsNolo) + ' ' + translate(t.ch[1],lsLoca,lsNolo) + ')' )
    elif '-lC' in t.ch[1].c:  return( translate(t.ch[1],lsLoca,lsNolo) )
    elif '-lG' in t.ch[0].c:  return( translate(t.ch[1],lsLoca,[translate(t.ch[0])] + lsNolo) )
    elif '-lH' in t.ch[1].c:  return( translate(t.ch[0],lsLoca,[translate(t.ch[1])] + lsNolo) )
    elif '-lR' in t.ch[0].c:  return( '(Mod ' + translate(t.ch[1],lsLoca,lsNolo) + ' ' + translate(t.ch[0],[],['(\\t \\u t x ^ u x)']) + ')' )
    elif '-lR' in t.ch[1].c:  return( '(Mod ' + translate(t.ch[0],lsLoca,lsNolo) + ' ' + translate(t.ch[1],[],['(\\t \\u t x ^ u x)']) + ')' )
    elif '-x%|' == t.ch[0].c[-4:]:  return( translate( t.ch[1], lsLoca, lsNolo  ) )  ## conjunction punctuation.
    else: print( 'ERROR: unhandled rule from ' + t.c + ' to ' + t.ch[0].c + ' ' + t.ch[1].c )
  else: print( 'ERROR: too many children in ', t )
'''

################################################################################
##
##  II. MAIN LOOP
##
################################################################################

for line in sys.stdin:

  if '!ARTICLE' in line:
    nSent = 0
    print( line[:-1] )

  else:
    t = tree.Tree()
    t.read( line )

    nSent += 1
    nWord = 0

    print( '===========' )
    getScopes( t )
    print( translate(t) )


