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
def getScopes( t, Scopes, nWord=0, aboveAllInSitu=True ):
#  global nWord

  t.aboveAllInSitu = aboveAllInSitu
  if len(t.ch) == 2 and aboveAllInSitu:
    if '-lA' in t.ch[0].c: aboveAllInSitu = False
    if '-lA' in t.ch[1].c: aboveAllInSitu = False
  for st in t.ch:
    nWord = getScopes( st, Scopes, nWord, aboveAllInSitu )

  if len(t.ch) == 0:
    t.sVar = str(nWord)
    nWord += 1
  elif len(t.ch) == 1:
    t.sVar = t.ch[0].sVar
  elif len(t.ch) == 2:
    t.sVar = t.ch[0].sVar if '-l' not in t.ch[0].c else t.ch[1].sVar if '-l' not in t.ch[1].c else None
  else: print( 'ERROR: too many children in ', t )

  if '-yQ' in t.c:
    Scopes[t.sVar] = '0'
  m = re.search( '-s([0-9][0-9])?([0-9][0-9])', t.c )
  if m != None:
    sDest = str(int(m.group(2)))
    if sDest not in Scopes: Scopes[sDest] = '0'
    Scopes[t.sVar] = sDest

  return( nWord )


########################################
#
#  I4. recursively translate tree to logic...
#
########################################

def translate( t, Scopes, lsNolo=[] ):

  print( t )
  print( '    ', lsNolo )

#  ## Raise...
#  if '-lA' in t.c and t.sVar in Scopes:
#    t.raised = [translate( t, Scopes,

  t.raised = []

  ## Pre-terminal branch...
  if len(t.ch) == 1 and len(t.ch[0].ch) == 0:
#    nWord += 1
    pred = getLemma( t.c, t.ch[0].c )
    output = 'Ident' if pred == '' else pred

  ## Unary branch...
  elif len(t.ch) == 1:
    if   '-lE' in t.ch[0].c and len(t.ch[0].c) >= len(t.c):  output = '(' + translate( t.ch[0], Scopes, lsNolo[:-1] ) + ' ' + lsNolo[-1] + ')'
    elif '-lE' in t.ch[0].c and len(t.ch[0].c) <  len(t.c):  output = '(Mod ' + translate( t.ch[0], Scopes, lsNolo[:-1] ) + ' ' + lsNolo[-1] + ')'
    elif '-lV' in t.ch[0].c:  output = '(Pasv x' + t.sVar + ' ' + translate( t.ch[0], Scopes, ['(Trace x'+t.sVar+')'] + lsNolo ) + ')'
    elif '-lZ' in t.ch[0].c:  output = '(Prop ' + translate( t.ch[0], Scopes, lsNolo ) + ')'
    else: output = translate( t.ch[0], Scopes, lsNolo )
    ## Propagate child stores...
    t.raised = t.ch[0].raised

  ## Binary branch...
  elif len(t.ch) == 2:
    m = getNoloArity(t.ch[0].c)
#    print( '********', t.ch[0].c, m, lsNolo[:m], lsNolo[m:] )
    ## Quant raising...
    if   '-lA' in t.ch[0].c and t.ch[0].sVar in Scopes:
      t.raised = [( translate( t.ch[0], Scopes, lsNolo[:m] ), t.ch[0].sVar )]
      output = '(' + translate( t.ch[1], Scopes, lsNolo[m:] ) + ' (RaiseTrace x'+t.ch[0].sVar+'))'
    elif '-lA' in t.ch[1].c and t.ch[1].sVar in Scopes: 
      t.raised = [( translate( t.ch[1], Scopes, lsNolo[:m] ), t.ch[1].sVar )]
      output = '(' + translate( t.ch[0], Scopes, lsNolo[m:] ) + ' (RaiseTrace x'+t.ch[1].sVar+'))'
    ## In-situ...
    elif '-lD' in t.ch[0].c or t.ch[0].c[0] in ',;:.!?':  output = translate( t.ch[1], Scopes, lsNolo )
    elif '-lD' in t.ch[1].c or t.ch[1].c[0] in ',;:.!?':  output = translate( t.ch[0], Scopes, lsNolo )
    elif '-lA' in t.ch[0].c or '-lU' in t.ch[0].c:  output = '(' + translate( t.ch[1], Scopes, lsNolo[m:] ) + ' ' + translate( t.ch[0], Scopes, lsNolo[:m] ) + ')'
    elif '-lA' in t.ch[1].c or '-lU' in t.ch[1].c:  output = '(' + translate( t.ch[0], Scopes, lsNolo[:m] ) + ' ' + translate( t.ch[1], Scopes, lsNolo[m:] ) + ')'
    elif '-lI' in t.ch[0].c:  output = '(' + translate( t.ch[1], Scopes, lsNolo[m:] ) + ' (SelfStore x' + t.ch[1].sVar + ' ' + translate( t.ch[0], Scopes, ['(Trace x'+t.ch[1].sVar+')'] + lsNolo[:m] ) + '))'
    elif '-lI' in t.ch[1].c:  output = '(' + translate( t.ch[0], Scopes, lsNolo[:m] ) + ' (SelfStore x' + t.ch[0].sVar + ' ' + translate( t.ch[1], Scopes, ['(Trace x'+t.ch[0].sVar+')'] + lsNolo[m:] ) + '))'
    elif '-lM' in t.ch[0].c:  output = '(Mod ' + translate( t.ch[1], Scopes, lsNolo[m:] ) + ' ' + translate( t.ch[0], Scopes, lsNolo[:m] ) + ')'
    elif '-lM' in t.ch[1].c:  output = '(Mod ' + translate( t.ch[0], Scopes, lsNolo[:m] ) + ' ' + translate( t.ch[1], Scopes, lsNolo[m:] ) + ')'
    elif '-lC' in t.ch[0].c:  output = '(And ' + translate( t.ch[0], Scopes, lsNolo ) + ' ' + translate( t.ch[1], Scopes, lsNolo ) + ')'
    elif '-lC' in t.ch[1].c:  output = translate( t.ch[1], Scopes, lsNolo )
    elif '-lG' in t.ch[0].c:  output = '(Store x' + t.ch[0].sVar + ' ' + translate( t.ch[0], Scopes, lsNolo[:m] ) + ' ' + translate( t.ch[1], Scopes, ['(Trace x'+t.ch[0].sVar+')'] + lsNolo[m:] )
    elif '-lH' in t.ch[1].c:  output = '(Store x' + t.ch[1].sVar + ' ' + translate( t.ch[1], Scopes, lsNolo[m:] ) + ' ' + translate( t.ch[0], Scopes, ['(Trace x'+t.ch[1].sVar+')'] + lsNolo[:m] )
    elif '-lR' in t.ch[0].c:  output = '(Mod ' + translate( t.ch[1], Scopes, lsNolo ) + ' ' + translate( t.ch[0], Scopes, ['(Trace x'+t.ch[1].sVar+')'] ) + ')'
    elif '-lR' in t.ch[1].c:  output = '(Mod ' + translate( t.ch[0], Scopes, lsNolo ) + ' ' + translate( t.ch[1], Scopes, ['(Trace x'+t.ch[0].sVar+')'] ) + ')'
    else: print( 'ERROR: unhandled rule from ' + t.c + ' to ' + t.ch[0].c + ' ' + t.ch[1].c )
    ## Propagate child stores...
    t.raised += (t.ch[0].raised if hasattr(t.ch[0],'raised') else []) + (t.ch[1].raised if hasattr(t.ch[1],'raised') else [])

  else: print( 'ERROR: too many children in ', t )

  print( '     raised: ', t.raised )
  if t.aboveAllInSitu:
    while len(t.raised) > 0:
      l = [ r for r in t.raised if r[1] not in Scopes.values() ]
      if len(l) > 0:
        print( 'killing', l[0] )
        output = '(' + l[0][0] + ' (\\x' + l[0][1] + ' True) (\\x' + l[0][1] + ' ' + output + ')'
        del Scopes[ l[0][1] ]
        t.raised.remove( l[0] )
      else:
        print( 'ERROR: no raisers (', t.raised, ') allowed by scope list (', Scopes, ')' )
        break

  return( output )


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
    Scopes = {}
    getScopes( t, Scopes )
    print( 'Scopes', Scopes )
    print( translate(t,Scopes) )


