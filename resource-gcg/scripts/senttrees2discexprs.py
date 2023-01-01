import sys
import re
import tree

VERBOSE = False    ## print debugging info.
for a in sys.argv:
  if a=='-d':
    VERBOSE = True


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

def getScopes( t, Scopes, nWord=0, aboveAllInSitu=True ):

  ## Mark sites for quantifier raising...
  t.aboveAllInSitu = aboveAllInSitu
  if len(t.ch) == 2 and aboveAllInSitu:
    if '-lA' in t.ch[0].c: aboveAllInSitu = False
    if '-lA' in t.ch[1].c: aboveAllInSitu = False
  for st in t.ch:
    nWord = getScopes( st, Scopes, nWord, aboveAllInSitu )

  ## Account head words as done in gcg annotation guidelines, in order to track scope...
  if len(t.ch) == 0:
    nWord += 1
    t.sVar = str(nWord)
  elif len(t.ch) == 1:
    t.sVar = t.ch[0].sVar
  elif len(t.ch) == 2:
    t.sVar = t.ch[0].sVar if '-lU' in t.ch[0].c else t.ch[1].sVar if '-lU' in t.ch[1].c else t.ch[0].sVar if '-l' not in t.ch[0].c else t.ch[1].sVar if '-l' not in t.ch[1].c else None
  else: print( 'ERROR: too many children in ', t )

  ## Store scopes...
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

def translate( t, Scopes, Raised=[], lsNolo=[] ):

  if VERBOSE: print( t )
  if VERBOSE: print( '     non-locals: ', lsNolo )
  if VERBOSE: print( '     raised: ', Raised )

  ## Quant raising...
  t.qstore = []
  if t.sVar in Scopes and t.sVar not in Raised:
    s = translate( t, Scopes, Raised+[t.sVar], lsNolo )
    t.qstore += [( s, t.sVar )]
    output = '(RaiseTrace x' + t.sVar + ')'

  ## Pre-terminal branch...
  elif len(t.ch) == 1 and len(t.ch[0].ch) == 0:
    pred = getLemma( t.c, t.ch[0].c )
    output = 'Ident' if pred == '' else pred

  ## Unary branch...
  elif len(t.ch) == 1:
    if   '-lE' in t.ch[0].c and len(t.ch[0].c) >= len(t.c):  output = '(' + translate( t.ch[0], Scopes, Raised, lsNolo[:-1] ) + ' ' + lsNolo[-1] + ')'
    elif '-lE' in t.ch[0].c and len(t.ch[0].c) <  len(t.c):  output = '(Mod ' + translate( t.ch[0], Scopes, Raised, lsNolo[:-1] ) + ' ' + lsNolo[-1] + ')'
    elif '-lV' in t.ch[0].c:  output = '(Pasv x' + t.sVar + ' ' + translate( t.ch[0], Scopes, Raised, ['(Trace x'+t.sVar+')'] + lsNolo ) + ')'
    elif '-lZ' in t.ch[0].c:  output = '(Prop ' + translate( t.ch[0], Scopes, Raised, lsNolo ) + ')'
    else: output = translate( t.ch[0], Scopes, Raised, lsNolo )
    ## Propagate child stores...
    t.qstore = t.ch[0].qstore

  ## Binary branch...
  elif len(t.ch) == 2:
    m = getNoloArity(t.ch[0].c)
    if VERBOSE: print( '********', t.ch[0].c, t.ch[1].c, m, lsNolo[:m], lsNolo[m:] )
    '''
    ## Quant raising...
    if   ('-lA' in t.ch[0].c or '-lG' in t.ch[0].c or '-lC' in t.ch[0].c) and t.ch[0].sVar in Scopes:
      t.qstore = [( translate( t.ch[0], Scopes, lsNolo[:m] ), t.ch[0].sVar )]
      output = '(' + translate( t.ch[1], Scopes, lsNolo[m:] ) + ' (RaiseTrace x'+t.ch[0].sVar+'))'
    elif ('-lA' in t.ch[1].c or '-lH' in t.ch[1].c or '-lC' in t.ch[1].c) and t.ch[1].sVar in Scopes: 
      t.qstore = [( translate( t.ch[1], Scopes, lsNolo[m:] ), t.ch[1].sVar )]
      output = '(' + translate( t.ch[0], Scopes, lsNolo[:m] ) + ' (RaiseTrace x'+t.ch[1].sVar+'))'
    '''
    ## In-situ...
    if   '-lD' in t.ch[0].c or t.ch[0].c[0] in ',;:.!?':  output = translate( t.ch[1], Scopes, Raised, lsNolo )
    elif '-lD' in t.ch[1].c or t.ch[1].c[0] in ',;:.!?':  output = translate( t.ch[0], Scopes, Raised, lsNolo )
    elif '-lA' in t.ch[0].c or '-lU' in t.ch[0].c:  output = '(' + translate( t.ch[1], Scopes, Raised, lsNolo[m:] ) + ' ' + translate( t.ch[0], Scopes, Raised, lsNolo[:m] ) + ')'
    elif '-lA' in t.ch[1].c or '-lU' in t.ch[1].c:  output = '(' + translate( t.ch[0], Scopes, Raised, lsNolo[:m] ) + ' ' + translate( t.ch[1], Scopes, Raised, lsNolo[m:] ) + ')'
    elif '-lI' in t.ch[0].c:  output = '(' + translate( t.ch[1], Scopes, Raised, lsNolo[m:] ) + ' (SelfStore x' + t.ch[1].sVar + ' ' + translate( t.ch[0], Scopes, Raised, ['(Trace x'+t.ch[1].sVar+')'] + lsNolo[:m] ) + '))'
    elif '-lI' in t.ch[1].c:  output = '(' + translate( t.ch[0], Scopes, Raised, lsNolo[:m] ) + ' (SelfStore x' + t.ch[0].sVar + ' ' + translate( t.ch[1], Scopes, Raised, ['(Trace x'+t.ch[0].sVar+')'] + lsNolo[m:] ) + '))'
    elif '-lM' in t.ch[0].c:  output = '(Mod ' + translate( t.ch[1], Scopes, Raised, lsNolo[m:] ) + ' ' + translate( t.ch[0], Scopes, Raised, lsNolo[:m] ) + ')'
    elif '-lM' in t.ch[1].c:  output = '(Mod ' + translate( t.ch[0], Scopes, Raised, lsNolo[:m] ) + ' ' + translate( t.ch[1], Scopes, Raised, lsNolo[m:] ) + ')'
    elif '-lC' in t.ch[0].c:  output = '(And ' + translate( t.ch[0], Scopes, Raised, lsNolo ) + ' ' + translate( t.ch[1], Scopes, Raised, lsNolo ) + ')'
    elif '-lC' in t.ch[1].c:  output = translate( t.ch[1], Scopes, Raised, lsNolo )
    elif '-lG' in t.ch[0].c:  output = '(Store x' + t.ch[0].sVar + ' ' + translate( t.ch[0], Scopes, Raised, lsNolo[:m] ) + ' ' + translate( t.ch[1], Scopes, Raised, ['(Trace x'+t.ch[0].sVar+')'] + lsNolo[m:] )
    elif '-lH' in t.ch[1].c and getNoloArity(t.ch[1].c)==1:  output = '(Store x' + t.ch[1].sVar + ' (SelfStore x' + t.ch[0].sVar + ' ' + translate( t.ch[1], Scopes, Raised, lsNolo[m:] + ['(Trace x'+t.ch[0].sVar+')'] ) + ') ' + translate( t.ch[0], Scopes, Raised, ['(Trace x'+t.ch[1].sVar+')'] + lsNolo[:m] )
    elif '-lH' in t.ch[1].c:  output = '(Store x' + t.ch[1].sVar + ' ' + translate( t.ch[1], Scopes, Raised, lsNolo[m:] ) + ' ' + translate( t.ch[0], Scopes, Raised, ['(Trace x'+t.ch[1].sVar+')'] + lsNolo[:m] )
    elif '-lR' in t.ch[0].c:  output = '(Mod ' + translate( t.ch[1], Scopes, Raised, lsNolo ) + ' ' + translate( t.ch[0], Scopes, Raised, ['(Trace x'+t.ch[1].sVar+')'] ) + ')'
    elif '-lR' in t.ch[1].c:  output = '(Mod ' + translate( t.ch[0], Scopes, Raised, lsNolo ) + ' ' + translate( t.ch[1], Scopes, Raised, ['(Trace x'+t.ch[0].sVar+')'] ) + ')'
    else: print( 'ERROR: unhandled rule from ' + t.c + ' to ' + t.ch[0].c + ' ' + t.ch[1].c )
    ## Propagate child stores...
    t.qstore += (t.ch[0].qstore if hasattr(t.ch[0],'qstore') else []) + (t.ch[1].qstore if hasattr(t.ch[1],'qstore') else [])

  else: print( 'ERROR: too many children in ', t )

  if VERBOSE: print( '     quant store: ', t.qstore )
  if t.aboveAllInSitu:
    while len(t.qstore) > 0:
      l = [ r for r in t.qstore if r[1] not in Scopes.values() ]
      if len(l) > 0:
        if VERBOSE: print( 'retrieving', l[0] )
        output = '(' + l[0][0] + ' (\\x' + l[0][1] + ' True) (\\x' + l[0][1] + ' ' + output + ')'
        del Scopes[ l[0][1] ]
        t.qstore.remove( l[0] )
      else:
        print( 'ERROR: nothing in quant store', t.qstore, 'allowed by scope list', Scopes )
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

nSent = None
for nLine,line in enumerate( sys.stdin ):

  print( '========== line ' + str(nLine) + ' ==========' )

  if '!ARTICLE' in line:
    nSent = 0
    print( line[:-1] )

  else:
    t = tree.Tree()
    t.read( line )
    print( line )

    nSent += 1

    print( '----------' )
    Scopes = {}
    getScopes( t, Scopes )
    if VERBOSE: print( 'Scopes', Scopes )
    print( translate(t,Scopes) )
    print( )

