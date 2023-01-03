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


def getLocalArity( cat ):
  cat = re.sub( '-x.*', '', cat )
  while '{' in cat:
    cat = re.sub('\{[^\{\}]*\}','X',cat)
  return len(re.findall('-[ab]',cat))


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

def getScopes( t, Scopes, nWord=0 ):

  ## Recurse...
  for st in t.ch:
    nWord = getScopes( st, Scopes, nWord )

  ## Account head words as done in gcg annotation guidelines, in order to track scope...
  if len(t.ch) == 0:
    nWord += 1
    t.sVar = str(nWord)
  elif len(t.ch) == 1:
    t.sVar = t.ch[0].sVar
  elif len(t.ch) == 2:
    t.sVar = t.ch[0].sVar if '-lU' in t.ch[0].c else t.ch[1].sVar if '-lU' in t.ch[1].c else t.ch[0].sVar if '-l' not in t.ch[0].c else t.ch[1].sVar if '-l' not in t.ch[1].c else None
  else: print( '\nERROR: too many children in ', t )

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
#  I4. mark sites for raising above all in-situs...
#
########################################

def markSites( t, Scopes, aboveAllInSitu=True ):

  ## Mark until argument...
  t.aboveAllInSitu = aboveAllInSitu
  if len(t.ch) == 2 and aboveAllInSitu and getLocalArity(t.c)==0:
    if ('-lA' in t.ch[0].c or '-lU' in t.ch[0].c) and t.ch[0].sVar not in Scopes: aboveAllInSitu = False
    if ('-lA' in t.ch[1].c or '-lU' in t.ch[1].c) and t.ch[1].sVar not in Scopes: aboveAllInSitu = False

  ## Recurse...
  for st in t.ch:
    markSites( st, Scopes, aboveAllInSitu )


########################################
#
#  I5. recursively translate tree to logic...
#
########################################

indent = 0
def translate( t, Scopes, Raised=[], lsNolo=[] ):

  ## A. Verbose reporting...
  global indent
  indent += 2
  if VERBOSE: print( ' '*indent, 'tree:', t )
  if VERBOSE: print( ' '*indent, 'non-locals:', lsNolo )
  if VERBOSE: print( ' '*indent, 'raised:', Raised )

  ## B.i. Store quantifier...
  t.qstore = []
  ## If can scope in situ, remove from scopes and translate further...
  if t.sVar in Scopes and t.sVar not in Raised and t.sVar not in Scopes.values():
    del Scopes[ t.sVar ]
  ## If scoped and cannot be in situ, store...
  if t.sVar in Scopes and t.sVar not in Raised and t.sVar in Scopes.values():
    markSites( t, Scopes )
    s = translate( t, Scopes, Raised+[t.sVar], lsNolo )
    t.qstore = [( t.qstore, s, t.sVar )]
    output = [ 'RaiseTrace', 'x'+t.sVar ]
#    t.aboveAllInSitu = False

  ## B.ii. Pre-terminal branch...
  elif len(t.ch) == 1 and len(t.ch[0].ch) == 0:
    pred = getLemma( t.c, t.ch[0].c )
    output = 'Ident' if pred == '' else '@'+pred

  ## B.iii. Unary branch...
  elif len(t.ch) == 1:
    if   '-lE' in t.ch[0].c and len(t.ch[0].c) >= len(t.c):  output = [ translate( t.ch[0], Scopes, Raised, lsNolo[:-1] ), lsNolo[-1] ]
    elif '-lE' in t.ch[0].c and len(t.ch[0].c) <  len(t.c):  output = [ 'Mod'+str(getLocalArity(t.ch[0].c)), translate( t.ch[0], Scopes, Raised, lsNolo[:-1] ), lsNolo[-1] ]
    elif '-lV' in t.ch[0].c:  output = [ 'Pasv', 'x'+ t.sVar, translate( t.ch[0], Scopes, Raised, [( 'Trace', 'x'+t.sVar )] + lsNolo ) ]
    elif '-lZ' in t.ch[0].c:  output = [ 'Prop', translate( t.ch[0], Scopes, Raised, lsNolo ) ]
    elif getLocalArity(t.c) < getLocalArity(t.ch[0].c): output = [ translate( t.ch[0], Scopes, Raised, lsNolo ), 'Some' ]
    else: output = translate( t.ch[0], Scopes, Raised, lsNolo )
    ## Propagate child stores...
    t.qstore = t.ch[0].qstore

  ## B.iv. Binary branch...
  elif len(t.ch) == 2:
    m = getNoloArity(t.ch[0].c)
    if VERBOSE: print( ' '*indent, 'child cats and nolos:', t.ch[0].c, t.ch[1].c, m, lsNolo[:m], lsNolo[m:] )
    ## In-situ...
    if   '-lD' in t.ch[0].c or t.ch[0].c[0] in ',;:.!?':  output = translate( t.ch[1], Scopes, Raised, lsNolo )
    elif '-lD' in t.ch[1].c or t.ch[1].c[0] in ',;:.!?':  output = translate( t.ch[0], Scopes, Raised, lsNolo )
    elif '-lA' in t.ch[0].c or '-lU' in t.ch[0].c:  output = [ translate( t.ch[1], Scopes, Raised, lsNolo[m:] ), translate( t.ch[0], Scopes, Raised, lsNolo[:m] ) ]
    elif '-lA' in t.ch[1].c or '-lU' in t.ch[1].c:  output = [ translate( t.ch[0], Scopes, Raised, lsNolo[:m] ), translate( t.ch[1], Scopes, Raised, lsNolo[m:] ) ]
    elif '-lI' in t.ch[0].c:  output = [ translate( t.ch[1], Scopes, Raised, lsNolo[m:] ), [ 'SelfStore', 'x'+t.ch[1].sVar, translate( t.ch[0], Scopes, Raised, [( 'Trace', 'x'+t.ch[1].sVar )] + lsNolo[:m] ) ] ]
    elif '-lI' in t.ch[1].c:  output = [ translate( t.ch[0], Scopes, Raised, lsNolo[:m] ), [ 'SelfStore', 'x'+t.ch[0].sVar, translate( t.ch[1], Scopes, Raised, [( 'Trace', 'x'+t.ch[0].sVar )] + lsNolo[m:] ) ] ]
    elif '-lM' in t.ch[0].c:  output = [ 'Mod'+str(getLocalArity(t.ch[1].c)), translate( t.ch[1], Scopes, Raised, lsNolo[m:] ), translate( t.ch[0], Scopes, Raised, lsNolo[:m] ) ]
    elif '-lM' in t.ch[1].c:  output = [ 'Mod'+str(getLocalArity(t.ch[0].c)), translate( t.ch[0], Scopes, Raised, lsNolo[:m] ), translate( t.ch[1], Scopes, Raised, lsNolo[m:] ) ]
    elif '-lC' in t.ch[0].c:  output = [ 'And'+str(getLocalArity(t.ch[0].c)), translate( t.ch[0], Scopes, Raised, lsNolo ), translate( t.ch[1], Scopes, Raised, lsNolo ) ]
    elif '-lC' in t.ch[1].c:  output = translate( t.ch[1], Scopes, Raised, lsNolo )
    elif '-lG' in t.ch[0].c:  output = [ 'Store', 'x'+t.ch[0].sVar, translate( t.ch[0], Scopes, Raised, lsNolo[:m] ), translate( t.ch[1], Scopes, Raised, [( 'Trace', 'x'+t.ch[0].sVar )] + lsNolo[m:] ) ]
    elif '-lH' in t.ch[1].c and getNoloArity(t.ch[1].c)==1:  output = [ 'Store', 'x'+t.ch[1].sVar, [ 'SelfStore', 'x'+t.ch[0].sVar, translate( t.ch[1], Scopes, Raised, lsNolo[m:] + [('Trace', 'x'+t.ch[0].sVar )] ) ],
                                                                        translate( t.ch[0], Scopes, Raised, [('Trace', 'x'+t.ch[1].sVar )] + lsNolo[:m] ) ]
    elif '-lH' in t.ch[1].c:  output = [ 'Store', 'x'+t.ch[1].sVar, translate( t.ch[1], Scopes, Raised, lsNolo[m:] ), translate( t.ch[0], Scopes, Raised, [( 'Trace', 'x'+t.ch[1].sVar )] + lsNolo[:m] ) ]
    elif '-lR' in t.ch[0].c:  output = [ 'Mod'+str(getLocalArity(t.ch[1].c)), translate( t.ch[1], Scopes, Raised, lsNolo ), translate( t.ch[0], Scopes, Raised, [( 'Trace', 'x'+t.ch[1].sVar )] ) ]
    elif '-lR' in t.ch[1].c:  output = [ 'Mod'+str(getLocalArity(t.ch[0].c)), translate( t.ch[0], Scopes, Raised, lsNolo ), translate( t.ch[1], Scopes, Raised, [( 'Trace', 'x'+t.ch[0].sVar )] ) ]
    else: print( '\nERROR: unhandled rule from ' + t.c + ' to ' + t.ch[0].c + ' ' + t.ch[1].c )
    ## Propagate child stores...
    t.qstore += (t.ch[0].qstore if hasattr(t.ch[0],'qstore') else []) + (t.ch[1].qstore if hasattr(t.ch[1],'qstore') else [])

  ## B.v. Fail...
  else: print( '\nERROR: too many children in ', t )

  ## C. Retrieve quantifier...
  if VERBOSE: print( ' '*indent, 'cat and scopes:', t.c, Scopes )
  if VERBOSE: print( ' '*indent, 'quant store: ', t.qstore )
  if t.aboveAllInSitu:
    while len(t.qstore) > 0:
      l = [ r for r in t.qstore if r[2] not in Scopes.values() ]
      if len(l) > 0:
        if VERBOSE: print( ' '*indent, 'retrieving:', l[0] )
        output = [ '\\r'+l[0][2], '\\s'+l[0][2], l[0][1], [ '\\x'+l[0][2], 'True' ], [ '\\x'+l[0][2], output, 'r'+l[0][2], 's'+l[0][2] ] ]
        del Scopes[ l[0][2] ]
        t.qstore += l[0][0]
        t.qstore.remove( l[0] )
      else:
#        print( '\nERROR: nothing in quant store', t.qstore, 'allowed by scope list', Scopes )
        break

  if VERBOSE: print( ' '*indent, 'returning:', output )
  indent -= 2
  return( output )


################################################################################
##
##  II. HELPER FUNCTIONS FOR MACRO SUBSTITUTION
##
################################################################################

'''
def unpack( t ):
  print( 'trying', t )
  if isinstance(t,str): return t
  elif t[0] == 'And1': return( ( '\\q', '\\r', '\\s', ('^', ( unpack(t[1]), 'q', 'r', 's' ), ( unpack(t[2]), 'q', 'r', 's' ) ) ) )
  elif t[0] == 'And0': return( ( '\\q', '\\r', ('^', ( unpack(t[1]), 'r', 's' ), ( unpack(t[2]), 'r', 's' ) ) ) )
  elif t[0] == 'Mod1': return( ( '\\q', '\\r', '\\s', ( unpack(t[1]), 'q', ( '\\x', ( '^', ('r', 'x' ), ( unpack(t[2]), ( '\\t', '\\u', '^', ('t','x'), ('u','x') ), 'U', 'U' ) ) ), 's' ) ) )
  elif t[0] == 'Mod0': return( ( '\\r', '\\s', ( unpack(t[1]), ( '\\x', ( '^', ('r', 'x' ), ( unpack(t[2]), ( '\\t', '\\u', '^', ('t','x'), ('u','x') ), 'U', 'U' ) ) ), 's' ) ) )
  elif t[0][0]=='@' and ':' in t[0] and getLocalArity( t[0].split(':')[0] ) == 1: return( ( ('\\r', '\\s', unpack(t[1]), 'U', ( '\\x', 'Some', ( '\\e', '^', ( unpack(t[0][1:]), 'x', 'e' ), ('r','e') ), 's' ) ) ) )
  elif t[0][0]=='@' and ':' in t[0] and getLocalArity( t[0].split(':')[0] ) == 2: return( ( ('\\r', '\\s', unpack(t[2]), 'U', ( '\\x', unpack(t[1]), 'U', ( '\\y', 'Some', ( '\\e', '^', ( unpack(t[0][1:]), 'e', 'x', 'y' ), ('r','e') ), 's' ) ) ) ) )
  elif len(t)==4: return( ( unpack(t[0]), unpack(t[1]), unpack(t[2]), unpack(t[3]) ) )
  elif len(t)==3: return( ( unpack(t[0]), unpack(t[1]), unpack(t[2]) ) )
  elif len(t)==2: return( ( unpack(t[0]), unpack(t[1]) ) )
  else: print( 'ERROR: cannot unpack: ' + str(t) )
'''


def unpack( t ):
  if not isinstance(t,str):
    return([ unpack(st) for st in t ])
  elif t=='And0': return( [ '\\f', '\\g',        '\\r', '\\s', ['^', [ 'g', 'r', 's' ],      [ 'f', 'r', 's' ] ] ] )
  elif t=='And1': return( [ '\\f', '\\g', '\\q', '\\r', '\\s', ['^', [ 'g', 'q', 'r', 's' ], [ 'f', 'q', 'r', 's' ] ] ] )
  elif t=='Mod0': return( [ '\\f', '\\g',        '\\r', '\\s', [ 'f',      [ '\\x', [ '^', ['r', 'x' ], [ 'g', [ '\\t', '\\u', '^', ['t','x'], ['u','x'] ], 'U', 'U' ] ] ], 's' ] ] )
  elif t=='Mod1': return( [ '\\f', '\\g', '\\q', '\\r', '\\s', [ 'f', 'q', [ '\\x', [ '^', ['r', 'x' ], [ 'g', [ '\\t', '\\u', '^', ['t','x'], ['u','x'] ], 'U', 'U' ] ] ], 's' ] ] )
  elif t.split(':')[0] == '@N-b{N-aD}': return( [ '\\q', '\\r', '\\s', t[1:], [ '\\x', '^', ['r','x'], ['q',['\\xx','Equal','xx','x'],'U'] ], 's' ] )
  elif t[0]=='@' and getLocalArity( t.split(':')[0] ) == 1: return( [        '\\q', '\\r', '\\s', 'q', 'U', [ '\\x',                    'Some', [ '\\e', '^', [t[1:],'e','x'    ], ['r','e'] ], 's'   ] ] )
  elif t[0]=='@' and getLocalArity( t.split(':')[0] ) == 2: return( [ '\\p', '\\q', '\\r', '\\s', 'q', 'U', [ '\\x', 'p', 'U', [ '\\y', 'Some', [ '\\e', '^', [t[1:],'e','x','y'], ['r','e'] ], 's' ] ] ] )
  else: return( t )


def replace( t, old, new ):
  if VERBOSE: print( 'replacing:', old, 'with', new, 'in', t )
  if t == old:
    return( new )
  elif isinstance(t,str):
    return( t )
  elif any( [ st[0]=='\\' and st[1:]==old for st in t ] ):
    return( t )
  else:
    return( [ replace( st, old, new ) for st in t ] )


def betaReduce( t ):
  if VERBOSE: print( 'reducing:', t )
  ## If string, skip...
  if isinstance(t,str):
    return
  ## If initial term is string, betaReduce children...
  elif isinstance(t[0],str):
    for st in t:
      betaReduce( st )
  ## Flatten initial application... 
  elif t[0][0][0]!='\\':
    t[:] = t[0] + t[1:]
    betaReduce( t )
  ## Substitute second term for initial lambda variable of initial (abstraction) term...
  elif len(t) >= 2:
    t[:] = [ replace( t[0][1:], t[0][0][1:], t[1] ) ] + t[2:]
    betaReduce( t )
  else:
    t[:] = t[0]
    betaReduce( t )

'''
def binarize( t ):
  if isinstance(t,str): return( t )
  elif len(t)==2: return( [ binarize(t[0]), binarize(t[1]) ] )
  elif t[0][0]=='\\': return( [ binarize(t[0]), binarize(t[1:]) ] )
  else: return( [ binarize(t[:-1]), binarize(t[-1]) ] )


def replace( t, old, new ):
  if t == old: return( new )
  elif isinstance(t,str): return( t )
  elif '\\' == t[0][0] and t[0][1:] == old: return( [ t[0], t[1] ] )
  else: return( [ replace(t[0],old,new), replace(t[1],old,new) ] )


def reduce( t ):
  if isinstance(t,str): return( t )
  elif isinstance(t[0],str): return( [ t[0], reduce(t[1]) ] )
  elif t[0][0][0] == '\\': return( reduce( replace( t[0][1], t[0][0][1:], t[1] ) ) )
  else: return( [ reduce(t[0]), reduce(t[1]) ] )
'''

################################################################################
##
##  III. MAIN LOOP
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
    markSites( t, Scopes )
    if VERBOSE: print( 'Scopes', Scopes )
    out = translate(t,Scopes)
    if t.qstore != []: print( '\nERROR: nothing in quant store', t.qstore, 'allowed by scope list', Scopes )
    print( out )
    out = unpack(out)
    betaReduce( out )
    print( out )
    print( )

