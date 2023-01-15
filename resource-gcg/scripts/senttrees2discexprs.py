import sys
import re
import tree
import lex

VERBOSE = False    ## print debugging info.
ANAPH = True
SKIPS = []
for a in sys.argv:
  if a=='-d':  VERBOSE = True
  if a=='-n':  ANAPH = False
  if a.startswith('-s'):  SKIPS += [ int(a[2:]) ]

################################################################################
##
##  I. HELPER FUNCTIONS
##
################################################################################

########################################
#
#  I.A. get number of local/nonlocal arguments...
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
#  I.B. set scopes and variable numbers...
#
########################################

def setHeadScopeAnaph( t, nSent, Scopes, Anaphs, nWord=0 ):

  ## Recurse...
  for st in t.ch:
    nWord = setHeadScopeAnaph( st, nSent, Scopes, Anaphs, nWord )

  ## Account head words as done in gcg annotation guidelines, in order to track scope...
  if len(t.ch) == 0:
    nWord += 1
    t.sVar = str(nSent*100+nWord)
  elif len(t.ch) == 1:
    t.sVar = t.ch[0].sVar
  elif len(t.ch) == 2:
    t.sVar = t.ch[0].sVar if '-lU' in t.ch[0].c else t.ch[1].sVar if '-lU' in t.ch[1].c else t.ch[0].sVar if '-l' not in t.ch[0].c else t.ch[1].sVar if '-l' not in t.ch[1].c else None
  else: print( '\nERROR: too many children in ', t )

  ## Store scopes and anaphora...
  ## Scopes...
  if '-yQ' in t.c:
    Scopes[t.sVar] = '0'
  m = re.search( '-s([0-9][0-9])?([0-9][0-9])', t.c )
  if m != None:
    sDest = str( (nSent if m.group(1)==None else int(m.group(1))) * 100 + int(m.group(2)) )
    if sDest not in Scopes: Scopes[sDest] = '0'
    Scopes[t.sVar] = sDest
  ## Anaphora...
  m = re.search( '-[nm]([0-9][0-9])?([0-9][0-9])', t.c )
  if m != None:
    Anaphs[t.sVar] = str( (nSent if m.group(1)==None else int(m.group(1))) * 100 + int(m.group(2)) )

  t.bMax = True
  for st in t.ch:
    st.bMax = ( st.sVar!= t.sVar )

  return( nWord )


########################################
#
#  I.C. mark sites for raising above all in-situs...
#
########################################

def markSites( t, Scopes, aboveAllInSitu=True ):

  ## Mark until un-scoped argument...
  t.aboveAllInSitu = aboveAllInSitu
  if len(t.ch) == 2 and aboveAllInSitu and getLocalArity(t.c)==0:
    if ('-lA' in t.ch[0].c or '-lU' in t.ch[0].c) and t.ch[0].sVar not in Scopes: aboveAllInSitu = False
    if ('-lA' in t.ch[1].c or '-lU' in t.ch[1].c) and t.ch[1].sVar not in Scopes: aboveAllInSitu = False

  ## Recurse...
  for st in t.ch:
    markSites( st, Scopes, aboveAllInSitu )


########################################
#
#  I.D. recursively translate tree to logic...
#
########################################

indent = 0
def translate( t, Scopes, Anaphs, lsNolo=[] ):

  ## 1. Verbose reporting...
  global indent
  indent += 2
  if VERBOSE: print( ' '*indent, 'var,max,tree:', t.sVar, t.bMax, t )
  if VERBOSE: print( ' '*indent, 'non-locals:', lsNolo )

  ## 2.a. If scoped and cannot be in situ, mark possible sites...
  t.qstore = []
  if t.bMax and t.sVar in Scopes and t.sVar in Scopes.values():
    markSites( t, Scopes )

  ## 2.b. Pre-terminal branch...
  if len(t.ch) == 1 and len(t.ch[0].ch) == 0:
    pred = lex.getFn( t.c, t.ch[0].c )
    if   pred == '' and t.c == 'N-b{N-aD}-x%|':  output = 'IdentSome'
    elif pred == '' and t.c == 'Ne-x%|':         output = 'Some'
    elif pred == '':                             output = 'Ident'
    elif t.c == 'R-aN-rN-xR%|A%':                output = [ '@'+pred, lsNolo[-1] ]
    elif t.c == 'N-rN':                          output = [ '@'+pred, lsNolo[-1] ]
    else:                                        output = '@'+pred
#    output = '@'+pred if pred != '' else ('IdentSome' if t.c == 'N-b{N-aD}-x%|'  else  'Some' if t.c == 'Ne-x%|'  else 'Ident')

  ## 3.c. Unary branch...
  elif len(t.ch) == 1:
    form = re.sub( '-[lmnstuwxy][^ ]*', '', t.ch[0].c + ' ' + t.c )
    if   '-lF' in t.ch[0].c and re.search( '^V-iN((?:-[ghirv][^ ]*)?) N\\1$', form ) != None:  output = [ '\\r', '\\s', 'All', [ '\\x'+t.sVar, '^', ['r','x'+t.sVar], translate( t.ch[0], Scopes, Anaphs, [ ['Trace','x'+t.sVar] ] + lsNolo ) ], 's' ]
    elif '-lE' in t.ch[0].c and getLocalArity(t.c) + 1 == getLocalArity(t.ch[0].c):  output = [ translate( t.ch[0], Scopes, Anaphs, lsNolo[:-1] ), lsNolo[-1] ]
    elif '-lE' in t.ch[0].c and getLocalArity(t.c) == getLocalArity(t.ch[0].c):  output = [ 'Mod'+str(getLocalArity(t.ch[0].c)), translate( t.ch[0], Scopes, Anaphs, lsNolo[:-1] ), lsNolo[-1] ]
    elif '-lQ' in t.ch[0].c and getLocalArity(t.ch[0].c) == 1:  output = [ '\\p', '\\q', translate( t.ch[0], Scopes, Anaphs, lsNolo ), 'p' ]
    elif '-lQ' in t.ch[0].c and getLocalArity(t.ch[0].c) == 2:  output = [ '\\p', '\\q', translate( t.ch[0], Scopes, Anaphs, lsNolo ), 'q', 'p' ]
    elif '-lV' in t.ch[0].c and re.search( '^L-aN-vN((?:-[ghirv][^ ]*)?) A-aN\\1$', form ) != None:  output = [ '\\q', '\\r', '\\s', 'q', Univ, [ '\\x'+t.sVar, translate( t.ch[0], Scopes, Anaphs, [ ['Trace','x'+t.sVar] ] + lsNolo ), 'Some', 'r', 's' ] ]
#   elif '-lV' in t.ch[0].c:  output = [ 'Pasv', 'x'+ t.sVar, translate( t.ch[0], Scopes, Anaphs, [ ['Trace','x'+t.sVar] ] + lsNolo ) ]
    ## ACCOMMODATE SLOPPY ANNOTATION of '-lZ' with arg elision...
    elif '-lZ' in t.ch[0].c and getLocalArity(t.c) == getLocalArity(t.ch[0].c):  output = [ 'Prop', [ translate( t.ch[0], Scopes, Anaphs, lsNolo ), 'Some' ] ]
    elif '-lZ' in t.ch[0].c and getLocalArity(t.c) == getLocalArity(t.ch[0].c) + 1:  output = [ 'Prop', translate( t.ch[0], Scopes, Anaphs, lsNolo ) ]
    elif '-lz' in t.ch[0].c and getLocalArity(t.c) == getLocalArity(t.ch[0].c) + 1:  output = [ 'Prop', translate( t.ch[0], Scopes, Anaphs, lsNolo ) ]
    elif '-l' not in t.ch[0].c and getLocalArity(t.c) + 1 == getLocalArity(t.ch[0].c):  output = [ translate( t.ch[0], Scopes, Anaphs, lsNolo ), 'Some' ]
    elif '-l' not in t.ch[0].c and getLocalArity(t.c) == getLocalArity(t.ch[0].c):  output = translate( t.ch[0], Scopes, Anaphs, lsNolo )
    else:
      print( 'WARNING: Assuming', t.c, '->', t.ch[0].c, getLocalArity(t.c), getLocalArity(t.ch[0].c), 'is simple type change' )
      output = translate( t.ch[0], Scopes, Anaphs, lsNolo )
    ## Propagate child stores...
    t.qstore = t.ch[0].qstore

  ## 2.d. Binary branch...
  elif len(t.ch) == 2:
    m = getNoloArity(t.ch[0].c)
    if VERBOSE: print( ' '*indent, 'child cats and nolos:', t.ch[0].c, t.ch[1].c, m, lsNolo[:m], lsNolo[m:] )
    ## Check...
    form = re.sub( '-[lmnstuwxy][^ ]*', '', t.ch[0].c + ' ' + t.ch[1].c + ' ' + t.c )
    if   ('-lA' in t.ch[0].c or '-lU' in t.ch[0].c) and getLocalArity(t.ch[1].c) != getLocalArity(t.c)+1:  sys.stdout.write( 'ERROR: Bad arity in ' + t.c + ' -> ' + t.ch[0].c + ' ' + t.ch[1].c + '\n' )
    elif ('-lA' in t.ch[0].c or '-lU' in t.ch[0].c) and re.search( '^(.*)((?:-[ghirv][^ ]*)?) (.*)-a\\1((?:-[ghirv][^ ]*)?) \\3\\2\\4$', form ) == None and re.search( '^(.*)((?:-[ghirv][^ ]*)?) (.*)-a{\\1}((?:-[ghirv][^ ]*)?) \\3\\2\\4$', form ) == None:
      sys.stdout.write( 'WARNING: Bad category in ' + t.c + ' -> ' + t.ch[0].c + ' ' + t.ch[1].c + '\n' )
    elif ('-lA' in t.ch[1].c or '-lU' in t.ch[1].c) and getLocalArity(t.ch[0].c) != getLocalArity(t.c)+1:  sys.stdout.write( 'ERROR: Bad arity in ' + t.c + ' -> ' + t.ch[0].c + ' ' + t.ch[1].c + '\n' )
    elif ('-lA' in t.ch[1].c or '-lU' in t.ch[1].c) and re.search( '^(.*)-b(.*)((?:-[ghirv][^ ]*)?) \\2((?:-[ghirv][^ ]*)?) \\1\\3\\4$', form ) == None and re.search( '^(.*)-b{(.*)}((?:-[ghirv][^ ]*)?) \\2((?:-[ghirv][^ ]*)?) \\1\\3\\4$', form ) == None:
      sys.stdout.write( 'WARNING: Bad category in ' + t.c + ' -> ' + t.ch[0].c + ' ' + t.ch[1].c + '\n' )
    elif '-lM' in t.ch[0].c and getLocalArity(t.ch[1].c) != getLocalArity(t.c):  sys.stdout.write( 'ERROR: Bad arity in ' + t.c + ' -> ' + t.ch[0].c + ' ' + t.ch[1].c + '\n' )
    elif '-lM' in t.ch[1].c and getLocalArity(t.ch[0].c) != getLocalArity(t.c):  sys.stdout.write( 'ERROR: Bad arity in ' + t.c + ' -> ' + t.ch[0].c + ' ' + t.ch[1].c + '\n' )
    ## In-situ ops...
    if   '-lD' in t.ch[0].c or t.ch[0].c[0] in ',;:.!?':  output = translate( t.ch[1], Scopes, Anaphs, lsNolo )
    elif '-lD' in t.ch[1].c or t.ch[1].c[0] in ',;:.!?':  output = translate( t.ch[0], Scopes, Anaphs, lsNolo )
    elif '-lA' in t.ch[0].c or '-lU' in t.ch[0].c:  output = [ translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] ), translate( t.ch[0], Scopes, Anaphs, lsNolo[:m] ) ]
    elif '-lA' in t.ch[1].c or '-lU' in t.ch[1].c:  output = [ translate( t.ch[0], Scopes, Anaphs, lsNolo[:m] ), translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] ) ]
    elif '-lI' in t.ch[1].c and getLocalArity(t.ch[1].c) == 0:  output = [ translate( t.ch[0], Scopes, Anaphs, lsNolo[:m] ), [        '\\q', '\\r', '\\s', 'q', Univ, [ '\\x'+t.ch[0].sVar, translate( t.ch[1], Scopes, Anaphs, [ ['Trace','x'+t.ch[0].sVar] ] + lsNolo[m:] ),      'r', 's' ] ] ]
    elif '-lI' in t.ch[1].c and getLocalArity(t.ch[1].c) == 1:  output = [ translate( t.ch[0], Scopes, Anaphs, lsNolo[:m] ), [ '\\p', '\\q', '\\r', '\\s', 'p', Univ, [ '\\x'+t.ch[0].sVar, translate( t.ch[1], Scopes, Anaphs, [ ['Trace','x'+t.ch[0].sVar] ] + lsNolo[m:] ), 'q', 'r', 's' ] ] ]
#    elif '-lI' in t.ch[0].c and getLocalArity(t.ch[1].c) == 0:  output = [ '\\q', '\\r', '\\s', 'q', Univ, [ '\\x'+t.ch[0].sVar, translate( t.ch[0], Scopes, Anaphs, lsNolo[:m] ), [ translate( t.ch[1], Scopes, Anaphs, [ ['Trace','x'+t.ch[0].sVar] ] + lsNolo[m:] ),         'r', 's' ] ] ]  # [ translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] ), [ 'SelfStore', 'x'+t.ch[1].sVar, translate( t.ch[0], Scopes, Anaphs, [ ['Trace','x'+t.ch[1].sVar] ] + lsNolo[:m] ) ] ]
#    elif '-lI' in t.ch[0].c and getLocalArity(t.ch[1].c) == 1:  output = [ '\\q', '\\r', '\\s', 'q', Univ, [ '\\x'+t.ch[0].sVar, translate( t.ch[0], Scopes, Anaphs, lsNolo[:m] ), [ translate( t.ch[1], Scopes, Anaphs, [ ['Trace','x'+t.ch[0].sVar] ] + lsNolo[m:] ), 'Some', 'r', 's' ] ] ]  # [ translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] ), [ 'SelfStore', 'x'+t.ch[1].sVar, translate( t.ch[0], Scopes, Anaphs, [ ['Trace','x'+t.ch[1].sVar] ] + lsNolo[:m] ) ] ]
#    elif '-lI' in t.ch[1].c:  output = [ translate( t.ch[0], Scopes, Anaphs, lsNolo[:m] ), [ 'SelfStore', 'x'+t.ch[0].sVar, translate( t.ch[1], Scopes, Anaphs, [ ['Trace','x'+t.ch[0].sVar] ] + lsNolo[m:] ) ] ]
    elif '-lM' in t.ch[0].c:  output = [ 'Mod'+str(getLocalArity(t.ch[1].c)), translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] ), translate( t.ch[0], Scopes, Anaphs, lsNolo[:m] ) ]
    elif '-lM' in t.ch[1].c:  output = [ 'Mod'+str(getLocalArity(t.ch[0].c)), translate( t.ch[0], Scopes, Anaphs, lsNolo[:m] ), translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] ) ]
    elif '-lC' in t.ch[0].c:  output = [ 'And'+str(getLocalArity(t.ch[0].c)), translate( t.ch[0], Scopes, Anaphs, lsNolo ), translate( t.ch[1], Scopes, Anaphs, lsNolo ) ]
    elif '-lC' in t.ch[1].c:  output = translate( t.ch[1], Scopes, Anaphs, lsNolo )
    elif '-lG' in t.ch[0].c:  output = [ translate( t.ch[1], Scopes, Anaphs, [ translate( t.ch[0], Scopes, Anaphs, lsNolo[:m] ) ] + lsNolo[m:] ) ]
##    elif '-lG' in t.ch[0].c:  output = [ 'Store', 'x'+t.ch[0].sVar, translate( t.ch[0], Scopes, Anaphs, lsNolo[:m] ), translate( t.ch[1], Scopes, Anaphs, [ ['Trace','x'+t.ch[0].sVar] ] + lsNolo[m:] ) ]
#    elif '-lG' in t.ch[0].c and getLocalArity(t.ch[0].c)==0:  output = [ '\\r', '\\s', translate( t.ch[0], Scopes, Anaphs, lsNolo[:m] ), Univ, [ '\\x'+t.ch[0].sVar, translate( t.ch[1], Scopes, Anaphs, [ ['Trace','x'+t.ch[0].sVar] ] + lsNolo[m:] ), 'r', 's' ] ]
#    elif '-lG' in t.ch[0].c and getLocalArity(t.ch[0].c)==1:  output = [ translate( t.ch[1], Scopes, Anaphs, [ translate( t.ch[0], Scopes, Anaphs, lsNolo[:m] ) ] + lsNolo[m:] ) ]
    elif '-lH' in t.ch[1].c and getNoloArity(t.ch[1].c)==1:  output = [ 'Store', 'x'+t.ch[1].sVar, [ 'SelfStore', 'x'+t.ch[0].sVar, translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] + [ ['Trace','x'+t.ch[0].sVar] ] ) ],
                                                                        translate( t.ch[0], Scopes, Anaphs, [ ['Trace','x'+t.ch[1].sVar] ] + lsNolo[:m] ) ]
    elif '-lH' in t.ch[1].c:  output = [ translate( t.ch[0], Scopes, Anaphs, [ translate( t.ch[1], Scopes, Anaphs, lsNolo[m-1:] ) ] + lsNolo[:m-1] ) ]
##    elif '-lH' in t.ch[1].c:  output = [ 'Store', 'x'+t.ch[1].sVar, translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] ), translate( t.ch[0], Scopes, Anaphs, [ ['Trace','x'+t.ch[1].sVar] ] + lsNolo[:m] ) ]
#    elif '-lH' in t.ch[1].c and getLocalArity(t.c)==1:  output = [ '\\q', '\\r', '\\s', translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] ), Univ, [ '\\x'+t.ch[1].sVar, translate( t.ch[0], Scopes, Anaphs, [ ['Trace','x'+t.ch[1].sVar] ] + lsNolo[:m] ), 'q', 'r', 's' ] ]
#    elif '-lH' in t.ch[1].c and getLocalArity(t.c)==0:  output = [        '\\r', '\\s', translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] ), Univ, [ '\\x'+t.ch[1].sVar, translate( t.ch[0], Scopes, Anaphs, [ ['Trace','x'+t.ch[1].sVar] ] + lsNolo[:m] ),      'r', 's' ] ]
    ## Relative clause modification by C-rN-lR...
    elif '-lR' in t.ch[1].c and getLocalArity(t.c)==0 and getLocalArity(t.ch[1].c)==0:  output = [        '\\r', '\\s', translate( t.ch[0], Scopes, Anaphs, lsNolo[:m]   ),      'r', [ '\\x'+t.ch[0].sVar, '^', [ translate( t.ch[1], Scopes, Anaphs, [ ['Trace','x'+t.ch[0].sVar] ] + lsNolo[m:]   ), Univ, Univ ], ['s','x'+t.ch[0].sVar] ] ]
    ## Relative clause modification by I-aN-gN-lR (e.g. 'a job to do _') -- event should be in future...
    elif '-lR' in t.ch[1].c and getLocalArity(t.c)==0 and getLocalArity(t.ch[1].c)==1:  output = [        '\\r', '\\s', translate( t.ch[0], Scopes, Anaphs, lsNolo[:m]   ),      'r', [ '\\x'+t.ch[0].sVar, '^', [ translate( t.ch[1], Scopes, Anaphs, [ ['Trace','x'+t.ch[0].sVar] ] + lsNolo[m:]   ), 'Some', Univ, Univ ], ['s','x'+t.ch[0].sVar] ] ]
    ## Relative clause modification of verb phrase...
    elif '-lR' in t.ch[1].c and getLocalArity(t.c)==1:  output = [ '\\q', '\\r', '\\s', translate( t.ch[0], Scopes, Anaphs, lsNolo[:m]   ), 'q', 'r', [ '\\x'+t.ch[0].sVar, '^', [ translate( t.ch[1], Scopes, Anaphs, [ ['Trace','x'+t.ch[0].sVar] ] + lsNolo[m:]   ), Univ, Univ ], ['s','x'+t.ch[0].sVar] ] ]
    ## Relative clause modification of complete phrase or clause...
    elif '-lR' in t.ch[0].c and getLocalArity(t.c)==0:  output = [        '\\r', '\\s', translate( t.ch[1], Scopes, Anaphs, lsNolo[m-1:] ),      'r', [ '\\x'+t.ch[1].sVar, '^', [ translate( t.ch[0], Scopes, Anaphs, [ ['Trace','x'+t.ch[1].sVar] ] + lsNolo[:m-1] ), Univ, Univ ], ['s','x'+t.ch[1].sVar] ] ]
#    elif '-lR' in t.ch[0].c:  output = [ 'Mod'+str(getLocalArity(t.ch[1].c)), translate( t.ch[1], Scopes, Anaphs, lsNolo ), translate( t.ch[0], Scopes, Anaphs, [ ['Trace','x'+t.ch[1].sVar] ] ) ]
#    elif '-lR' in t.ch[1].c:  output = [ 'Mod'+str(getLocalArity(t.ch[0].c)), translate( t.ch[0], Scopes, Anaphs, lsNolo ), translate( t.ch[1], Scopes, Anaphs, [ ['Trace','x'+t.ch[0].sVar] ] ) ]
    else:  print( '\nERROR: unhandled rule from ' + t.c + ' to ' + t.ch[0].c + ' ' + t.ch[1].c )
    ## Propagate child stores...
    t.qstore += (t.ch[0].qstore if hasattr(t.ch[0],'qstore') else []) + (t.ch[1].qstore if hasattr(t.ch[1],'qstore') else [])

  ## 2.e. Fail...
  else: print( '\nERROR: too many children in ', t )

  ## 3. Mark anaphora...
  if ANAPH and t.bMax and t.sVar in Anaphs:
    output = [ 'Anaphor', Anaphs[t.sVar], output ]
  if ANAPH and t.bMax and t.sVar in Anaphs.values():
    output = [ 'Antecedent', t.sVar, output ]

  ## 4. Retrieve quantifier...
  if VERBOSE: print( ' '*indent, 'cat and scopes:', t.c, Scopes )
  if VERBOSE: print( ' '*indent, 'quant store: ', t.qstore )
  ## If not stored but can scope in situ, remove from scopes and carry on translating...
  if t.bMax and t.sVar in Scopes and t.sVar not in Scopes.values():
    del Scopes[ t.sVar ]
  ## If stored and can scope in situ, remove from scopes and carry on translating...
  if t.aboveAllInSitu and getLocalArity(t.c) == 0:
    while len(t.qstore) > 0:
      l = [ r for r in t.qstore if r[2] not in Scopes.values() ]
      if len(l) > 0:
        if VERBOSE: print( ' '*indent, 'retrieving:', l[0] )
        output = [ '\\r'+l[0][2], '\\s'+l[0][2], l[0][1], [ '\\x'+l[0][2], 'True' ], [ '\\x'+l[0][2], output, 'r'+l[0][2], 's'+l[0][2] ] ]
        del Scopes[ l[0][2] ]
        t.qstore += l[0][0]
        t.qstore.remove( l[0] )
        ## If not stored but can scope in situ, remove from scopes and carry on translating...
        if t.bMax and t.sVar in Scopes and t.sVar not in Scopes.values():
          del Scopes[ t.sVar ]
      else:
        break

  ## 5. If scoped and cannot be in situ, store...
  if t.bMax and t.sVar in Scopes and t.sVar in Scopes.values():
    t.qstore = [( t.qstore, output, t.sVar )]
    output = [ 'RaiseTrace', 'x'+t.sVar ]

  if VERBOSE: print( ' '*indent, 'returning:', output )
  indent -= 2
  return( output )


################################################################################
##
##  II. HELPER FUNCTIONS FOR MACRO SUBSTITUTION
##
################################################################################

########################################
#
#  II.A. Replace constants with lambda functions...
#
########################################

Univ = [ '\\z', 'True' ]

def unpack( expr ):
  if not isinstance( expr, str ):  return([ unpack(subexpr) for subexpr in expr ])
  elif expr == 'And0':  return( [ '\\f', '\\g',               '\\r', '\\s', '^', [ 'g',           'r', 's' ], [ 'f',           'r', 's' ] ] )
  elif expr == 'And1':  return( [ '\\f', '\\g',        '\\q', '\\r', '\\s', '^', [ 'g',      'q', 'r', 's' ], [ 'f',      'q', 'r', 's' ] ] )
  elif expr == 'And2':  return( [ '\\f', '\\g', '\\p', '\\q', '\\r', '\\s', '^', [ 'g', 'p', 'q', 'r', 's' ], [ 'f', 'p', 'q', 'r', 's' ] ] )
  elif expr == 'Mod0':  return( [ '\\f', '\\g',               '\\r', '\\s', 'f',           [ '\\x', '^', ['r', 'x' ], [ 'g', [ '\\t', '\\u', '^', ['t','x'], ['u','x'] ], Univ, Univ ] ], 's' ] )
  elif expr == 'Mod1':  return( [ '\\f', '\\g',        '\\q', '\\r', '\\s', 'f',      'q', [ '\\x', '^', ['r', 'x' ], [ 'g', [ '\\t', '\\u', '^', ['t','x'], ['u','x'] ], Univ, Univ ] ], 's' ] )
  elif expr == 'Mod2':  return( [ '\\f', '\\g', '\\p', '\\q', '\\r', '\\s', 'f', 'p', 'q', [ '\\x', '^', ['r', 'x' ], [ 'g', [ '\\t', '\\u', '^', ['t','x'], ['u','x'] ], Univ, Univ ] ], 's' ] )
  elif expr == 'Prop':  return( [ '\\p', '\\q', '\\r', '\\s', 'q', Univ, [ '\\x', 'p', Univ, ['\\y','Equal','y','x'] ] ] )
#  elif expr == 'Pasv':  return( [ '\\q', '\\r', '\\s', 'q', Univ, [ '\\x', 'p', Univ, ['\\y','Equal','y','x'] ] ] )
  elif expr == 'Ident':  return( [ '\\f', 'f' ] )
  elif expr == 'IdentSome':  return( [ '\\f', 'f', 'Some' ] )
#  elif expr == 'Store':  return( [ '\\n', '\\f', '\\q', '\\r', '\\s', 'q', Univ, ['\\x','f','r','s'] ] )
  elif expr == 'Trace':  return( [ '\\v', '\\t', '\\u', '^', ['t','v'], ['u','v'] ] )
  elif expr == 'RaiseTrace':  return( [ '\\v', '\\t', '\\u', '^', ['t','v'], ['u','v'] ] )
#  elif expr == 'SelfStore':  return( [ '\\v', '\\q', '\\r', '\\s', 'q', Univ, [ '\\x'+?????
  elif re.search( '^@N(-[lmnstuxyz].*)?:', expr ) != None:  return( [ '\\r', '\\s', 'Some', [ '\\z', '^', [ 'Some', [ '\\e', expr[1:],'e','z' ], Univ ], ['r','z'] ], 's' ] )
  elif expr.split(':')[0] == '@N-aD':  return( [ '\\q', '\\r', '\\s', 'Some', [ '\\zz', '^', [ 'Some', [ '\\e', expr[1:],'e','zz' ], Univ ], ['r','zz'] ], 's' ] )
  elif expr.split(':')[0] == '@N-b{N-aD}':  return( [ '\\f', '\\r', '\\s', expr[1:], [ '\\x', '^', ['r','x'], ['f','Some',['\\xx','Equal','xx','x'],Univ] ], 's' ] )
  elif expr.split(':')[0] == '@N-aD-b{N-aD}':  return( [ '\\f', '\\q', '\\r', '\\s', expr[1:], [ '\\x', '^', ['r','x'], ['f','q',['\\xx','Equal','xx','x'],Univ] ], 's' ] )
#  elif expr.split(':')[0] == '@B-aN-b{A-aN}':  return( [ '\\f', '\\q', '\\r', '\\s', 'f', 'q', [ '\\e', '^', [expr[1:],'e'], ['r','e'] ], 's' ] )
#  elif expr.split(':')[0] == '@B-aN-b{B-aN}':  return( [ '\\f', '\\q', '\\r', '\\s', 'f', 'q', [ '\\e', '^', [expr[1:],'e'], ['r','e'] ], 's' ] )
#  elif expr.split(':')[0] == '@I-aN-b{B-aN}':  return( [ '\\f', '\\q', '\\r', '\\s', 'f', 'q', [ '\\e', '^', [expr[1:],'e'], ['r','e'] ], 's' ] )
  elif expr.split(':')[0] == '@N-rN':  return( [ '\\q', '\\r', '\\s', 'q', Univ, [ '\\x', 'Some', [ '\\e', '^', [expr[1:],'e','x'], ['r','e'] ], 's' ] ] )
  elif expr.split(':')[0] == '@A-aN-rN':  return( [ '\\p', '\\q', '\\r', '\\s', 'q', Univ, [ '\\x', 'p', Univ, [ '\\y', 'Some', [ '\\e', '^', [expr[1:],'e','x','y'], ['r','e'] ], 's' ] ] ] )
#  elif expr.split(':')[0] == '@B-aN-bA':  return( [ '\\p', '\\q', '\\r', '\\s', 'q', Univ, [ '\\x', 'Some', [ '\\e', '^', [ expr[1:], 'e', 'x', [ 'Intension', [ 'p', Univ, Univ ] ] ], ['r','e'] ], 's' ] ] )
  ## Intransitive...
  elif re.search( '^@[A-Za-z0-9]+-[ab][A-Za-z]*(-[lx].*)?:', expr ) != None:
    return( [               '\\q', '\\r', '\\s', 'q', Univ, [ '\\x',                                        'Some', [ '\\e', '^', [expr[1:],'e','x'        ], ['r','e'] ], 's'     ] ] )
  ## Transitive...
  elif re.search( '^@[A-Za-z0-9]+-[ab][A-Za-z]+-[ab][DNOPa-z]+(-[lx].*)?:', expr ) != None:
    return( [        '\\p', '\\q', '\\r', '\\s', 'q', Univ, [ '\\x', 'p', Univ, [ '\\y',                    'Some', [ '\\e', '^', [expr[1:],'e','x','y'    ], ['r','e'] ], 's'   ] ] ] )
  ## Ditransitive...
  elif re.search( '^@[A-Za-z0-9]+-[ab][A-Za-z]+-[ab][A-Za-z]+-[ab][A-Za-z]+(-[lx].*)?:', expr ) != None:
    return( [ '\\o', '\\p', '\\q', '\\r', '\\s', 'q', Univ, [ '\\x', 'p', Univ, [ '\\y', 'o', Univ, ['\\z', 'Some', [ '\\e', '^', [expr[1:],'e','x','y','z'], ['r','e'] ], 's' ] ] ] ] )
  ## Raising...
  elif re.search( '^@[A-Za-z0-9]+-[ab][A-Za-z]+-[ab]\{[A-Za-z]+-[ab][A-Za-z]+\}(-[lx].*)?:', expr ) != None:
    return( [        '\\f', '\\q', '\\r', '\\s',                                                          'f', 'q', [ '\\e', '^', [expr[1:],'e'            ], ['r','e'] ], 's'       ] )
  ## Sent comp...
  elif re.search( '^@[A-Za-z0-9]+-[ab][A-Za-z]+-[ab][ABCEFGIVQRSVa-z]+(-[lx].*)?:', expr ) != None:
    return( [        '\\p', '\\q', '\\r', '\\s', 'q', Univ, [ '\\x', 'Some', [ '\\e', '^', [ expr[1:], 'e', 'x', [ 'Intension', [ 'p', Univ, Univ ] ] ], ['r','e'] ], 's' ] ] )
  ## Tough constructions...
  elif re.search( '^@[A-Za-z0-9]+-[ab][A-Za-z]+-[ab]{I-aN-gN}(-[lx].*)?:', expr ) != None:
    return( [        '\\f', '\\q', '\\r', '\\s', 'q', Univ, [ '\\x', 'Gen', [ '\\e', 'f', [ '\\t', '\\u', '^', ['t','x'], ['u','x'] ], 'Some', 'r', [ '\\d', '^', ['s','d'], ['Equal','d','e'] ] ], [ '\\e',expr[1:],'e'] ] ] )
  ## Embedded question...
  elif re.search( '^@[A-Za-z0-9]+-[ab][A-Za-z]+-[ab]{V-iN}(-[lx].*)?:', expr ) != None:
    return( [        '\\f', '\\q', '\\r', '\\s', 'q', Univ, [ '\\x', 'f', [ '\\t', '\\u', '^', ['t','z'], ['u','z'] ], Univ, [ '\d', 'Some', 'r', ['\\e', '^', [expr[1:],'e','x','d'], ['s','d'] ] ] ] ] )
  ## Bare relatives (bad take as eventuality of V is not constrained by wh-noun)...
  elif re.search( '^@[A-Za-z0-9]+-[ab]{V-gN}(-[lx].*)?:', expr ) != None:
    return( [        '\\f',        '\\r', '\\s', 'Some', [ '\\z', '^', [ '^', [ 'Some', ['\\e',expr[1:],'e','z'], Univ ], ['r','z'] ], [ 'f', [ '\\t', '\\u', '^', ['t','z'], ['u','z'] ], Univ, Univ ] ], 's' ] )
#  elif expr[0]=='@' and getLocalArity( t.split(':')[0] ) == 1:  return( [        '\\q', '\\r', '\\s', 'q', Univ, [ '\\x',                     'Some', [ '\\e', '^', [expr[1:],'e','x'    ], ['r','e'] ], 's'   ] ] )
#  elif expr[0]=='@' and getLocalArity( t.split(':')[0] ) == 2:  return( [ '\\p', '\\q', '\\r', '\\s', 'q', Univ, [ '\\x', 'p', Univ, [ '\\y', 'Some', [ '\\e', '^', [expr[1:],'e','x','y'], ['r','e'] ], 's' ] ] ] )
  else:  return( expr )


########################################
#
#  II.B. Replace in beta reduce...
#
########################################

def replace( expr, old, new ):
#  if VERBOSE: print( 'replacing:', old, 'with', new, 'in', expr )
  if expr == old:
    return( new )
  elif isinstance( expr, str ):
    return( expr )
  elif any( [ subexpr[0]=='\\' and subexpr[1:]==old for subexpr in expr ] ):
    return( expr )
  else:
    return( [ replace( subexpr, old, new ) for subexpr in expr ] )


########################################
#
#  II.C. Beta reduce...
#
########################################

def betaReduce( expr ):
  if VERBOSE: print( 'reducing:', expr )
  ## If string, skip...
  if isinstance(expr,str):
    return
  ## Find first non-lambda
  for i in range(len(expr)):
    if expr[i][0]!='\\':
      break
  if VERBOSE: print( 'i =', i )
  ## If initial term is string, betaReduce children...
  if isinstance(expr[i],str):
    for subexpr in expr:
      betaReduce( subexpr )
  ## Flatten initial application... 
  elif expr[i][0][0]!='\\':
    expr[:] = expr[:i] + expr[i] + expr[i+1:]
    betaReduce( expr )
  ## Substitute second term for initial lambda variable of initial (abstraction) term...
  elif len(expr) > i+1:
    expr[:] = expr[:i] + [ replace( expr[i][1:], expr[i][0][1:], expr[i+1] ) ] + expr[i+2:]
    betaReduce( expr )
  else:
    expr[:] = expr[:i] + expr[i]
    betaReduce( expr )


########################################
#
#  II.D. Conjunction elimination...
#
########################################

def simplify( expr ):

  if isinstance( expr, str ):  return

  ## Eliminate conjunctions with tautology...
  if len(expr)==3 and expr[0]=='^' and expr[1]==['True']:
    expr[:] = expr[2]
    simplify( expr )
  elif len(expr)==3 and expr[0]=='^' and expr[2]==['True']:
    expr[:] = expr[1]
    simplify( expr )
  elif len(expr)==4 and expr[0][0]=='\\' and expr[1]=='^' and expr[2]==['True']:
    expr[:] = [ expr[0], expr[3] ]
    simplify( expr )
  elif len(expr)==4 and expr[0][0]=='\\' and expr[1]=='^' and expr[3]==['True']:
    expr[:] = [ expr[0], expr[2] ]
    simplify( expr )

  ## Eliminate existentials with conjunctions with equality...
#  if expr[0]=='Some': print( expr )
  if expr[0]=='Some' and expr[2][1]=='^' and expr[2][3][0]=='Equal' and expr[2][3][1]==expr[2][0][1:]:
#    print( 'input:', expr )
    expr[:] = [ '^', replace( expr[1][1:], expr[1][0][1:], expr[2][3][2] ), replace( expr[2][1:], expr[2][0][1:], expr[2][3][2] ) ]
#    print( 'output:', expr )
#   if 'Some' in expr:
#     print( expr )
#     ## Each of restrictor and nuclear scope sets...
#     for i in range( len(expr)-2, len(expr) ):
#       if expr[i][0][0] == '\\' and expr[i][1] == '^':
#         ## Each conjunct...
#         for j in range( 2, len(expr[i]) ):
#           if expr[i][j][0] == 'Equal' and expr[i][0][1:] == expr[i][j][1]:
#             print( 'input:', expr )
#             k = j-1 if j==len(expr[i])-1 else j+1
#             expr[:] = expr[:-3] + replace( expr[i][k][1:], expr[i][0][1:], expr[i][j][2] )
#             print( 'output:', expr )
#             break

  ## Recurse...
  else:
    for subexpr in expr:
      simplify( subexpr )


################################################################################
##
##  III. MAIN LOOP
##
################################################################################

nArticle = -1

## For each article...
while True:

  nArticle += 1
  Scopes = {}
  Anaphs = {}
  Trees = []

#  if nArticle == 75: VERBOSE = True

  ## For each tree in article...
  for nLine,line in enumerate( sys.stdin ):

#    print( '========== Article ' + str(nArticle) + ' Tree ' + str(nLine) + ' ==========' )
#    print( line[:-1] )

    if '!ARTICLE' in line:  break

    t = tree.Tree()
    t.read( line )
    Trees += [ t ]
    setHeadScopeAnaph( t, nLine, Scopes, Anaphs )
    markSites( t, Scopes )

  ## Skip articles...
  if nArticle in SKIPS:
    print( 'NOTE: skipping article', nArticle, 'as specified in command line arguments.' )
    continue

  ## Process trees given anaphs...
  for nLine,t in enumerate( Trees ):

#    sys.stderr.write( '========== Article ' + str(nArticle) + ' Tree ' + str(nLine) + ' ==========\n' )
    print( '========== Article ' + str(nArticle) + ' Tree ' + str(nLine) + ' ==========' )
    print( t )
  
    print( '----------' )
    if VERBOSE: print( 'Scopes', Scopes )
    shortExpr = translate( t, Scopes, Anaphs )
    if t.qstore != []:
      print( 'ERROR: nothing in quant store', t.qstore, 'allowed by scope list', Scopes )
      exit(0)
    print( shortExpr )
  
    print( '----------' )
    fullExpr = [ unpack(shortExpr), Univ, Univ ]
    print( fullExpr )

    print( '----------' )
    betaReduce( fullExpr )
    print( fullExpr )

    print( '----------' )
    simplify( fullExpr )
    print( fullExpr )

  if '!ARTICLE' not in line:
    break


