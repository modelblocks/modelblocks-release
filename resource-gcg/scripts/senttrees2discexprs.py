import sys
import re
import copy
import tree
import lex

## Flags mutable by command-line arguments...
VERBOSE = False    ## print debugging info.
ANAPH = True       ## apply anaphor expansions.
SKIPS = []         ## skip articles.
ONLY = -1          ## print debugging info only for certain articles.

## Process command-line arguments...
for a in sys.argv:
  if a=='-d':  VERBOSE = True
  if a=='-n':  ANAPH = False
  if a[:2]=='-s':  SKIPS += [ int( a[2:] ) ]
  if a[:2]=='-o':  ONLY = int( a[2:] )

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
#  cat = re.sub( '-x[^-} ].*', '', cat )
  while '{' in cat and '}' in cat:
    cat = re.sub('\{[^\{\}]*\}','X',cat)
  return len(re.findall('-[ghirv]',cat))


def getLocalArity( cat ):
#  cat = re.sub( '-x[^-} ].*', '', cat )
  while '{' in cat and '}' in cat:
    cat = re.sub('\{[^\{\}]*\}','X',cat)
  return len(re.findall('-[ab]',cat))


########################################
#
#  I.B. set scopes and variable numbers...
#
########################################

def setHeadScopeAnaph( t, nSent, Scopes, Anaphs, nWord=0 ):

  ## Flush '-x' singletons as pre-process...
  t.c = re.sub( '-x([-} ])', '\\1', t.c )
  ## Flush equations as pre-process...
  t.c = re.sub( '^(.*)-x.*?=.*?(-[mnstuw].*)?$', '\\1\\2', t.c )

  ## Recurse...
  for st in t.ch:
    nWord = setHeadScopeAnaph( st, nSent, Scopes, Anaphs, nWord )

  ## Account head words as done in gcg annotation guidelines, in order to track scope...
  if len(t.ch) == 0:
    nWord += 1
    t.sVar = str(nSent*100+nWord)
  elif len(t.ch) == 1:
    t.sVar = t.ch[0].sVar
    if t.c[:3] == 'V-g' and ( t.ch[0].c == 'N-lE' or t.ch[0].c == 'R-aN-lE' ): t.sVar += '?'  ## Special case for function extraction.
    if '-lZ' in t.ch[0].c: t.sVar += '?'  ## Special case for zero-head rule.
  elif len(t.ch) == 2:
    t.sVar = t.ch[0].sVar if '-lU' in t.ch[0].c else t.ch[1].sVar if '-lU' in t.ch[1].c else t.ch[0].sVar if '-l' not in t.ch[0].c and len(t.ch[0].c)>0 and t.ch[0].c[0] not in ',;' else t.ch[1].sVar if '-l' not in t.ch[1].c else None
    if t.sVar == None: print( 'ERROR: Illegal categories in ', t )  #, t.c, '->', t.ch[0].c, t.ch[1].c )
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

  if VERBOSE: print( 'head:', t.sVar, t )

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
#    if '=' in t.c:
#      print( 'pre-empting', t.c )
#      t.c = re.sub( '-x[^x]*=.*', '', t.c )  #re.sub( '-x[^-} ].*', '', t.c )  ## Pre-empt any equations.
#      print( 'pre-empted', t.c )
    pred = lex.getFn( t.c, t.ch[0].c )
    if   pred == '' and t.c == 'N-b{N-aD}-x%|':        output = 'IdentSome'
    elif pred == '' and t.c == 'Ne-x%|':               output = 'Some'
    elif pred == '' and re.search( '^P[a-z]+', t.c ):  output = 'Some'
    elif pred == '':                                   output = 'Ident'
    elif re.match( '^.*-[ri]\w+(-[lmnstuwxy].*)?$', t.c ) != None:
      if lsNolo == []:  print( 'ERROR: missing non-local argument in', t )
      output = [ '@'+pred+'@'+t.sVar, lsNolo[-1] ]
    elif lsNolo != []:  print( 'WARNING: un-extracted non-local argument', lsNolo, 'in', t )
#    elif t.c == 'R-aN-rN-xR%|A%':                output = [ '@'+pred+'@'+t.sVar, lsNolo[-1] ]
#    elif t.c == 'N-rN':                          output = [ '@'+pred+'@'+t.sVar, lsNolo[-1] ]
    else:  output = '@'+pred+'@'+t.sVar
#    output = '@'+pred if pred != '' else ('IdentSome' if t.c == 'N-b{N-aD}-x%|'  else  'Some' if t.c == 'Ne-x%|'  else 'Ident')

  ## 2.c. Unary branch...
  elif len(t.ch) == 1:
#    form = re.sub( '-[lmnstuwxy][^ ]*', '', t.ch[0].c + ' ' + t.c )
    form = re.sub( '-[mnstuwxy][^ ]*', '', re.sub('-x([-} ])','\\1',t.ch[0].c) ) + ' ' + re.sub( '-[lx][^ ]*', '', t.c )
#    print( 'form:', form )
    bform = re.sub( '-[lx][^ ]*', '', t.c ) + ' ' + re.sub( '-[mnstuwxy][^ ]*', '', t.ch[0].c )
#    if   '-lF' in t.ch[0].c and re.search( '^V-iN((?:-[ghirv][^ ]*)?) N\\1$', form ) != None:  output = [ '\\r', '\\s', 'All', [ '\\x'+t.sVar, '^', ['r','x'+t.sVar], translate( t.ch[0], Scopes, Anaphs, [ ['Trace','x'+t.sVar] ] + lsNolo ) ], 's' ]
    ## Extractions / non-local introduction...
    if '-lE' in t.ch[0].c:
      if lsNolo == []: print( 'ERROR: no non-locals in', t )
      ## E5. Argument extraction with zero-head rule applied to filler: e.g. V-aN-gN -> V-aN-b{A-aN}-lE...
      if re.search( '^(.*)-[ab]\{\w+-[ab]\w+\}((?:-[ghirv][^ ]*)?)-lE \\1-[ghriv]\w+\\2$', form ):  output = [ translate( t.ch[0], Scopes, Anaphs, lsNolo[1:] ), [ 'Pred', lsNolo[0] ] ]
      ## E4. Function extraction: e.g. V-g{V-aN} -> N-lE...
      elif re.search( '^(.*)((?:-[ghirv][^ ]*)?)-lE (\w+)-[ghirv]\{\\3-[abghirv](\\1|\{\\1\})\}\\2$', form ):  output = [ lsNolo[0], translate( t.ch[0], Scopes, Anaphs, lsNolo[1:] ) ]
      ## E3. Function extraction: e.g. V-gV -> R-aN-lE...
      elif re.search( '^(.*)((?:-[ghirv][^ ]*)?)-lE (\w+)-[ghirv]\\3\\2$', form ):  output = [ 'Mod'+str(getLocalArity(t.c)), lsNolo[0], translate( t.ch[0], Scopes, Anaphs, lsNolo[1:] ) ]
      ## E1. Argument extraction with exact match: e.g. V-aN-gN -> V-aN-bN-lE... 
      elif re.search( '^(.*)-[ab]([^ ]+)((?:-[ghirv][^ ]*)?)-lE \\1-[ghirv]\\2\\3$', form ):  output = [ translate( t.ch[0], Scopes, Anaphs, lsNolo[1:] ), lsNolo[0] ]
      ## E2. Modifier extraction: e.g. V-aN-g{R-aN} -> V-aN-lE... 
      elif re.search( '^(.*)((?:-[ghirv][^ ]*)?)-lE \\1-[ghirv]\{[^ ]*\}\\2$', form ):  output = [ 'Mod'+str(getLocalArity(t.ch[0].c)), translate( t.ch[0], Scopes, Anaphs, lsNolo[1:] ), lsNolo[0] ]
#      elif '-lE' in t.ch[0].c and not( re.search( '^(.*)-[ab]([^ ]+)((?:-[ghirv][^ ]*)?)-lE \\1-[ghirv]\\2\\3$', form ) ) and not( re.search( '^(.*)((?:-[ghirv][^ ]*)?)-lE \\1-[ghirv]\{[^ ]*\}\\2$', form ) ):
#      elif '-lE' in t.ch[0].c and getLocalArity(t.c) + 1 == getLocalArity(t.ch[0].c):  output = [ translate( t.ch[0], Scopes, Anaphs, lsNolo[1:] ), lsNolo[0] ]
#      elif '-lE' in t.ch[0].c and getLocalArity(t.c) == getLocalArity(t.ch[0].c):  output = [ 'Mod'+str(getLocalArity(t.ch[0].c)), translate( t.ch[0], Scopes, Anaphs, lsNolo[1:] ), lsNolo[0] ]
      else: 
        print( 'ERROR: Ill-formed extraction:', t )  #, form ) # t.c, '->', t.ch[0].c )
        exit( 0 )
    ## M3. Unary modifier...
    elif re.search( '^A-aN((?:-[ghirv][^ ]*)?)-lM N-aD\\1$', form ):  output = [ 'Mod1', ['\\q','\\t','\\u','q','t','u'], translate( t.ch[0], Scopes, Anaphs, lsNolo ) ]
    ## O1-2. Reordering rules...
    elif re.search( '^\w+-a\w+-lQ \w+-a\w+-b\{\w+-[abghirv]\w+\}$', form ):  output = [ '\\f', '\\q', translate( t.ch[0], Scopes, Anaphs, lsNolo ), [ 'f', 'Some' ] ]
    elif '-lQ' in t.ch[0].c and getLocalArity(t.ch[0].c) == 1:  output = [ '\\p', '\\q', translate( t.ch[0], Scopes, Anaphs, lsNolo ), 'p' ]
    elif '-lQ' in t.ch[0].c and getLocalArity(t.ch[0].c) == 2:  output = [ '\\p', '\\q', translate( t.ch[0], Scopes, Anaphs, lsNolo ), 'q', 'p' ]
    ## V. Passive rule...
    elif re.search( '^L-aN-vN((?:-[ghirv][^ ]*)?)-lV [AR]-aN\\1$', form ) != None:  output = [ '\\q', '\\r', '\\s', 'q', Univ, [ '\\zz'+t.sVar, translate( t.ch[0], Scopes, Anaphs, [ ['Trace','zz'+t.sVar] ] + lsNolo ), 'Some', 'r', 's' ] ]
#   elif '-lV' in t.ch[0].c:  output = [ 'Pasv', 'x'+ t.sVar, translate( t.ch[0], Scopes, Anaphs, [ ['Trace','x'+t.sVar] ] + lsNolo ) ]
    ## Z3. Zero-head rule with expletive subject: e.g. A-aNe -> N-lZ...
    elif re.search( '^\w+-lZ \w+-[ab]Ne$', form ):  output = [ '\\q', translate( t.ch[0], Scopes, Anaphs, lsNolo ) ]
    ## Z1-2. ACCOMMODATE SLOPPY ANNOTATION of '-lZ' with arg elision...
    elif '-lZ' in t.ch[0].c and getLocalArity(t.c) == getLocalArity(t.ch[0].c):  output = [ 'Pred', [ translate( t.ch[0], Scopes, Anaphs, lsNolo ), 'Some' ] ]
    elif '-lZ' in t.ch[0].c and getLocalArity(t.c) == getLocalArity(t.ch[0].c) + 1:  output = [ 'Pred', translate( t.ch[0], Scopes, Anaphs, lsNolo ) ]
    elif '-lz' in t.ch[0].c and getLocalArity(t.c) == getLocalArity(t.ch[0].c) + 1:  output = [ 'Pred', translate( t.ch[0], Scopes, Anaphs, lsNolo ) ]
    ## T1. S -> Q-iN or N -> V-iN...
    elif re.search( '^\w+-i\w+(-lF)? \w+$', form ):  output = [ '\\r', '\\s', 'Gen', [ '\\zz'+t.sVar, translate( t.ch[0], Scopes, Anaphs, [ ['Trace','zz'+t.sVar] ] + lsNolo ), 'r', Univ ], 's' ]
                                                                                                        # [ '\\x'+t.sVar, 'Explain', 'ThisArticle', 'x'+t.sVar ] ]
    ## T2. V-rN -> V-g{R-aN}...
    elif '-l' not in t.ch[0].c and re.search( '^\w+-[ghirv]\{\w+-[abghirv]\w+\} \w+-[ghirv]\w+$', form ): output = translate( t.ch[0], Scopes, Anaphs, lsNolo[:-1] + [ [ 'Pred', lsNolo[-1] ] ] )
    ## T3. Elision...
    elif re.search( '^(.*)-[ab]\{\w+-[abghirv]\{\w+-[abghirv]\w+\}\} \\1$', form ):  output = [ translate( t.ch[0], Scopes, Anaphs, lsNolo ), [ '\\f', '\\r', '\\s', 'True' ] ]
    elif re.search( '^(.*)-[ab]\{\w+-[abghirv]\w+\} \\1$', form ):  output = [ translate( t.ch[0], Scopes, Anaphs, lsNolo ), [ '\\f', '\\r', '\\s', 'True' ] ]
    elif '-l' not in t.ch[0].c and getLocalArity(t.c) + 1 == getLocalArity(t.ch[0].c):  output = [ translate( t.ch[0], Scopes, Anaphs, lsNolo ), 'Some' ]
    elif '-l' not in t.ch[0].c and getLocalArity(t.c) == getLocalArity(t.ch[0].c):  output = translate( t.ch[0], Scopes, Anaphs, lsNolo )
    else:
      print( 'WARNING: Assuming simple type change in', t )    #', t.c, '->', t.ch[0].c, getLocalArity(t.c), getLocalArity(t.ch[0].c), 'is simple type change' )
      output = translate( t.ch[0], Scopes, Anaphs, lsNolo )
    ## Propagate child stores...
    t.qstore = t.ch[0].qstore

  ## 2d. Binary branch...
  elif len(t.ch) == 2:
    m = getNoloArity(t.ch[0].c)
    if VERBOSE: print( ' '*indent, 'child cats and nolos:', t.ch[0].c, t.ch[1].c, m, lsNolo[:m], lsNolo[m:] )
    ## Error check...
    form = re.sub( '-[lmnstuwxy][^ ]*', '', t.ch[0].c + ' ' + t.ch[1].c + ' ' + t.c )
    if   ('-lA' in t.ch[0].c or '-lU' in t.ch[0].c) and getLocalArity(t.ch[1].c) != getLocalArity(t.c)+1:  sys.stdout.write( 'ERROR: Bad arity in ' + str(t) + '\n' )  # + t.c + ' -> ' + t.ch[0].c + ' ' + t.ch[1].c + '\n' )
    elif ('-lA' in t.ch[0].c or '-lU' in t.ch[0].c) and re.search( '^(.*)((?:-[ghirv][^ ]*)?) (.*)-a\\1((?:-[ghirv][^ ]*)?) \\3\\2\\4$', form ) == None and re.search( '^(.*)((?:-[ghirv][^ ]*)?) (.*)-a{\\1}((?:-[ghirv][^ ]*)?) \\3\\2\\4$', form ) == None:
      sys.stdout.write( 'WARNING: Bad category in ' + str(t) + '\n' )  #+ t.c + ' -> ' + t.ch[0].c + ' ' + t.ch[1].c + '\n' )
    elif ('-lA' in t.ch[1].c or '-lU' in t.ch[1].c) and getLocalArity(t.ch[0].c) != getLocalArity(t.c)+1:  sys.stdout.write( 'ERROR: Bad arity in ' + str(t) + '\n' )  # + t.c + ' -> ' + t.ch[0].c + ' ' + t.ch[1].c + '\n' )
    elif ('-lA' in t.ch[1].c or '-lU' in t.ch[1].c) and re.search( '^(.*)-b(.*)((?:-[ghirv][^ ]*)?) \\2((?:-[ghirv][^ ]*)?) \\1\\3\\4$', form ) == None and re.search( '^(.*)-b{(.*)}((?:-[ghirv][^ ]*)?) \\2((?:-[ghirv][^ ]*)?) \\1\\3\\4$', form ) == None:
      sys.stdout.write( 'WARNING: Bad category in ' + str(t) + '\n' )  # + t.c + ' -> ' + t.ch[0].c + ' ' + t.ch[1].c + '\n' )
    elif '-lM' in t.ch[0].c and getLocalArity(t.ch[1].c) != getLocalArity(t.c):  sys.stdout.write( 'ERROR: Bad arity in ' + str(t) + '\n' )  # + t.c + ' -> ' + t.ch[0].c + ' ' + t.ch[1].c + '\n' )
    elif '-lM' in t.ch[1].c and getLocalArity(t.ch[0].c) != getLocalArity(t.c):  sys.stdout.write( 'ERROR: Bad arity in ' + str(t) + '\n' )  # + t.c + ' -> ' + t.ch[0].c + ' ' + t.ch[1].c + '\n' )
    ## D1-2,A1-2. In-situ ops...
    if   '-lD' in t.ch[0].c or t.ch[0].c[0] in ',;:.!?':  output = translate( t.ch[1], Scopes, Anaphs, lsNolo )
    elif '-lD' in t.ch[1].c or t.ch[1].c[0] in ',;:.!?':  output = translate( t.ch[0], Scopes, Anaphs, lsNolo )
    elif '-lA' in t.ch[0].c or '-lU' in t.ch[0].c:  output = [ translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] ), translate( t.ch[0], Scopes, Anaphs, lsNolo[:m] ) ]
    elif '-lA' in t.ch[1].c or '-lU' in t.ch[1].c:  output = [ translate( t.ch[0], Scopes, Anaphs, lsNolo[:m] ), translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] ) ]
    ## I1-2. Non-local elimination in argument: e.g. V-aN -> V-aN-b{I-aN-gN} I-aN-gN-lI...
    elif re.match( '^\w+-[ghriv]\{\w+-[ab]\w+\}-lI', t.ch[1].c ):  output = [ translate( t.ch[0], Scopes, Anaphs, lsNolo[:m] ), [ '\\fi', translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] + [ 'fi' ] ) ] ]
# ## THESE TWO DIDNT WORK AND ARENT NEEDED AND DONT MAKE SENSE ANYWAY
#    elif '-lI' in t.ch[1].c and re.match( '(\w+)-b{(\w+-[ghriv]\w+)} \\2 \\1', form ):
#      output = [ '\\r', '\\s', translate( t.ch[1], Scopes, Anaphs, [ [ '\\t'+t.ch[1].sVar, '\\u'+t.ch[1].sVar, translate( t.ch[0], Scopes, Anaphs, lsNolo[:m-1] ), [ '\\qq', '\\t', '\\u', 'qq', 't'+t.ch[1].sVar, 'u'+t.ch[1].sVar ], 'r', 's' ] ] + lsNolo[m-1:] ), Univ, Univ ]
#    elif '-lI' in t.ch[1].c and re.match( '(\w+-[ab]\w+)-b{(\w+-[ghriv]\w+)} \\2 \\1', form ):
#      output = [ '\\q', '\\r', '\\s', translate( t.ch[1], Scopes, Anaphs, [ [ '\\t'+t.ch[1].sVar, '\\u'+t.ch[1].sVar, translate( t.ch[0], Scopes, Anaphs, lsNolo[:m-1] ), [ '\\qq', '\\t', '\\u', 'qq', 't'+t.ch[1].sVar, 'u'+t.ch[1].sVar ], 'q', 'r', 's' ] ] + lsNolo[m-1:] ), Univ, Univ ]
#    elif re.match( '^\w+-[ghriv]\w+-lI', t.ch[1].c ):  output = [ translate( t.ch[0], Scopes, Anaphs, lsNolo[:m] ), [ '\\qi', 'r', 's', 'qi', Univ, [ '\\xi'+t.ch[1].sVar, translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] + [ ['Trace','xi'+t.ch[1].sVar] ] ), 'r', 's' ] ] ]
    elif re.match( '^\w+-[ghriv]\w+-lI', t.ch[1].c ):  output = [ translate( t.ch[0], Scopes, Anaphs, lsNolo[:m] ), [ '\\qi', translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] + [ 'qi' ] ) ] ]
#    elif '-lI' in t.ch[1].c and getLocalArity(t.ch[1].c) == 0:  output = [ translate( t.ch[0], Scopes, Anaphs, lsNolo[:m] ), [        '\\q', '\\r', '\\s', 'q', Univ, [ '\\zz'+t.ch[0].sVar, translate( t.ch[1], Scopes, Anaphs, [ ['Trace','zz'+t.ch[0].sVar] ] + lsNolo[m:] ),      'r', 's' ] ] ]
    elif '-lI' in t.ch[1].c and getLocalArity(t.ch[1].c) == 1:  output = [ translate( t.ch[0], Scopes, Anaphs, lsNolo[:m] ), [ '\\p', '\\q', '\\r', '\\s', 'p', Univ, [ '\\zz'+t.ch[0].sVar, translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] + [ ['Trace','zz'+t.ch[0].sVar] ] ), 'q', 'r', 's' ] ] ]
    ## M1-2. Modifier...
    elif '-lM' in t.ch[0].c:  output = [ 'Mod'+str(getLocalArity(t.ch[1].c)), translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] ), translate( t.ch[0], Scopes, Anaphs, lsNolo[:m] ) ]
    elif '-lM' in t.ch[1].c:  output = [ 'Mod'+str(getLocalArity(t.ch[0].c)), translate( t.ch[0], Scopes, Anaphs, lsNolo[:m] ), translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] ) ]
    ## C1-2. Conjunction...
    elif '-lC' in t.ch[0].c and ( re.match( '^(.*) \\1((?:-p\w+)?)-c\\1 \\1(-c\\1\\2)?$', form ) or re.match( '^\w+ (\w+)((?:-p\w+)?)-c\\1 \\1(-c\\1\\2)?$', form ) or
                                  re.match( '^(.*) \\1((?:-p\w+)?)-c{\\1} \\1(-c{\\1}\\2)?$', form ) or 
                                  re.match( '^\w+ X-cX-cX (\w+)-c\\1$', form ) ):
      output = [ 'And'+str(getLocalArity(t.ch[0].c)), translate( t.ch[0], Scopes, Anaphs, lsNolo ), translate( t.ch[1], Scopes, Anaphs, lsNolo ) ]
    elif '-lC' in t.ch[1].c and ( re.match( 'X-cX.*-dX (.*) \\1((?:-p\w+)?)-c\\1', form ) or re.match( 'X-cX-dX (.*) \\1((?:-p\w+)?)-c{\\1}', form ) ):
      output = translate( t.ch[1], Scopes, Anaphs, lsNolo )
#    elif '-lC' in t.ch[0].c and '-lC' in t.ch[1].c and re.match( '^(.*) \\1 \\1(-c\\1|-c{\\1})?$', form ):
#      output = [ 'And'+str(getLocalArity(t.ch[0].c)), translate( t.ch[0], Scopes, Anaphs, lsNolo ), translate( t.ch[1], Scopes, Anaphs, lsNolo ) ]
#    elif '-lG' in t.ch[0].c and re.search( '^(.*)((?:-[ghirv][^ ]*)?) (.*)((?:-[ghirv][^ ]*)?)-g\\1 \\3\\2\\4$', form ) == None and re.search( '^(.*)((?:-[ghirv][^ ]*)?) (.*)((?:-[ghirv][^ ]*)?)-g{\\1} \\3\\2\\4$', form ) == None:
#      sys.stdout.write( 'WARNING: Bad category in ' + t.c + ' -> ' + t.ch[0].c + ' ' + t.ch[1].c + '\n' )
    ## G1-2. Gap filler attachment...
    elif '-lG' in t.ch[0].c:
      ## Non-local simple argument: V-rN -> N-rN V-gN...
      if re.search( '^(\w+)((?:-[ghirv][^ ]*)?) (\w+)((?:-[ghirv][^ ]*)?)-g\\1 \\3\\2\\4$', form ):
        output = [ '\\r', '\\s', translate( t.ch[0], Scopes, Anaphs, lsNolo[:m] ), Univ, [ '\\xx'+t.ch[0].sVar, translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] + [ ['Trace','xx'+t.ch[0].sVar] ] ), 'r', 's' ] ]
      ## Non-local modifier argument: V-rN -> R-aN-rN V-g{R-aN}...
      elif re.search( '^(\w+-[ab]\w+)((?:-[ghirv][^ ]*)?) (\w+(?:-[ab]\w+)*)((?:-[ghirv][^ ]*)?)-g{\\1} \\3\\2\\4$', form ):
        output = [ '\\r', '\\s', translate( t.ch[0], Scopes, Anaphs, lsNolo[:m] ),
                                 [ '\\t'+t.ch[0].sVar, '\\u'+t.ch[0].sVar, translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] + [ ['\\p', '\\t', '\\u', 'p', 't'+t.ch[0].sVar, 'u'+t.ch[0].sVar ] ] ), 'r', 's' ],
                                 Univ, Univ ]
      else:
        sys.stdout.write( 'WARNING: Bad G rule: ' + str(t) + '\n' )  # + t.c + ' -> ' + t.ch[0].c + ' ' + t.ch[1].c + '\n' )
#    elif '-lG' in t.ch[0].c:  output = [ translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] + [ translate( t.ch[0], Scopes, Anaphs, lsNolo[:m] ) ] ) ]
#    elif '-lH' in t.ch[1].c and re.search( '^(.*)-h(.*) \\2 \\1$', form ) == None and re.search( '^(.*)-h{(.*)} \\2 \\1$', form ) == None:
#      sys.stdout.write( 'WARNING: Bad category in ' + t.c + ' -> ' + t.ch[0].c + ' ' + t.ch[1].c + '\n' )
    ## H. Heavy shift / extraposition...
    elif '-lH' in t.ch[1].c:
      ## Zero-ary (complete) non-local with simple argument: N -> N-hO O...
      if re.search( '^(\w+)((?:-[ghirv][^ ]*)?)-h(\w+) \\3((?:-[ghirv][^ ]*)?) \\1\\2\\4$', form ):
        output = [ '\\r', '\\s', translate( t.ch[1], Scopes, Anaphs, lsNolo[m-1:] ), Univ, [ '\\xx'+t.ch[1].sVar, translate( t.ch[0], Scopes, Anaphs, lsNolo[:m-1] + [ ['Trace','xx'+t.ch[1].sVar] ] ), 'r', 's' ] ]
      ## Unary non-local with simple argument: B-aN -> B-aN-hO O...
      elif re.search( '^(\w+-[ab](?:\w+|{\w+-[a-z]\w+}))((?:-[ghirv][^ ]*)?)-h(\w+) \\3((?:-[ghirv][^ ]*)?) \\1\\2\\4$', form ):
        output = [ '\\f', '\\r', '\\s', translate( t.ch[1], Scopes, Anaphs, lsNolo[m-1:] ), Univ, [ '\\xx'+t.ch[1].sVar, translate( t.ch[0], Scopes, Anaphs, lsNolo[:m-1] + [ ['Trace','xx'+t.ch[1].sVar] ] ), 'f', 'r', 's' ] ]
      ## Zero-ary (complete) non-local with modifier: N -> N-h{A-aN} A-aN...
      elif re.search( '^(\w+)((?:-[ghirv][^ ]*)?)-h{(\w+-[ab]\w+)} \\3((?:-[ghirv][^ ]*)?) \\1\\2\\4$', form ):
        output = [ '\\r', '\\s', translate( t.ch[1], Scopes, Anaphs, lsNolo[m-1:] ),
                                 [ '\\t'+t.ch[1].sVar, '\\u'+t.ch[1].sVar, translate( t.ch[0], Scopes, Anaphs, lsNolo[:m-1] + [ ['\\ff', '\\t', '\\u', 'ff', 't'+t.ch[1].sVar, 'u'+t.ch[1].sVar ] ] ), 'r', 's' ], Univ, Univ ]
      ## Unary non-local with modifier: B-aN -> B-aN-h{A-aN} A-aN...
      elif re.search( '^(\w+-[ab]\w+)((?:-[ghirv][^ ]*)?)-h{(\w+-[ab]\w+)} \\3((?:-[ghirv][^ ]*)?) \\1\\2\\4$', form ):
        output = [ '\\q', '\\r', '\\s', translate( t.ch[1], Scopes, Anaphs, lsNolo[m-1:] ),
                                        [ '\\t'+t.ch[1].sVar, '\\u'+t.ch[1].sVar, translate( t.ch[0], Scopes, Anaphs, lsNolo[:m-1] + [ ['\\ff', '\\t', '\\u', 'ff', 't'+t.ch[1].sVar, 'u'+t.ch[1].sVar ] ] ), 'q', 'r', 's' ], Univ, Univ ]
      ## Zero-ary (complete) non-local containing zero-ary (complete) non-local: N -> N-h{C-rN} C-rN...
      elif re.search( '^(\w+)((?:-[ghirv][^ ]*)?)-h{(\w+-[ghirv]\w+)} \\3((?:-[ghirv][^ ]*)?) \\1\\2\\4$', form ):
        output = [ '\\r', '\\s', translate( t.ch[1], Scopes, Anaphs, [ [ '\\t'+t.ch[1].sVar, '\\u'+t.ch[1].sVar, translate( t.ch[0], Scopes, Anaphs, lsNolo[:m-1] + [ [ '\\qq', '\\t', '\\u', 'qq', 't'+t.ch[1].sVar, 'u'+t.ch[1].sVar ] ] ), 'r', 's' ] ] + lsNolo[m-1:] ), Univ, Univ ]
      ## Unary non-local containing zero-ary (complete) non-local: A-aN -> A-aN-h{F-gN} F-gN...
      elif re.search( '^(\w+-[ab]\w+)((?:-[ghirv][^ ]*)?)-h{(\w+-[ghirv]\w+)} \\3((?:-[ghirv][^ ]*)?) \\1\\2\\4$', form ):
        output = [ '\\q', '\\r', '\\s', translate( t.ch[1], Scopes, Anaphs, [ [ '\\t'+t.ch[1].sVar, '\\u'+t.ch[1].sVar, translate( t.ch[0], Scopes, Anaphs, lsNolo[:m-1] + [ [ '\\qq', '\\t', '\\u', 'qq', 't'+t.ch[1].sVar, 'u'+t.ch[1].sVar ] ] ), 'q', 'r', 's' ] ] + lsNolo[m-1:] ), Univ, Univ ]
      ## Zero-ary (complete) non-local containing unary non-local: N -> N-h{Cas-g{V-aN}} Cas-g{V-aN}...
      elif re.search( '^(\w+)((?:-[ghirv][^ ]*)?)-h{(\w+-[ghirv]{\w+-[ab]\w+})} \\3((?:-[ghirv][^ ]*)?) \\1\\2\\4$', form ):
        output = [ '\\r', '\\s', translate( t.ch[1], Scopes, Anaphs, [ [ '\\q', '\\t'+t.ch[1].sVar, '\\u'+t.ch[1].sVar, 'q', Univ, [ '\\xh'+t.ch[1].sVar, translate( t.ch[0], Scopes, Anaphs, lsNolo[:m-1] + [ [ '\\fh', 'fh', [QuantEq,'xh'+t.ch[1].sVar] ] ] ), 'r', 's' ] ] ] + lsNolo[m-1:] ), Univ, Univ ]
      ## Unary non-local containing unary non-local: B-aN -> B-aN-h{Cas-g{V-aN}} Cas-g{V-aN}...
      elif re.search( '^(\w+-[ab]\w+)((?:-[ghirv][^ ]*)?)-h{(\w+-[ghirv]{\w+-[ab]\w+})} \\3((?:-[ghirv][^ ]*)?) \\1\\2\\4$', form ):
        output = [ '\\q', '\\r', '\\s', translate( t.ch[1], Scopes, Anaphs, [ [ '\\p', '\\t'+t.ch[1].sVar, '\\u'+t.ch[1].sVar, 'p', Univ, [ '\\xh'+t.ch[1].sVar, translate( t.ch[0], Scopes, Anaphs, lsNolo[:m-1] + [ [ '\\fh', 'fh', [QuantEq,'xh'+t.ch[1].sVar] ] ] ), 'q', 'r', 's' ] ] ] + lsNolo[m-1:] ), Univ, Univ ]
#      elif re.search( '^(\w+)((?:-[ghirv][^ ]*)?)-h{(\w+-[ghirv](?:\w+|{.*}))} \\3((?:-[ghirv][^ ]*)?) \\1\\2\\4$', form ):
#        output = [ '\\r', '\\s', translate( t.ch[1], Scopes, Anaphs, [ [ '\\t'+t.ch[1].sVar, '\\u'+t.ch[1].sVar, translate( t.ch[0], Scopes, Anaphs, [ ['\\ff', '\\t', '\\u', 'ff', 't'+t.ch[1].sVar, 'u'+t.ch[1].sVar ] ] + lsNolo[:m-1] ), 'r', 's' ] ] + lsNolo[m-1:] ) ]
      else:
        sys.stdout.write( 'WARNING: Bad H rule: ' + str(t) + '\n' )  # + t.c + ' -> ' + t.ch[0].c + ' ' + t.ch[1].c + '\n' )
#    ## Non-local in non-local: N -> N-h{V-g{V-aN}} V-g{V-aN}-lH...
#    elif re.match( '^\w+-[ghirv]\{\w+-[ab][A-Z-az]+\}-lH$', t.ch[1].c ):  output = [ translate( t.ch[0], Scopes, Anaphs, lsNolo[:m-1] + [ [ '\\ff', translate( t.ch[1], Scopes, Anaphs, lsNolo[m-1:] + [ 'ff' ] ) ] ] ) ]
      #  output = translate( t.ch[0], Scopes, Anaphs, [ [ '\\f', '\\r', '\\s', 'f', [ '\\q', '\\t', '\\u', 'q', Univ, [ '\\zz'+t.ch[1].sVar, translate( t.ch[1], Scopes, Anaphs, [ ['Trace','zz'+t.ch[1].sVar] ] + lsNolo[:m] ), 't', 'u' ] ], 'r', 's' ] ] + lsNolo[:m] )
    ## Non-local in non-local: -h{C-rN}
#    elif '-lH' in t.ch[1].c and getNoloArity(t.ch[1].c)==1:  output = translate( t.ch[0], Scopes, Anaphs, lsNolo[:m-1] + [ [ '\\q', '\\r', '\\s', 'q', Univ, [ '\\z'+t.ch[1].sVar, translate( t.ch[1], Scopes, Anaphs, lsNolo[:m-1] + [ ['Trace','z'+t.ch[1].sVar] ] ), 'r', 's' ] ] ] )
#    elif '-lH' in t.ch[1].c:  output = [ translate( t.ch[0], Scopes, Anaphs, lsNolo[:m-1] + [ translate( t.ch[1], Scopes, Anaphs, lsNolo[m-1:] ) ] ) ]
#    ## Relative clause modification by C-rN-lR...
#    elif '-lR' in t.ch[1].c and re.search( '^(.*)((?:-[ghirv][^ ]*)?) (.*)((?:-[ghirv][^ ]*)?)-r\\1 \\1\\2\\4$', form ) == None:  
#      sys.stdout.write( 'WARNING: Bad category in ' + t.c + ' -> ' + t.ch[0].c + ' ' + t.ch[1].c + '\n' )
    ## R1-2. Relative clause modification...
    elif '-lR' in t.ch[1].c:
      ## Relative clause modification by N -> N C-rN-lR...
      if re.search( '^(\w+)((?:-[ghirv][^ ]*)?) \w+((?:-[ghirv][^ ]*)?)-r\w+ \\1\\2\\3$', form ):
        output = [        '\\r', '\\s', translate( t.ch[0], Scopes, Anaphs, lsNolo[:m]   ),      'r', [ '\\zz'+t.ch[0].sVar, '^', [ translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] + [ ['Trace','zz'+t.ch[0].sVar] ]   ), Univ, Univ ],
                                                                                                                                  [ 's', 'zz'+t.ch[0].sVar ] ] ]
      ## Relative clause modification by N -> N F-g{R-aN}-lR...
      elif re.search( '^(\w+)((?:-[ghirv][^ ]*)?) \w+((?:-[ghirv][^ ]*)?)-[gr]\{\w+-[ab]\w+\} \\1\\2\\3$', form ):
        output = [        '\\r', '\\s', translate( t.ch[0], Scopes, Anaphs, lsNolo[:m]   ),      'r', [ '\\zz'+t.ch[0].sVar, '^', [ translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] + [ [ '\\p', '\\t', '\\u', 'p', Univ, ['\\xx', 'Some', ['\\ee', 'A-aN-bN:for', 'ee', 'xx', 'zz'+t.ch[0].sVar], Univ ] ] ]   ), Univ, Univ ],
                                                                                                                                  ['s','zz'+t.ch[0].sVar] ] ]
#    elif '-lR' in t.ch[1].c and getLocalArity(t.c)==0 and getLocalArity(t.ch[1].c)==0:  output = [        '\\r', '\\s', translate( t.ch[0], Scopes, Anaphs, lsNolo[:m]   ),      'r', [ '\\zz'+t.ch[0].sVar, '^', [ translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] + [ ['Trace','zz'+t.ch[0].sVar] ]   ), Univ, Univ ], ['s','zz'+t.ch[0].sVar] ] ]
      ## Relative clause modification by N -> N I-aN-gN-lR (e.g. 'a job to do _') -- event should be in future...
      elif re.search( '^(\w+)((?:-[ghirv][^ ]*)?) \w+-[ab]\w+((?:-[ghirv][^ ]*)?)-[gr]\\1 \\1\\2\\3$', form ):
        output = [        '\\r', '\\s', translate( t.ch[0], Scopes, Anaphs, lsNolo[:m]   ),      'r', [ '\\zz'+t.ch[0].sVar, '^', [ translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] + [ ['Trace','zz'+t.ch[0].sVar] ]   ), 'Some', Univ, Univ ],
                                                                                                                                  ['s','zz'+t.ch[0].sVar] ] ]
      ## Relative clause modification by N -> N I-aN-g{R-aN}-lR (e.g. 'a time to rest _') -- event should be in future...
      elif re.search( '^(\w+)((?:-[ghirv][^ ]*)?) \w+-[ab]\w+((?:-[ghirv][^ ]*)?)-[gr]\{\w+-[ab]\w+\} \\1\\2\\3$', form ):
        output = [        '\\r', '\\s', translate( t.ch[0], Scopes, Anaphs, lsNolo[:m]   ),      'r', [ '\\zz'+t.ch[0].sVar, '^', [ translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] + [ [ '\\p', '\\t', '\\u', 'p', Univ, ['\\xx', 'Some', ['\\ee', 'A-aN-bN:for', 'ee', 'xx', 'zz'+t.ch[0].sVar], Univ ] ] ] ), 'Some', Univ, Univ ],
                                                                                                                                  ['s','zz'+t.ch[0].sVar] ] ]
#    elif '-lR' in t.ch[1].c and getLocalArity(t.c)==0 and getLocalArity(t.ch[1].c)==1:  output = [        '\\r', '\\s', translate( t.ch[0], Scopes, Anaphs, lsNolo[:m]   ),      'r', [ '\\zz'+t.ch[0].sVar, '^', [ translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] + [ ['Trace','zz'+t.ch[0].sVar] ]   ), 'Some', Univ, Univ ], ['s','zz'+t.ch[0].sVar] ] ]
      ## Relative clause modification of verb phrase...
      elif getLocalArity(t.c)==1:
        output = [ '\\q', '\\r', '\\s', translate( t.ch[0], Scopes, Anaphs, lsNolo[:m]   ), 'q', 'r', [ '\\zz'+t.ch[0].sVar, '^', [ translate( t.ch[1], Scopes, Anaphs, lsNolo[m:] + [ ['Trace','zz'+t.ch[0].sVar] ]   ), Univ, Univ ],
                                                                                                                                  ['s','zz'+t.ch[0].sVar] ] ]
      else:
        sys.stdout.write( 'WARNING: Bad R rule: ' + str(t) + '\n' )  # + t.c + ' -> ' + t.ch[0].c + ' ' + t.ch[1].c + '\n' )
    ## Relative clause modification of complete phrase or clause...
    elif '-lR' in t.ch[0].c and getLocalArity(t.c)==0:
      output = [        '\\r', '\\s', translate( t.ch[1], Scopes, Anaphs, lsNolo[m-1:] ),      'r', [ '\\zz'+t.ch[1].sVar, '^', [ translate( t.ch[0], Scopes, Anaphs, lsNolo[:m-1] + [ ['Trace','zz'+t.ch[1].sVar] ] ), Univ, Univ ],
                                                                                                                                ['s','zz'+t.ch[1].sVar] ] ]
    else:  print( '\nERROR: unhandled rule from ' + str(t) + '\n' )  # + t.c + ' to ' + t.ch[0].c + ' ' + t.ch[1].c )
    ## Propagate child stores...
    t.qstore += (t.ch[0].qstore if hasattr(t.ch[0],'qstore') else []) + (t.ch[1].qstore if hasattr(t.ch[1],'qstore') else [])

  ## 2.e. Fail...
  else: print( '\nERROR: too many children in ', t )

  ## 3. Mark antecedents and anaphors...
#  t.Ants = [ ]
#  t.Anas = [ ]
  ## Mark low if anaphor...
  if ANAPH and t.bMax and t.sVar in Anaphs:
#    t.Anas += [ Anaphs[t.sVar] ]
    output = [ 'Anaphor'+str(getLocalArity(t.c)), Anaphs[t.sVar], output ]
  ## Mark high if antecedent...
  if ANAPH and t.bMax and t.sVar in Anaphs.values():
#    t.Ants += [ t.sVar ]
    output = [ 'Antecedent'+str(getLocalArity(t.c)), t.sVar, output ]
  if VERBOSE and ANAPH: print( ' '*indent, 'anaph/antec tags:', prettyForm(output) )
#  ## Mark set def sites for anaphors...
#  for a,st in t.SomeSets:
#    ## Non-discourse anaphor...
#    if st == None:  output = [ 'SomeSet', [ '\\a'+a, 'EqualSet', 'a'+a, [ '\\x', 'Equal', 'x', 'x'+a         ] ], [ '\\a'+a, output ] ]
#    ## Discourse anaphor...
#    else:           output = [ 'SomeSet', [ '\\a'+a, 'EqualSet', 'a'+a, [ '\\x'+a, access( translate( st, Scopes, Anaphs, lsNolo ) ) ] ], [ '\\a'+a, output ] ]

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
        output = [ '\\r'+l[0][2], '\\s'+l[0][2], l[0][1], [ '\\xx'+l[0][2], 'True' ], [ '\\xx'+l[0][2], output, 'r'+l[0][2], 's'+l[0][2] ] ]
        del Scopes[ l[0][2] ]
        t.qstore += l[0][0]
        t.qstore.remove( l[0] )
        ## If not stored but can scope in situ, remove from scopes and carry on translating...
        if t.bMax and t.sVar in Scopes and t.sVar not in Scopes.values():
          del Scopes[ t.sVar ]
      else:
        break

  ## 5. If scoped and cannot be in situ, store quantified noun phrase...
  if t.bMax and t.sVar in Scopes and t.sVar in Scopes.values():
#  if t.bMax and lsNolo == [] and t.sVar in Scopes and t.sVar in Scopes.values():
    ## If modal opeartor or negation...
    if getLocalArity(t.c) == 2:
      t.qstore = [( t.qstore, [ output, ['\\q','Some'], 'Some' ], t.sVar )]
      output = [ 'RaiseTraceModal', 'xx'+t.sVar ]
    ## If quantified noun phrase...
    else:
      t.qstore = [( t.qstore, output, t.sVar )]
      if VERBOSE: print( ' '*indent, 'stored ' + prettyForm(output) )
      output = [ 'RaiseTrace', 'xx'+t.sVar ]

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
#  II.A. Unpack lexical (`non-logical') constants with lambda functions...
#
########################################

Univ = [ '\\z', 'True' ]
Equal = [ '\\a', '\\b', 'Equal', 'a', 'b' ]
QuantEq = [ '\\z', '\\t', '\\u', '^', ['t','z'], ['u','z'] ]

def unpack( expr ):
  if '@' in expr[1:]:
    expr,sVar = expr[1:].split('@')
    expr = '@'+expr

  ## Recurse...
  if not isinstance( expr, str ):  return( [ unpack(subexpr) for subexpr in expr ] )

  ## Unpack grammatical functions...
  elif expr == 'Anaphor0':  return( [ '\\v',        '\\q', '\\r', '\\s',      'q', ['\\a','^',['r','a'],['InAnaphorSet','v','a']], 's' ] )
  elif expr == 'Anaphor1':  return( [ '\\v', '\\f', '\\q', '\\r', '\\s', 'f', 'q', ['\\a','^',['r','a'],['InAnaphorSet','v','a']], 's' ] )
  elif expr == 'Antecedent0':  return( [ '\\v', '\\q',        '\\r', '\\s', 'q',      ['\\a','^',['r','a'],['InAntecedentSet','v','a']], 's' ] )  #return( [ unpack(expr[2:]) ] )  #
  elif expr == 'Antecedent1':  return( [ '\\v', '\\f', '\\q', '\\r', '\\s', 'f', 'q', ['\\a','^',['r','a'],['InAntecedentSet','v','a']], 's' ] )  #return( [ unpack(expr[2:]) ] )  #
  elif expr == 'And0':  return( [ '\\f', '\\g',               '\\r', '\\s', '^', [ 'f',           'r', 's' ], [ 'g',           'r', 's' ] ] )
  elif expr == 'And1':  return( [ '\\f', '\\g',        '\\q', '\\r', '\\s', '^', [ 'f',      'q', 'r', 's' ], [ 'g',      'q', 'r', 's' ] ] )
  elif expr == 'And2':  return( [ '\\f', '\\g', '\\p', '\\q', '\\r', '\\s', '^', [ 'f', 'p', 'q', 'r', 's' ], [ 'g', 'p', 'q', 'r', 's' ] ] )
  elif expr == 'And3':  return( [ '\\f', '\\g', '\\o', '\\p', '\\q', '\\r', '\\s', '^', [ 'f', 'o', 'p', 'q', 'r', 's' ], [ 'g', 'o', 'p', 'q', 'r', 's' ] ] )
  elif expr == 'Mod0':  return( [ '\\f', '\\g',               '\\r', '\\s', 'f',           [ '\\x', '^', ['r', 'x' ], [ 'g', [ '\\t', '\\u', '^', ['t','x'], ['u','x'] ], Univ, Univ ] ], 's' ] )
  elif expr == 'Mod1':  return( [ '\\f', '\\g',        '\\q', '\\r', '\\s', 'f',      'q', [ '\\x', '^', ['r', 'x' ], [ 'g', [ '\\t', '\\u', '^', ['t','x'], ['u','x'] ], Univ, Univ ] ], 's' ] )
  elif expr == 'Mod2':  return( [ '\\f', '\\g', '\\p', '\\q', '\\r', '\\s', 'f', 'p', 'q', [ '\\x', '^', ['r', 'x' ], [ 'g', [ '\\t', '\\u', '^', ['t','x'], ['u','x'] ], Univ, Univ ] ], 's' ] )
  elif expr == 'Pred':  return( [ '\\p', '\\q', '\\r', '\\s', 'q', 'r', [ '\\x', '^', [ 'p', Univ, ['\\y','Equal','y','x'] ], [ 's', 'x' ] ] ] )
#  elif expr == 'Pasv':  return( [ '\\q', '\\r', '\\s', 'q', Univ, [ '\\x', 'p', Univ, ['\\y','Equal','y','x'] ] ] )
  elif expr == 'Ident':  return( [ '\\f', 'f' ] )
  elif expr == 'IdentSome':  return( [ '\\f', 'f', 'Some' ] )
#  elif expr == 'Store':  return( [ '\\n', '\\f', '\\q', '\\r', '\\s', 'q', Univ, ['\\x','f','r','s'] ] )
  elif expr == 'Trace':  return( [ '\\v', '\\t', '\\u', '^', ['t','v'], ['u','v'] ] )
  elif expr == 'RaiseTrace':  return( [ '\\v', '\\t', '\\u', '^', ['t','v'], ['u','v'] ] )
  elif expr == 'RaiseTraceModal': return( [ '\\v', '\\f', '\\q', '\\r', '\\s', 'f', 'q', ['\\d','^',['Equal','d','v'],['r','d']], 's' ] )
#  elif expr == 'SelfStore':  return( [ '\\v', '\\q', '\\r', '\\s', 'q', Univ, [ '\\x'+?????

  ## Unpack terminals...
  ## Cardinal...
  elif expr.split(':')[0] == '@K':  return( expr[1:] )
  ## Possessive marker...
  elif re.search( '@\w+POSS-[ab]{\w+-[ab]\w+}-[ab]\w+:', expr ):  return( [ '\\q', '\\f', '\\r', '\\s', 'q', Univ, [ '\\x', 'f', 'Some', ['\\z','^',['r','z'],['Have','x','z']], 's' ] ] )
  ## Quantifier...
  elif expr.split(':')[0] in ['@N-bO','@N-bN']:  return( [ '\\q', '\\r', '\\s', expr[1:], [ '\\x'+sVar, 'q', 'r', ['\\x','Equal','x','x'+sVar] ], 's' ] )
  elif expr.split(':')[0] == '@N-b{N-aD}':  return( [ '\\f', '\\r', '\\s', expr[1:], [ '\\x'+sVar, 'f', 'Some', 'r', ['\\x','Equal','x','x'+sVar] ], 's' ] )
  elif expr.split(':')[0] == '@N-aD-b{N-aD}':  return( [ '\\f', '\\q', '\\r', '\\s', expr[1:], [ '\\x'+sVar, 'f', 'q', 'r', ['\\x','Equal','x','x'+sVar] ], 's' ] )
  elif re.search( '@\w+-bO-aK:', expr ):  return( [ '\\n', '\\q', '\\r', '\\s', 'CountEq', [ expr[1:], 'n' ], [ '\\z', '^', [ 'r', 'z' ], [ 'q', [Equal,'z'], Univ ] ], 's' ] )
#  elif re.search( '@\w+-bO-b{N-aD-b{N-aD}}:', expr ):
#    return( [ '\\f', '\\q', '\\r', '\\s', expr[1:], [ '\\x'+sVar, 'f', 'q', 'r', ['\\x','Equal','x','x'+sVar] ], 's' ] )
#  ## Without: AMDL-aN-bN...
#  elif re.search( '^@[A-Z]MDL-[ab]\w+-[ab]\w+(-[stuwxyz].*)?:', expr ) != None:
#    return( [        '\\p', '\\q', '\\r', '\\s', expr[1:], [ '\\e', '^', ['r', 'e'], ['q', Univ, ['\\x', 'p', Univ, ['\\y', expr[1:-3],'e','x','y'] ] ] ], 's' ] )
  ## Ordinal quantifier...
  elif expr.split(':')[0] == '@NNORD-aD-b{N-aD}':  return( [ '\\f', '\\q', '\\r', '\\s', 'Some', [ '\\x'+sVar, '^', [ 'f', 'q', 'r', ['\\x','Equal','x','x'+sVar] ],
                                                                                                                    [ expr[1:]+'MinusOne', [ '\\y'+sVar, 'f', 'q', 'r', ['\\y','Equal','y','y'+sVar] ], ['\\y'+sVar, 'Prec','x'+sVar,'y'+sVar] ] ], 's' ] )
  ## Ordinal superlative quantifier...
  elif re.search( '@[AN]NORDSUP3?-aD-b{N-aD}-b{A-aN}:', expr ):  #expr.split(':')[0] == '@ANORDSUP-aD-b{N-aD}-b{A-aN}':
    return( [ '\\f', '\\g', '\\q', '\\r', '\\s', 'Some', [ '\\x'+sVar, '^', [ 'g', 'q', 'r', [Equal,'x'+sVar] ],
                                                                            [ expr[1:]+'MinusOne', [ '\\y'+sVar, 'g', 'q', 'r', [Equal,'y'+sVar] ],
                                                                                                   [ '\\y'+sVar, 'More', ['\\z'+sVar,'f',[QuantEq,'y'+sVar],[Equal,'z'+sVar],Univ],
                                                                                                                         ['\\z'+sVar,'f',[QuantEq,'x'+sVar],[Equal,'z'+sVar],Univ] ] ] ], 's' ] )
  ## Superlative quantifier...
#  elif expr.split(':')[0] == '@NNSUP3-aD-b{N-aD}-b{A-aN}':  return( [ '\\f', '\\g', '\\q', '\\r', '\\s', 'Some', [ '\\x'+sVar, '^', [ 'g', 'q', 'r', ['\\z',Equal,'x'+sVar] ],
  elif re.search( '@[AN]NSUP3?-aD-b{N-aD}-b{A-aN}:', expr ) or re.search( '@ANSUP3-aN:', expr ):
    return( [ '\\f', '\\g', '\\q', '\\r', '\\s', 'Some', [ '\\x'+sVar, '^', [ 'g', 'q', 'r', [Equal,'x'+sVar] ],
                                                                            [ 'None', [ '\\y'+sVar, 'g', 'q', 'r', [Equal,'y'+sVar] ],
                                                                                      [ '\\y'+sVar, expr[1:], ['\\z','f',[QuantEq,'y'+sVar],[Equal,'z'],Univ],
                                                                                                              ['\\z','f',[QuantEq,'x'+sVar],[Equal,'z'],Univ] ] ] ], 's' ] )
  elif re.search( '@[AN]NSUP-aD-b{N-aD}:', expr ):
    return( [ '\\f', '\\q', '\\r', '\\s', 'Some', [ '\\x'+sVar, '^', [ 'f', 'q', 'r', [Equal,'x'+sVar] ],
                                                                     [ 'None', [ '\\y'+sVar, 'f', 'q', 'r', [Equal,'y'+sVar] ],
                                                                               [ '\\y'+sVar, 'More', ['\\z',expr[1:],'y'+sVar,'z'],
                                                                                                     ['\\z',expr[1:],'x'+sVar,'z'] ] ] ], 's' ] )
#                                                                               [ '\\y'+sVar, 'More', ['\\z',expr[1:],[QuantEq,'y'+sVar],[Equal,'z'],Univ],
#                                                                                                     ['\\z',expr[1:],[QuantEq,'x'+sVar],[Equal,'z'],Univ] ] ] ], 's' ] )
  elif re.search( '@[AN]NSUP-aD-bO:', expr ):
    return( [ '\\q', '\\r', '\\s', 'Some', [ '\\x'+sVar, '^', [ 'q', 'r', [Equal,'x'+sVar] ],
                                                              [ 'None', [ '\\y'+sVar, 'f', 'q', 'r', [Equal,'y'+sVar] ],
                                                                        [ '\\y'+sVar, 'More', ['\\z',expr[1:],'y'+sVar,'z'],
                                                                                              ['\\z',expr[1:],'x'+sVar,'z'] ] ] ], 's' ] )
#                                                                        [ '\\y'+sVar, 'More', ['\\z',expr[1:],[QuantEq,'y'+sVar],[Equal,'z'],Univ],
#                                                                                              ['\\z',expr[1:],[QuantEq,'x'+sVar],[Equal,'z'],Univ] ] ] ], 's' ] )
  ## Nominal clause: how adj a thing is...
  elif re.search( '^@N-b\{\w+-g\{\w+-a\w+\}\}-b\{\w+-a\w+\}(-[lmnstuxyz].*)?:', expr ) != None:  return( [ '\\f', '\\g', '\\r', '\\s', expr[1:], [ '\\a'+sVar, 'g', [ '\\q', '\\t', '\\u', 'f', 'q', ['\\e'+sVar, '^', [ 't', 'e'+sVar ], ['In', 'a'+sVar, 'e'+sVar] ], 'u' ], 'r', 's' ], Univ ] )
  ## Pronoun...
  elif re.search( '^@NNGEN\w*(-[lmnstuxyz].*)?:', expr ) != None:  return( [ '\\r', '\\s', 'Gen', [ '\\z'+sVar, '^', [ 'Some', [ '\\e'+sVar, expr[1:],'e'+sVar,'z'+sVar ], Univ ], ['r','z'+sVar] ], 's' ] )
  elif re.search( '^@NNNEG\w*(-[lmnstuxyz].*)?:', expr ) != None:  return( [ '\\r', '\\s', 'None', [ '\\z'+sVar, '^', [ 'Some', [ '\\e'+sVar, expr[1:],'e'+sVar,'z'+sVar ], Univ ], ['r','z'+sVar] ], 's' ] )
  elif re.search( '^@[DN]\w*(-[lmnstuxyz].*)?:', expr ) != None:  return( [ '\\r', '\\s', 'Some', [ '\\z'+sVar, '^', [ 'Some', [ '\\e'+sVar, expr[1:],'e'+sVar,'z'+sVar ], Univ ], ['r','z'+sVar] ], 's' ] )
  ## Two-argument noun using possessive as possessor e.g. NNASSOC*-aD...
  elif re.search( '^@NNASSOC\w*-[ab]\w*(-[stuwxyz].*)?:', expr ):
    return( [        '\\q', '\\r', '\\s', 'Some', [ '\\x'+sVar, '^', [ 'q', Univ, [ '\\z'+sVar, '^', [ 'Some', [ '\\e'+sVar, expr[1:],'e'+sVar,'x'+sVar,'z'+sVar ], Univ ], [ 'Assoc', 'x'+sVar, 'z'+sVar ] ] ], ['r','x'+sVar] ], 's' ] )
  ## Two-argument noun using possessive as argument e.g. NNREL*-aD...
  elif re.search( '^@NNREL\w*-[ab]\w*(-[stuwxyz].*)?:', expr ):
    return( [        '\\q', '\\r', '\\s', 'Some', [ '\\x'+sVar, '^', [ 'q', Univ, [ '\\z'+sVar, 'Some', [ '\\e'+sVar, expr[1:],'e'+sVar,'x'+sVar,'z'+sVar ], Univ ] ], ['r','x'+sVar] ], 's' ] )
  ## Two-argument noun e.g. N-aD-bO --- NOTE: middle (determiner) argument 'q'/'y' gets ignored!...
  elif re.search( '^@N\w*-[ab]\w*-[ab]\w*(-[stuwxyz].*)?:', expr ):
    return( [ '\\p', '\\q', '\\r', '\\s', 'Some', [ '\\x'+sVar, '^', [ 'p', Univ, [ '\\z'+sVar, 'Some', [ '\\e'+sVar, expr[1:],'e'+sVar,'x'+sVar,'z'+sVar ], Univ ] ], ['r','x'+sVar] ], 's' ] )
  ## One-argument noun e.g. N-aD with generic force --- NOTE: middle (determiner) argument 'q'/'y' gets ignored...
  elif re.search( '^@NNGEN\w*-[ab]\w*(-[stuwxyz].*)?:', expr ):
    return( [        '\\q', '\\r', '\\s', 'Gen',  [ '\\x'+sVar, '^', [                          'Some', [ '\\e'+sVar, expr[1:],'e'+sVar,'x'+sVar          ], Univ   ], ['r','x'+sVar] ], 's' ] )
  ## One-argument noun with determiner argument interpreted as possessive e.g. N-aD...
  elif re.search( '@\w+POSS-[ab]\w+:', expr ):
    return( [        '\\q', '\\r', '\\s', 'q', Univ, [ '\\x', 'Some', [ '\\z'+sVar, '^', [      'Some', [ '\\e'+sVar, expr[1:], 'e'+sVar, 'z'+sVar        ], Univ   ], [ '^', [ 'r','z'+sVar], ['Have','x','z'+sVar] ] ], 's' ] ] )
  ## One-argument noun e.g. N-aD --- NOTE: middle (determiner) argument 'q'/'y' gets ignored...
  elif re.search( '^@N\w*-[ab]\w*(-[stuwxyz].*)?:', expr ):
    return( [        '\\q', '\\r', '\\s', 'Some', [ '\\x'+sVar, '^', [                          'Some', [ '\\e'+sVar, expr[1:],'e'+sVar,'x'+sVar          ], Univ   ], ['r','x'+sVar] ], 's' ] )
#  elif expr.split(':')[0] == '@N-aD':  return( [ '\\q', '\\r', '\\s', 'Some', [ '\\z'+sVar, '^', [ 'Some', [ '\\e'+sVar, expr[1:],'e'+sVar,'z'+sVar ], Univ ], ['r','z'+sVar] ], 's' ] )
  elif expr.split(':')[0] == '@A-aN-b{F-gN}':  return( [ '\\f', '\\q', '\\r', '\\s', expr[1:], [ '\\zz', 'q', Univ, [Equal,'zz'] ],
                                                                                               [ '\\zz', 'f', [QuantEq,'zz'], 'r', 's' ] ] )
  ## Sentential subject e.g. A-a{V-iN}:clear or bare relative: N-b{V-gN}:what
  elif re.search( '^@\w+-[ab]\{\w+-[abghirv]\w+\}(-[stuwxyz].*)?:', expr ):  return( [ '\\f', '\\r', '\\s', 'Gen', [ '\\z', '^', [ 'f', [QuantEq,'z'], Univ, Univ ], ['r','z'] ], 's' ] )
  ## Comparatives: A-aN-b{V-g{V-aN}}...
#  elif expr.split(':')[0] == '@Acomp-aN-b{V-g{V-aN}}':
  elif re.search( '^@\w+-[ab]\w+-[ab]\{\w+-[abghirv]\{\w+-[abghirv]\w+\}\}(-[stuwxyz].*)?:', expr ):
    return( [ '\\f', '\\q', '\\r', '\\s', 'q', Univ, [ '\\x'+sVar, 'f', [ '\\p', '\\t', '\\u', 'p', Univ, [ '\\y'+sVar, '^', ['u','y'+sVar], [ 'More', ['\\z'+sVar,'Some',['\\e','^',['t','e'],[expr[1:]+'nessContains','e','x'+sVar,'z'+sVar]],'u'],
                                                                                                                                                       ['\\z'+sVar,'Some',['\\e','^',['t','e'],[expr[1:]+'nessContains','e','y'+sVar,'z'+sVar]],'u'] ] ] ], 'r', 's' ] ] )
  ## Bare relative: N-b{V-g{R-aN}}...
  elif re.search( '^@\w+-[ab]\{\w+-[ghirv]\{\w+-[ab]\w+\}\}(-[stuwxyz].*)?:', expr ) != None:
    return( [ '\\f', '\\r', '\\s', 'f', [ '\\q', '\\t', '\\u', 'All', Univ, [ '\\x'+sVar, 'q', Univ, [ '\\y'+sVar, expr[1:], 'x'+sVar, 'y'+sVar ] ] ], 'r', 's' ] )
  ## Bare relative: N-b{V-gN}-b{N-aD}...
  elif re.search( '^@\w+-[ab]\{\w+-[ghirv]\w+\}-[ab]\{\w+-[ab]\w+\}(-[stuwxyz].*)?:', expr ) != None:
    return( [ '\\f', '\\g', '\\r', '\\s', 'All', [ '\\z'+sVar, '^', [ '^', [ 'f', 'Some', [Equal,'z'+sVar], Univ ], [ 'g', [QuantEq,'z'+sVar], Univ, Univ ] ], ['r','z'+sVar] ],
                                                 's' ] )
#  elif expr.split(':')[0] == '@B-aN-b{A-aN}':  return( [ '\\f', '\\q', '\\r', '\\s', 'f', 'q', [ '\\e', '^', [expr[1:],'e'], ['r','e'] ], 's' ] )
#  elif expr.split(':')[0] == '@B-aN-b{B-aN}':  return( [ '\\f', '\\q', '\\r', '\\s', 'f', 'q', [ '\\e', '^', [expr[1:],'e'], ['r','e'] ], 's' ] )
#  elif expr.split(':')[0] == '@I-aN-b{B-aN}':  return( [ '\\f', '\\q', '\\r', '\\s', 'f', 'q', [ '\\e', '^', [expr[1:],'e'], ['r','e'] ], 's' ] )
  ## Nominal relative or interrogative pronoun: N-rN...
  elif re.search( '^@[DN]-[ri]N:', expr ):  return( [ '\\q', '\\r', '\\s', 'q', Univ, [ '\\x'+sVar, 'Some', [ '\\e'+sVar, '^', [expr[1:],'e'+sVar,'x'+sVar], ['r','e'+sVar] ], 's' ] ] )
#  elif expr.split(':')[0] == '@N-rN':  return( [ '\\q', '\\r', '\\s', 'q', Univ, [ '\\x'+sVar, 'Some', [ '\\e'+sVar, '^', [expr[1:],'e'+sVar,'x'+sVar], ['r','e'+sVar] ], 's' ] ] )
  ## Adverbial relative or interrogative pronoun: R-aN-rN...
  elif re.search( '^@A-aN-[ri]N:', expr ):  return( [ '\\p', '\\q', '\\r', '\\s', 'q', Univ, [ '\\x'+sVar, 'p', Univ, [ '\\y'+sVar, 'Some', [ '\\e'+sVar, '^', [expr[1:],'e'+sVar,'x'+sVar,'y'+sVar], ['r','e'+sVar] ], 's' ] ] ] )
  ## Determiner relative or interrogative pronoun: N-b{N-aD}-iN...
  elif re.search( '^@N-b{N-aD}-[ri]N:', expr ):  return( [ '\\q', '\\f', '\\r', '\\s', 'q', Univ, [ '\\x'+sVar, 'f', 'Some', [ '\\y'+sVar, '^', [Equal,'x'+sVar,'y'+sVar], ['r','y'+sVar] ], 's' ] ] )
#  elif expr.split(':')[0] == '@A-aN-rN':  return( [ '\\p', '\\q', '\\r', '\\s', 'q', Univ, [ '\\x'+sVar, 'p', Univ, [ '\\y', 'Some', [ '\\e'+sVar, '^', [expr[1:],'e'+sVar,'x'+sVar,'y'+sVar], ['r','e'+sVar] ], 's' ] ] ] )
#  elif expr.split(':')[0] == '@B-aN-bA':  return( [ '\\p', '\\q', '\\r', '\\s', 'q', Univ, [ '\\x', 'Some', [ '\\e', '^', [ expr[1:], 'e', 'x', [ 'Intension', [ 'p', Univ, Univ ] ] ], ['r','e'] ], 's' ] ] )
  ## Formula: B...
  elif re.search( '^@B(-[stuwxyz].*)?:', expr ) != None:
    return( [                      '\\r', '\\s',                                                                           'Some', [ '\\e'+sVar, '^', [expr[1:],'e'+sVar,'x'+sVar                  ], ['r','e'+sVar] ], 's'       ] )
  ## Intransitive: B-aN...
  elif re.search( '^@\w+-[ab]\w*(-[stuwxyz].*)?:', expr ) != None:
    return( [               '\\q', '\\r', '\\s', 'q', Univ, [ '\\x'+sVar,                                                  'Some', [ '\\e'+sVar, '^', [expr[1:],'e'+sVar,'x'+sVar                  ], ['r','e'+sVar] ], 's'     ] ] )
  ## Transitive: B-aN-bN...
  elif re.search( '^@\w+-[ab]\w+-[ab][DNOPa-z]+(-[stuwxyz].*)?:', expr ) != None:
    return( [        '\\p', '\\q', '\\r', '\\s', 'q', Univ, [ '\\x'+sVar, 'p', Univ, [ '\\y'+sVar,                         'Some', [ '\\e'+sVar, '^', [expr[1:],'e'+sVar,'x'+sVar,'y'+sVar         ], ['r','e'+sVar] ], 's'   ] ] ] )
  ## Ditransitive: B-aN-bN-bN...
  elif re.search( '^@\w+-[ab]\w+-[ab]\w+-[ab]\w+(-[stuwxyz].*)?:', expr ) != None:
    return( [ '\\o', '\\p', '\\q', '\\r', '\\s', 'q', Univ, [ '\\x'+sVar, 'p', Univ, [ '\\y'+sVar, 'o', Univ, ['\\z'+sVar, 'Some', [ '\\e'+sVar, '^', [expr[1:],'e'+sVar,'x'+sVar,'y'+sVar,'z'+sVar], ['r','e'+sVar] ], 's' ] ] ] ] )
  ## Tritransitive: B-aN-bN-bN-bN...
  elif re.search( '^@\w+-[ab]\w+-[ab]\w+-[ab]\w+-[ab]\w+(-[stuwxyz].*)?:', expr ) != None:
    return( [ '\\n', '\\o', '\\p', '\\q', '\\r', '\\s', 'q', Univ, [ '\\w'+sVar, 'p', Univ, [ '\\x'+sVar, 'o', Univ, ['\\y'+sVar, 'n', Univ, ['\\z'+sVar, 'Some', [ '\\e'+sVar, '^', [expr[1:],'e'+sVar,'w'+sVar,'x'+sVar,'y'+sVar,'z'+sVar], ['r','e'+sVar] ], 's' ] ] ] ] ] )
  ## Modal auxiliary: BMDL-aN-b{A-aN}...
  elif re.search( '^@[A-Z]MDL-[ab]\w+-[ab]\{\w+-[ab]\w+\}(-[stuwxyz].*)?:', expr ) != None:
    return( [        '\\f', '\\q', '\\r', '\\s', expr[1:], 'r', [ '\\e', 'f', 'q', ['\\d', 'Equal','d','e'], 's' ] ] )
  ## Raising auxiliary: B-aN-b{A-aN}...
  elif re.search( '^@\w+-[ab]\w+-[ab]\{\w+-[ab]\w+\}(-[stuwxyz].*)?:', expr ) != None:
    return( [        '\\f', '\\q', '\\r', '\\s',                                                                         'f', 'q', [ '\\e'+sVar, '^', [expr[1:],'e'+sVar                           ], ['r','e'+sVar] ], 's'       ] )
  ## Raising auxiliary: B-a{I-aN}-b{A-aN}...
  elif re.search( '^@\w+-[ab]{\w+-[ab]\w+}-[ab]\{\w+-[ab]\w+\}(-[stuwxyz].*)?:', expr ) != None:
    return( [        '\\f', '\\g', '\\r', '\\s',                                                                         'f', ['g','Some'], [ '\\e'+sVar, '^', [expr[1:],'e'+sVar                           ], ['r','e'+sVar] ], 's'       ] )
  ## Raising goal: A-aN-aPin-b{I-aN}:order...
  elif re.search( '^@\w+-[ab]\w+-[ab]\w+-[ab]{\w+-[ab]\w+}(-[stuwxyz].*)?:', expr ) != None:
    return( [ '\\f', '\\p', '\\q', '\\r', '\\s', 'q', Univ, [ '\\x'+sVar, 'Some', [ '\\e'+sVar, '^', [ expr[1:], 'e'+sVar, 'x'+sVar, [ 'Intension', [ 'f', [QuantEq,'x'+sVar], Univ, Univ ] ] ], ['r','e'+sVar] ], 's' ] ] )
  ## Range modifier (??): A-aN-b{A-aN}-bN...
  elif re.search( '^@\w+-[ab]\w+-[ab]\{\w+-[ab]\w+\}-[ab]\w+(-[stuwxyz].*)?:', expr ) != None:
    return( [        '\\p', '\\f', '\\q', '\\r', '\\s', 'p', Univ, [ '\\x'+sVar,                                         'f', 'q', [ '\\e'+sVar, '^', [expr[1:],'e'+sVar                           ], ['r','e'+sVar] ], 's'     ] ] )
  ## Take construction: B-a{A-aN}-bN-bO...
  elif re.search( '^@\w+-[ab]\{\w+-[ab]\w+\}-[ab]\w+-[ab]\w+(-[stuwxyz].*)?:', expr ) != None:
    return( [        '\\p', '\\q', '\\f', '\\r', '\\s', 'f', 'p', 'r', [ '\\e'+sVar, '^', ['s','e'+sVar], [ 'q', Univ, ['\\y'+sVar,expr[1:],'e'+sVar,'y'+sVar] ] ] ] )
  ## Sent comp: B-aN-bC:think...
  elif re.search( '^@\w+-[ab]\w+-[ab][ABCEFGIVQRSVa-z]+(-[stuwxyz].*)?:', expr ) != None:
    return( [        '\\p', '\\q', '\\r', '\\s', 'q', Univ, [ '\\x'+sVar, 'Some', [ '\\e'+sVar, '^', [ expr[1:], 'e'+sVar, 'x'+sVar, [ 'Intension', [ 'p', Univ, Univ ] ] ], ['r','e'+sVar] ], 's' ] ] )
  ## Sent comp: A-aN-b{I-gN}:for...
  elif re.search( '^@\w+-[ab]\w+-[ab]{\w+-[ghirv]\w+}(-[stuwxyz].*)?:', expr ) != None:
    return( [ '\\f', '\\q', '\\r', '\\s', 'q', Univ, [ '\\x'+sVar, 'Some', [ '\\e'+sVar, '^', [ expr[1:], 'e'+sVar, 'x'+sVar, [ 'Intension', [ 'f', [QuantEq,'x'+sVar], Univ, Univ ] ] ], ['r','e'+sVar] ], 's' ] ] )
  ## Sent comp: B-aN-b{V-iN}-bN:tell...
  elif re.search( '^@\w+-[ab]\w+-[ab]{\w+-[ghirv]\w+}-[ab]\w+(-[stuwxyz].*)?:', expr ) != None:
    return( [ '\\p', '\\f', '\\q', '\\r', '\\s', 'q', Univ, [ '\\x'+sVar, 'p', Univ, [ '\\y'+sVar, 'Some', [ '\\e'+sVar, '^', [ expr[1:], 'e'+sVar, 'x'+sVar, [ 'Intension', [ 'f', [QuantEq,'y'+sVar], Univ, Univ ] ] ], ['r','e'+sVar] ], 's' ] ] ] )
  ## Tough constructions: A-aN-b{I-aN-gN}...
  elif re.search( '^@\w+-[ab]\w+-[ab]{I-aN-gN}(-[stuwxyz].*)?:', expr ) != None:
    return( [        '\\f', '\\q', '\\r', '\\s', 'q', Univ, [ '\\x'+sVar, 'Gen', [ '\\e'+sVar, 'f', [ '\\t', '\\u', '^', ['t','x'+sVar], ['u','x'+sVar] ], 'Some', 'r', [ '\\d'+sVar, '^', ['s','d'+sVar], ['Equal','d'+sVar,'e'+sVar] ] ], [ '\\e'+sVar,expr[1:],'e'+sVar] ] ] )
  ## Tough constructions with comparative: A-aN-b{I-aN-gN}-b{Cthan-g{V-aN}}...
  elif re.search( '^@\w+-[ab]\w+-[ab]{I-aN-gN}-[ab]\{\w+-[abghirv]\{\w+-[abghirv]\w+\}\}(-[stuwxyz].*)?:', expr ) != None:
    return( [ '\\fCgVaN', '\\gIaNgN', '\\q', '\\r', '\\s', 'fCgVaN', [ '\\p', '\\t', '\\u', 'q', Univ, [ '\\x'+sVar, 'p', Univ, [ '\\y'+sVar, 'Gen', [ '\\eX', 'Gen', [ '\\eY', '^', ['u','y'+sVar], [ 'More', ['\\z'+sVar, 'gIaNgN', [ QuantEq,'x'+sVar], 'Some', 'r', [ '\\d'+sVar, '^', [ '^', ['s','d'+sVar], ['Equal','d'+sVar,'eX'] ], [ expr[1:]+'nessContains','d'+sVar,'z'+sVar] ] ],
                                                                                                                                                                                                               ['\\z'+sVar, 'gIaNgN', [ QuantEq,'y'+sVar], 'Some', 'r', [ '\\d'+sVar, '^', [ '^', ['s','d'+sVar], ['Equal','d'+sVar,'eY'] ], [ expr[1:]+'nessContains','d'+sVar,'z'+sVar] ] ] ] ], Univ ], Univ ] ] ] ] )
  ## Use to V P _ constructions: B-aN-b{I-aN-gN}-bN...
  elif re.search( '^@\w+-[ab]\w+-[ab]{I-aN-gN}-[ab]\w+(-[stuwxyz].*)?:', expr ) != None:
    return( [ '\\p', '\\f', '\\q', '\\r', '\\s', 'q', Univ, [ '\\x'+sVar, 'p', Univ, [ '\\y'+sVar, 'Gen', [ '\\e'+sVar, 'f', [ '\\tt', '\\uu', '^', ['tt','y'+sVar], ['uu','y'+sVar] ], [ '\\t', '\\u', '^', ['t','x'+sVar], ['u','x'+sVar] ], 'r', [ '\\d'+sVar, '^', ['s','d'+sVar], ['Equal','d'+sVar,'e'+sVar] ] ], [ '\\e'+sVar,expr[1:],'e'+sVar] ] ] ] )
#    return( [ '\\f', '\\q', '\\r', '\\s', 'q', Univ, [ '\\x'+sVar, 'f', [ '\\p', '\\t', '\\u', 'p', Univ, [ '\\y'+sVar, '^', ['u','y'+sVar], [ 'More', ['\\z'+sVar,'Some',['\\e','^',['t','e'],[expr[1:]+'nessContains','e','x'+sVar,'z'+sVar]],'u'],
#                                                                                                                                                       ['\\z'+sVar,'Some',['\\e','^',['t','e'],[expr[1:]+'nessContains','e','y'+sVar,'z'+sVar]],'u'] ] ] ], 'r', 's' ] ] )
  ## Embedded question: B-aN-b{V-iN}...
  elif re.search( '^@\w+-[ab]\w+-[ab]{V-iN}(-[stuwxyz].*)?:', expr ) != None:
    return( [        '\\f', '\\q', '\\r', '\\s', 'q', Univ, [ '\\x'+sVar, 'f', [ '\\t', '\\u', '^', ['t','x'+sVar], ['u','x'+sVar] ], Univ, [ '\\d'+sVar, 'Some', 'r', ['\\e'+sVar, '^', [expr[1:],'e'+sVar,'x'+sVar,'d'+sVar], ['s','d'+sVar] ] ] ] ] )
  ## 'No matter' construction: A-aN-b{V-iN}-aPno...
  elif re.search( '^@\w+-[ab]\w+-[ab]{V-iN}-aP[a-z]+(-[stuwxyz].*)?:', expr ) != None:
    return( [        '\\f', '\\q', '\\r', '\\s', 'q', Univ, [ '\\x'+sVar, 'f', [ '\\t', '\\u', '^', ['t','x'+sVar], ['u','x'+sVar] ], Univ, [ '\\d'+sVar, 'Some', 'r', ['\\e'+sVar, '^', [expr[1:],'e'+sVar,'x'+sVar,'d'+sVar], ['s','d'+sVar] ] ] ] ] )
  ## Bare relatives (bad take as eventuality of V is not constrained by wh-noun): N-b{V-gN}...
  elif re.search( '^@\w+-[ab]{V-gN}(-[stuwxyz].*)?:', expr ) != None:
    return( [        '\\f',        '\\r', '\\s', 'Some', [ '\\x'+sVar, '^', [ '^', [ 'Some', ['\\e'+sVar,expr[1:],'e'+sVar,'x'+sVar], Univ ], ['r','x'+sVar] ], [ 'f', [ '\\t', '\\u', '^', ['t','x'+sVar], ['u','x'+sVar] ], Univ, Univ ] ], 's' ] )
#  elif expr[0]=='@' and getLocalArity( t.split(':')[0] ) == 1:  return( [        '\\q', '\\r', '\\s', 'q', Univ, [ '\\x',                     'Some', [ '\\e', '^', [expr[1:],'e','x'    ], ['r','e'] ], 's'   ] ] )
#  elif expr[0]=='@' and getLocalArity( t.split(':')[0] ) == 2:  return( [ '\\p', '\\q', '\\r', '\\s', 'q', Univ, [ '\\x', 'p', Univ, [ '\\y', 'Some', [ '\\e', '^', [expr[1:],'e','x','y'], ['r','e'] ], 's' ] ] ] )
  else:
    if expr[0]=='@':  print( 'WARNING: no macro expansion for', expr )
    return( expr )


########################################
#
#  II.B. Non-destructive replace variable with substituted expression in beta reduce...
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
  if VERBOSE: print( 'reducing:', prettyForm(expr) )
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

  ## Base case...
  if isinstance( expr, str ):  return

  ## Recurse...
  for i in range( len(expr) ):
    simplify( expr[i] )
#    ## Remove unaries...
#    if isinstance( expr[i], list ) and len(expr[i]) == 1:  expr[i] = expr[i][0]

  if 'AntecTmp' in expr:
    i = expr.index( 'AntecTmp' )
    expr[:] = expr[:i] + expr[i+2:]

  try:
#    ## Eliminate unaries...
#    if len(expr)==1:
#      print( 'was:', expr )
#      expr[:] = expr[0]
#      print( 'now:', expr )

    ## Eliminate conjunctions with tautology...
    if len(expr)==3 and expr[0]=='^' and expr[1]==['True']:
      expr[:] = expr[2]
#      simplify( expr )
    elif len(expr)==3 and expr[0]=='^' and expr[2]==['True']:
      expr[:] = expr[1]
#      simplify( expr )
    elif len(expr)==4 and expr[0][0]=='\\' and expr[1]=='^' and expr[2]==['True']:
      expr[:] = [ expr[0], expr[3] ]
#      simplify( expr )
    elif len(expr)==4 and expr[0][0]=='\\' and expr[1]=='^' and expr[3]==['True']:
      expr[:] = [ expr[0], expr[2] ]
#      simplify( expr )

    ## Eliminate existentials with conjunctions with equality...
#    print( prettyForm(expr) )
    if expr[0]=='Some' and expr[2][1]=='Equal':
      newVar = expr[2][3] if expr[2][3] != expr[2][0][1:] else expr[2][2]
      if VERBOSE:  print( 'pruningA replacing', expr[1][0][1:], 'with', newVar, 'in', prettyForm(expr) )
      expr[:] = replace( expr[1][1:], expr[1][0][1:], newVar )
      if VERBOSE:  print( '    yields', prettyForm(expr) )
    if len(expr)==4 and expr[1]=='Some' and expr[3][1]=='Equal':
      newVar = expr[3][3] if expr[3][3] != expr[3][0][1:] else expr[3][2]
      if VERBOSE:  print( 'pruningB replacing', expr[2][0][1:], 'with', newVar, 'in', prettyForm(expr) )
      expr[:] = [ expr[0] ] + replace( expr[2][1:], expr[2][0][1:], newVar )
      if VERBOSE:  print( '    yields', prettyForm(expr) )
    if expr[0]=='Some' and expr[2][1]=='^' and expr[2][3][0]=='Equal' and expr[2][3][1]==expr[2][0][1:]:
      if VERBOSE:  print( 'pruningC replacing', expr[2][0][1:], 'with', expr[2][3][2], 'in', prettyForm(expr) )
      expr[:] = [ '^', replace( expr[1][1:], expr[1][0][1:], expr[2][3][2] ), replace( expr[2][2], expr[2][0][1:], expr[2][3][2] ) ]
      if VERBOSE:  print( '    yields', prettyForm(expr) )
    if len(expr)==4 and expr[1]=='Some' and expr[3][1]=='^' and expr[3][3][0]=='Equal' and expr[3][3][1]==expr[3][0][1:]:
      if VERBOSE:  print( 'pruningD replacing', expr[3][0][1:], 'with', expr[3][3][2], 'in', prettyForm(expr) )
      expr[:] = [ expr[0], '^', replace( expr[2][1:], expr[2][0][1:], expr[3][3][2] ), replace( expr[3][2], expr[3][0][1:], expr[3][3][2] ) ]
      if VERBOSE:  print( '    yields', prettyForm(expr) )
  except ValueError:  # Exception as e:
    print( 'ERROR: unreduced expr (had set-valued variable):', prettyForm(expr) )
    print( e )
    exit( 0 )

  ## Remove unaries...
  while len( expr ) == 1 and isinstance( expr[0], list ):
    expr[:] = expr[0]
  ## Remove parents left lambda...
  if len( expr ) == 2 and expr[0][0] == '\\' and isinstance( expr[1], list ):
    expr[:] = [ expr[0] ] + expr[1]

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


########################################
#
#  II.E. percolate unbound anaphora and antecedents up expr...
#
########################################

#'''
def percAntAna( expr, Anaphs ):

  Anas = [ ]
  Ants = [ ]
  Vars = [ ]

  if VERBOSE: print( 'working on:', expr )

  if isinstance( expr, str ): return( [], [], [] )

  ## If anaphor, add...
  if expr[0] == 'InAnaphorSet':
    Anas += [ expr[1] ]
    expr[:] = [ 'a'+expr[1], expr[2] ]
  ## If antecedent, add...
  if expr[0] == 'InAntecedentSet':
    Ants += [ expr[1] ]
    Vars += [ expr[2] ]
    expr[:] = [ 'True' ]

  ## Recurse and percolate...
  AntsAnasVars = [ percAntAna( subexpr, Anaphs ) for subexpr in expr ]
  for subAnts,subAnas,subVars in AntsAnasVars:
    Anas += subAnas
    Ants += subAnts
    Vars += subVars

  ## If antecedent and anaphor meet, define antecedent set at top of anaphor branch...
  for iAnt in range( len( expr ) ):
    for i in range( len( AntsAnasVars[iAnt][0] ) ):
      a =  AntsAnasVars[iAnt][0][i]
      for iAna in range( len( expr ) ):
        if a in AntsAnasVars[iAna][1]:
          if VERBOSE: print('grabbing',a,'for antecedent var:',AntsAnasVars[iAnt][2][i],'...')
          if VERBOSE: print('I went from this:',prettyForm(expr))
          ## If antecedent is restrictor / nuclear scope...
#          for i in range( len( AntsAnasVars[iAnt][0] ) ):
#            if AntsAnasVars[iAnt][0][i] == a and AntsAnasVars[iAnt][2][i] == expr[iAnt][0][1:]:
          if AntsAnasVars[iAnt][2][i] == expr[iAnt][0][1:]:
#            print( 'FOUND MATCH', a, expr[iAnt][0][1:] )
            expr[iAna][:] = [ expr[iAna][0], 'Some', [ '\\a'+a, 'Equal', 'a'+a, [ '\\v'+a, 'Equal', 'v'+a, expr[iAna][0][1:] ]                                      ], [ '\\a'+a ] + expr[iAna][1:] ]
          ## If antecedent is in restrictor / nuclear scope (and anaphor is in nuclear scope / restrictor)...
          elif expr[iAnt][0][0] == '\\' and expr[iAna][0][0] == '\\': # and iAnt == len(expr)-2:
#            expr[iAna][:] = [ expr[iAna][0], 'Some', [ '\\a'+a, 'Equal', 'a'+a, [ '\\v'+a ] + access(replace(expr[iAnt][1:],expr[iAnt][0][1:],expr[iAna][0][1:]),a) ], [ '\\a'+a ] + expr[iAna][1:] ]
            expr[iAna][:] = [ expr[iAna][0], 'Some', [ '\\a'+a, 'Equal', 'a'+a, [ '\\v'+a ] + access( replace( replace(expr[iAnt][1:],AntsAnasVars[iAnt][2][i],'v'+a), expr[iAnt][0][1:], expr[iAna][0][1:] ), a )   ], [ '\\a'+a ] + expr[iAna][1:] ]
#          ## If antecedent is in restrictor...
#          if expr[iAnt][0][0] == '\\' and expr[iAna][0][0] == '\\':  expr[iAna][:] = [ expr[iAna][1], 'SomeSet', [ '\\a'+a, 'EqualSet', 'a'+a, [ '\\v'+a, 'Equal', 'v'+a, expr[iAna][1] ] ], [ '\\a'+a ] + expr[iAna][1:] ]
          ## If antecedent is not in restrictor...
          else:
            expr[iAna][:] = [                'Some', [ '\\a'+a, 'Equal', 'a'+a, [ '\\v'+a ] + access( replace(expr[iAnt][:],AntsAnasVars[iAnt][2][i],'v'+a), a )    ], [ '\\a'+a ] + expr[iAna][:]  ]
          while a in AntsAnasVars[iAna][1]:
            AntsAnasVars[iAna][1].remove( a )
          Anas = [ b for b in Anas if b != a ]
          if VERBOSE: print('I went  to  this:',prettyForm(expr))

  ## If quantifier over antecedent variable...
  if len(expr) > 2 and isinstance( expr[-2], list ) and expr[-2][0][0] == '\\' and expr[-1][0][0] == '\\':  # and expr[-1][0][1:] in Vars:
#    expr[:] = [ 'AntecTmp', Vars[0] ] + 
    for i in range( len( Vars ) ):
      if Vars[i] == expr[-2][0][1:]:
        expr[:] = expr[:-3] + [ 'AntecTmp', Ants[i] ] + copy.deepcopy(expr[-3:])
#    expr[:] = [ '^' [ replace( expr[-2], expr[-2][0][1:], expr[-1][0][1:] ) ], replace( expr[-1], expr[-1][0][1:], 'v'+Vars[0] ]

  if VERBOSE: print( 'I made this:', expr )

  return( Ants, Anas, Vars )
#'''


def findAnaph( expr, antvar ):
#  print( 'findAnaph ' + prettyForm(expr) + ' ' + antvar )
  if isinstance( expr, list ):
    if 'InAnaphorSet' in expr: return expr[ expr.index( 'InAnaphorSet' ) + 1 ] == antvar
#  if isinstance( expr, str ): return expr == antvar
    for subexpr in expr:
      if findAnaph( subexpr, antvar ): return True
  return False


def corefScoper( ant, expr, Ants, ScopedAnts ):
#  print( 'corefScoper ' + str(Ants) + ' ' + str(ScopedAnts) )
  for antnum in Ants:
    if antnum not in ScopedAnts and findAnaph( expr, antnum ):
#      print( '  found ' + antnum )
      return( [ 'Some', [ '\\a'+antnum, 'Equal', 'a'+antnum, [ '\\v'+antnum, access( ant, antnum ) ] ], [ '\\a'+antnum, corefScoper( ant, expr, Ants, ScopedAnts + [ antnum ] ) ] ] )
  return( coref( expr, Ants, ScopedAnts ) )


def coref( expr, Ants, ScopedAnts ):
#  print( 'coref ' + prettyForm(expr) )

  ## If lambdas, copy over...
  if isinstance( expr, list ) and '\\' == expr[0][0]:
    return( [ expr[0] ] + coref( expr[1:], Ants, ScopedAnts ) )

  ## If encountering quantifier...
  if isinstance( expr, list ) and len(expr) > 2 and isinstance( expr[-2], list ) and len(expr[-2]) > 0 and expr[-2][0][0] == '\\':
    lambdavar = expr[-2][0][1:]
    exprOut = expr[:-2] + [ coref( expr[-2], Ants, ScopedAnts ) ]
    prefix = []
    ## If quantifier variable is indexed lambda variable, index expression by antecedent number...
    if lambdavar in Ants:
      antvar = Ants[ lambdavar ]
      Ants[ antvar ] = True
      del Ants[ lambdavar ]
      prefix = [ 'AntecTmp', antvar ]
#    print( 'replacing ' + expr[-2][0][1:] + ' ' + expr[-1][0][1:] )
#    print( 'building from ' + prettyForm(exprOut) )
    return( prefix + exprOut + [ [ expr[-1][0], corefScoper( replace( exprOut[-1][1:], expr[-2][0][1:], expr[-1][0][1:] ), expr[-1][1:], Ants, ScopedAnts ) ] ] )

  ## If conjunction...
  if isinstance( expr, list ) and len(expr) > 0 and '^' == expr[0]:
#    print( 'thinking conjunction ' + prettyForm(expr) )
    exprOut = [ ]
    for i in range( len( expr ) ):
      exprOut += [ coref( expr[i], Ants, ScopedAnts ) ]
      if any([ antnum not in ScopedAnts and findAnaph( expr[i+1:], antnum ) for antnum in Ants ]):
        return( exprOut[:i+1] + [ corefScoper( exprOut[i], ( ( [ '^' ] + expr[i+1:] ) if len(expr)>i+2 else expr[i+1] ), Ants, ScopedAnts ) ] )
    return( exprOut )
 
  ## If encountering InAntecedentSet keyword, index antecedent number by lambda variable name...
  if 'InAntecedentSet' in expr:
    i = expr.index( 'InAntecedentSet' )
    Ants[ expr[ i + 2 ] ] = expr[ i + 1 ]
    return( expr[:i] + [ 'True' ] + expr[i+3:] )

#  ## If encountering InAnaphorSet keyword, replace with antecedent variable...
#  if 'InAnaphorSet' in expr:
#    i = expr.index( 'InAnaphorSet' )
#    return( expr[:i] + 'a' + expr[i+1] + expr[i+2:] )

#  print( 'NOPE: ' + prettyForm(expr) )

  ## If other predicate...
  return( expr )
    


########################################
#
#  II.F. access for anaphors...
#
########################################

def access( expr, a, bSpine=True ):
  if VERBOSE: print( 'access', a, prettyForm(expr) )

  if isinstance( expr, list ):
    ## If antecedent quantifier, make union of restrictor and nuclear scope...
#    if len(expr) > 2 and 'AntecTmp' == expr[0]:
#      return( expr[2:-3] + [ '^', replace(expr[-2][1:],expr[-2][0][1:],'v'+expr[1]), replace(expr[-1][1:],expr[-1][0][1:],'v'+expr[1]) ] )
    if len(expr) > 4 and expr[-5] == 'AntecTmp' and expr[-4] == a:
      return( expr[:-5] + [ '^', replace(expr[-2][1:],expr[-2][0][1:],'v'+expr[-4]), replace(expr[-1][1:],expr[-1][0][1:],'v'+expr[-4]) ] )
    ## If normal quantifier, make existential and recurse into nuclear scope...
    elif bSpine and len(expr) > 2 and expr[-2][0][0]=='\\' and expr[-1][0][0]=='\\':
      return( expr[:-3] + [ 'Some', access(expr[-2],a,False), access(expr[-1],a,bSpine) ] )
    else:
      return( [ access(subexpr,a,bSpine) for subexpr in expr ] )

  return( expr )


########################################
#
#  II.G. Pretty print...
#
########################################

def prettyForm( expr ):
  if isinstance( expr, str ):  return( expr )
  else:  return( '(' + ' '.join([ prettyForm(subexpr) for subexpr in expr ]) + ')' )


########################################
#
#  II.F. Check unbound vars...
#
########################################

def checkUnboundVars( expr, Bound=[] ):
  ## If variable, check if bound...
  if isinstance( expr, str ):
    if expr[0].islower() and expr not in Bound:  print( 'WARNING: variable', expr, 'unbound!' )
  ## If lambda expr...
  elif expr[0][0] == '\\':
    if expr[0][1:] in Bound:  print( 'WARNING: variable', expr[0][1:], 'multiply bound.' )
    checkUnboundVars( expr[1:], Bound + [ expr[0][1:] ] )
  ## Recurse...
  else:
    for subexpr in expr:
      checkUnboundVars( subexpr, Bound )


########################################
#
#  II.F. Check well-formed...
#
########################################

def isWellFormedProp( expr ):
  ## Conjunction...
  if isinstance( expr, list ) and len(expr) > 0 and expr[0] == '^':
    if all([ isWellFormedProp(subexpr) for subexpr in expr[1:] ]):  return True
    print( 'Ill-formed proposition: ' + prettyForm(expr) )
    return False
  ## Quantifier...
  if isinstance( expr, list ) and len(expr) > 2 and isinstance( expr[0], str ) and expr[0][0] != '\\' and isinstance( expr[1], list ) and len(expr[1]) > 0 and expr[1][0][0]=='\\' and isinstance( expr[2], list ) and len(expr[2]) > 0 and expr[2][0][0]=='\\' and isWellFormedSet( expr[1] ) and isWellFormedSet( expr[2] ):  return True
  ## Predicate...
  if isinstance( expr, list ) and len(expr) > 0 and expr[0][0] != '\\' and all([ isWellFormedEntity(subexpr) for subexpr in expr ]):  return True
  print( 'Ill-formed proposition: ' + prettyForm(expr) )
  return False

def isWellFormedEntity( expr ):
  ## Atom...
  if isinstance( expr, str ) and len(expr) > 0 and expr[0] != '\\':  return True
  ## Intension...
  if isinstance( expr, list) and len(expr) > 1 and expr[0] == 'Intension' and isWellFormedProp( expr[1] ):  return True
  print( 'Ill-formed entity: ' + prettyForm(expr) )
  return False

def isWellFormedSet( expr ):
  if isinstance( expr, list ) and len(expr) > 1 and isinstance( expr[0], str ) and len(expr[0]) > 0 and expr[0][0] == '\\' and isWellFormedProp( expr[1:] ):  return True
  print( 'Ill-formed set: ' + prettyForm(expr) )
  return False


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

  VERBOSE = ( nArticle == ONLY )

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
    print( nArticle, 'CG:', t )

  if len(Trees) == 0:  continue

  ## Combine article's trees in conjunction...
  t = None
  for i in range(len(Trees)):
    if i == 0:
      t = tree.Tree()
      t.aboveAllInSitu = True
      t.bMax = False
      t.sVar = '0'
      t.Anas = [ ]
      t.Ants = [ ]
      t.SomeSets = [ ]
      t.c = 'S-cS'
      t.ch = [ Trees[-1] ]
      t.ch[0].c = t.ch[0].c.replace( '-lS', '' )
#      t.ch = [ Trees[-1], tree.Tree() ]
#      t.ch[1].aboveAllInSitu = True
#      t.ch[1].bMax = False
#      t.ch[1].sVar = '0'
#      t.ch[1].Anas = [ ]
#      t.ch[1].Ants = [ ]
#      t.ch[1].SomeSets = [ ]
#      t.ch[1].c = 'X-cX-cX-x%|'
#      t.ch[1].ch = [ tree.Tree() ]
#      t.ch[1].ch[0].c = 'and'
    else:
      ch = [ Trees[-1-i], t ]
      t = tree.Tree()
      t.aboveAllInSitu = True
      t.bMax = False
      t.sVar = '0'
      t.Anas = [ ]
      t.Ants = [ ]
      t.SomeSets = [ ]
      t.c = 'S-cS'
      t.ch = ch
      t.ch[0].c = t.ch[0].c.replace( '-lS', '-lC' )

  if VERBOSE:  print( '========== Article ' + str(nArticle) + ' ==========' )
  print( nArticle, 'DERIV:', t )

  ## Skip article if specified in command line...
  if nArticle in SKIPS:
    if VERBOSE:  print( 'NOTE: skipping article', nArticle, 'as specified in command line arguments.' )
    print( nArticle, 'MACRO:' )
    print( nArticle, 'LOGIC:' )
    continue

  if VERBOSE:  print( '----- translate -----' )
  if VERBOSE:  print( 'Scopes', Scopes )
  shortExpr = translate( t, Scopes, Anaphs )
  if t.qstore != []:
    print( 'ERROR: nothing in quant store', t.qstore, 'allowed by scope list', Scopes )
    exit(0)
  print( nArticle, 'MACRO:', prettyForm(shortExpr) )

  if VERBOSE:  print( '----- unpack -----' )
  fullExpr = [ unpack(shortExpr), Univ, Univ ]
  if VERBOSE:  print( prettyForm(fullExpr) )
  print( nArticle, 'UNPACK:', prettyForm(fullExpr) )

  if VERBOSE:  print( '----- beta reduce -----' )
  betaReduce( fullExpr )
  if VERBOSE:  print( prettyForm(fullExpr) )
  print( nArticle, 'BETARED:', prettyForm(fullExpr) )

  print( nArticle, 'FIRST: ', end='' )
  bWell = isWellFormedProp( fullExpr )
  print()
  print( nArticle, 'WELL-FORMED:', bWell )
  if bWell:

    if VERBOSE:  print( '----- simplify -----' )
    simplify( fullExpr )
    if VERBOSE:  print( prettyForm(fullExpr) )
    print( nArticle, 'PRE-LOGIC:', prettyForm(fullExpr) )
  
    if VERBOSE:  print( '----- percolate -----' )
    if VERBOSE:  print( 'Anaphs', Anaphs )
#    percAntAna( fullExpr, Anaphs )
    Anaphs = { }
    ScopedAnts = [ ]
    fullExpr = coref( fullExpr, Anaphs, ScopedAnts )
    if VERBOSE:  print( prettyForm(fullExpr) )
    print( nArticle, 'COREFD:', prettyForm(fullExpr) )
  
    if VERBOSE:  print( '----- simplify -----' )
    simplify( fullExpr )
    print( nArticle, 'LOGIC:', prettyForm(fullExpr) )
  
    checkUnboundVars( fullExpr )
  
  if '!ARTICLE' not in line:
    break


  '''
  ## Process trees given anaphs...
  for nLine,t in enumerate( Trees ):

#    sys.stderr.write( '========== Article ' + str(nArticle) + ' Tree ' + str(nLine) + ' ==========\n' )
    print( '========== Article ' + str(nArticle) + ' Tree ' + str(nLine) + ' ==========' )
    print( t )
  
    print( '----------' )
    if VERBOSE:  print( 'Scopes', Scopes )
    shortExpr = translate( t, Scopes, Anaphs )
    if t.qstore != []:
      print( 'ERROR: nothing in quant store', t.qstore, 'allowed by scope list', Scopes )
      exit(0)
    print( prettyForm(shortExpr) )
  
    if VERBOSE:  print( '----------' )
    fullExpr = [ unpack(shortExpr), Univ, Univ ]
    if VERBOSE:  print( fullExpr )

    if VERBOSE:  print( '----------' )
    betaReduce( fullExpr )
    if VERBOSE:  print( fullExpr )

    print( '----------' )
    simplify( fullExpr )
    print( prettyForm(fullExpr) )

  if '!ARTICLE' not in line:
    break
  '''


