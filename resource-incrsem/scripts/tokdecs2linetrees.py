import os, sys, re
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
from tree import Tree

SS = [ ]   ## derivation fragments (partial trees) derived at each storestate

IDENTIFY_EXTRACTIONS = True #False

## attach children to base category of derivation fragment...
def attachB(t,lu):
  if len(t.ch)>0: attachB(t.ch[-1],lu)
  else:           t.ch = lu
  return t

## modify base category of derivation fragment...
def annotB(t,lu):
  if len(t.ch)>0: annotB(t.ch[-1],lu)
  else:           t.c += lu
  return t

## return base category of derivation fragment...
def getB(t):
  return getB(t.ch[-1]) if len(t.ch)>0 else t.c

## calculate list of projected categories (top down) given extraction outcome and ancestor category...
def unaryprojlist(e,c):
  l = [ ]
  for o in e:
    if o=='V':            c = re.sub( '^A-aN(.*)$', 'L-aN\\1-vN', c )
    if o=='O':            c = re.sub( '^([^-]*)(-[ab][^-{}]*|-[ab]\{[^{}]*\})(-[ab][^-{}]*|-[ab]\{[^{}]*\})(.*)$', '\\1\\3\\2\\4-lQ', c )
    if o>='0' and o<='9': c = re.sub( '^(.*)-[gh](.*?)$', '\\1-b\\2-lE', c )  ## NOTE: ALWAYS -b BC CANNOT KNOW AND -a CAUSES -lQ TO WRONGLY ATTACH
    if o=='M':            c = re.sub( '^(.*)-[gh](.*?)$', '\\1-lE', c )
    l += [ c ]
  return l

for s in sys.stdin:
    if s == '!ARTICLE\n':
      print( '!ARTICLE' )
      continue
    s = re.sub( '/[^ ;/]*\^;[^ ;/]*', '', s )   ## remove bottom carriers. (OBSOLETE)
    s = re.sub( '(?<=.);[ \n]', ' ', s )        ## remove trailing ;
    s = re.sub( '/ ', ' ', s )                  ## remove trailing /
    s = re.sub( '/([^; ]*)/', '/', s )          ## remove A carriers
    s = re.sub( ';([^/ ]*);', ';', s )          ## remove B carriers
    s = re.sub( '\]\[', '', s )                 ## remove synarg divisions in context lists
    m = re.search('^([^ ]*) (?!pos)(?:\[[^\]]*\]:)?([^ ]*) ([^ ]*) ([^ ]*) (?:.*;)?([^ ;\n]*)[ \n]',s)
    if m is not None:
        w,p,f,j,q = m.groups()

        ## remove berk stuff (not needed after berk removed)...
        p = re.sub( '_[0-9]+', '', p )

        treeW = Tree(w)

        ## annotate operator tags if semproc...
        # if '&' in f:
        #   bF,eF,kF = f.split('&')
        #   treeW = Tree( re.sub(':.*','',kF), [ re.sub('.*:(.*)_.*','\\1',kF) ] )
        #   for c in reversed( unaryprojlist(eF,p) ):
        #     treeW = Tree( c, [treeW] )

        ## apply fork decision...
        if f=='1' or f.startswith('f1'): SS.append( Tree(p,[treeW]) ) #[Tree(w)]) )
        else:                            attachB( SS[-1], [ Tree(p,[treeW]) ] ) #[Tree(w)]) ] )

        ## if store empty, complete tree...
        if q=='':
            if len(SS)==1: print( SS[0] )
            else: print( 'ERROR: not a valid tree!' )
            SS = [ ]

        ## if store no empty...
        else:
            a,b = re.split( '/' , re.sub( '\[[^\]]*\]:', '', q ) )   # get a,b from last q

            ## remove berk stuff (not needed after berk removed)...
            a = re.sub( '_[0-9]+', '', a )
            b = re.sub( '_[0-9]+', '', b )

            ## annotate operator tags if semproc...
            kids = [ SS[-1], Tree(b) ]
            if '&' in j:
              bJ,eJ,oL,oR = j.split('&')
              ## calc op tags for left and right children...
              if oL!='.' and oL!='U' and oL!='u' or oR=='U' or oR=='u' : SS[-1].c     += '-l' + ('A' if oL>='1' and oL<='9' else 'U' if oR=='U' or oR=='u' else oL)
              if oR!='.' and oR!='U' and oR!='u' or oL=='U' or oL=='u' : b            += '-l' + ('A' if oR>='1' and oR<='9' else 'U' if oL=='U' or oL=='u' else oR)
              ####sys.stderr.write( j + ' trying SS[-2]:' + ( str(SS[-2]) if len(SS)>1 else '')  + ' SS[-1]:' + (str(SS[-1]) if len(SS)>0 else '') + ' a:' + a + ' b:' + b + '\n' )
              kids = [ SS[-1], Tree(b) ]  # bc b updated
              for c in reversed( unaryprojlist( eJ, getB(SS[-2]) if j.startswith('j1') else a ) ):
                kids = [ Tree( c, kids ) ]

            ## apply join decision...
            if j=='1' or j.startswith('j1'): SS = SS[:-2] + [ attachB( SS[-2], kids ) ]
            else:                            SS = SS[:-1] + [ Tree( a, kids ) ]


