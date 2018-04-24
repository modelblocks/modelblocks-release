import os, sys, re
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
from tree import Tree

SS = [ ]

def attachB(t,lu):
  if len(t.ch)>1: attachB(t.ch[1],lu)
  else:           t.ch = lu
  return t

for s in sys.stdin:
    s = re.sub( '/[^ ;/]*\^;[^ ;/]*', '', s )   ## remove bottom carriers. (OBSOLETE)
    s = re.sub( ';[ \n]', ' ', s )                   ## remove trailing ;
    s = re.sub( '/ ', ' ', s )                  ## remove trailing /
    s = re.sub( '/([^; ]*)/', '/', s )          ## remove A carriers
    s = re.sub( ';([^/ ]*);', ';', s )          ## remove B carriers
    m = re.search('^([^ ]*) (?!pos)(?:\[[^\]]*\]:)?([^ ]*) ([^ ]*) ([^ ]*) (?:.*;)?([^ ;\n]*)[ \n]',s)
    if m is not None:
        w,p,f,j,q = m.groups()

        if f=='1' or f.startswith('f1'): SS.append( Tree(p,[Tree(w)]) )
        else:                            attachB( SS[-1], [ Tree(p,[Tree(w)]) ] )

        if q=='':
            if len(SS)==1: print( SS[0] )
            else: print( 'ERROR: not a valid tree!' )
            SS = [ ]

        else:
            a,b = re.split( '/' , re.sub( '\[[^\]]*\]:', '', q ) )   # get a,b from last q

            # print( w, p, f, j, a, b )

            if j=='1' or j.startswith('j1'): SS = SS[:-2] + [ attachB( SS[-2], [ SS[-1], Tree(b) ] ) ]
            else:                            SS = SS[:-1] + [ Tree( a, [ SS[-1], Tree(b) ] ) ]

            # for t in SS: print( ' ... ' + str(t) )

