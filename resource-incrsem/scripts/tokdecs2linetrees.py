import os, sys, re
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
from tree import Tree

SS = [ ]   ## derivation fragments (partial trees) derived at each storestate

IDENTIFY_EXTRACTIONS = False

def attachB(t,lu):
  if len(t.ch)>1: attachB(t.ch[1],lu)
  else:           t.ch = lu
  return t

def annotB(t,lu):
  if len(t.ch)>1: annotB(t.ch[1],lu)
  else:           t.c += lu
  return t

for s in sys.stdin:
    s = re.sub( '/[^ ;/]*\^;[^ ;/]*', '', s )   ## remove bottom carriers. (OBSOLETE)
    s = re.sub( '(?<=.);[ \n]', ' ', s )        ## remove trailing ;
    s = re.sub( '/ ', ' ', s )                  ## remove trailing /
    s = re.sub( '/([^; ]*)/', '/', s )          ## remove A carriers
    s = re.sub( ';([^/ ]*);', ';', s )          ## remove B carriers
    m = re.search('^([^ ]*) (?!pos)(?:\[[^\]]*\]:)?([^ ]*) ([^ ]*) ([^ ]*) (?:.*;)?([^ ;\n]*)[ \n]',s)
    if m is not None:
        w,p,f,j,q = m.groups()

        ## remove berk stuff (not needed after berk removed)...
        p = re.sub( '_[0-9]+', '', p )

        ## annotate operator tags if semproc...
        if len(f.split('&'))>1:
          bF,eF,_ = f.split('&')
          if eF!='' and bF=='f1' and IDENTIFY_EXTRACTIONS: p            += '-l' + eF
          if eF!='' and bF=='f0' and IDENTIFY_EXTRACTIONS: annotB( SS[-1], '-l' + eF )

        if f=='1' or f.startswith('f1'): SS.append( Tree(p,[Tree(w)]) )
        else:                            attachB( SS[-1], [ Tree(p,[Tree(w)]) ] )

        if q=='':
            if len(SS)==1: print( SS[0] )
            else: print( 'ERROR: not a valid tree!' )
            SS = [ ]

        else:
            a,b = re.split( '/' , re.sub( '\[[^\]]*\]:', '', q ) )   # get a,b from last q

            ## remove berk stuff (not needed after berk removed)...
            a = re.sub( '_[0-9]+', '', a )
            b = re.sub( '_[0-9]+', '', b )

            # print( w, p, f, j, a, b )

            ## annotate operator tags if semproc...
            if len(j.split('&'))>1:
              bJ,eJ,oL,oR = j.split('&')
              if eJ!='' and bJ=='j1' and IDENTIFY_EXTRACTIONS: annotB( SS[-2], '-l' + eJ )
              if eJ!='' and bJ=='j0' and IDENTIFY_EXTRACTIONS: a            += '-l' + eJ
              if oL!='I' and oL!='m' or oR=='m'              : SS[-1].c     += '-l' + ('A' if oL>='1' and oL<='9' else 'A' if oR=='m' else oL)
              if oR!='I' and oR!='m' or oL=='m'              : b            += '-l' + ('A' if oR>='1' and oR<='9' else 'A' if oL=='m' else oR)

            if j=='1' or j.startswith('j1'): SS = SS[:-2] + [ attachB( SS[-2], [ SS[-1], Tree(b) ] ) ]
            else:                            SS = SS[:-1] + [ Tree( a, [ SS[-1], Tree(b) ] ) ]

            # for t in SS: print( ' ... ' + str(t) )

