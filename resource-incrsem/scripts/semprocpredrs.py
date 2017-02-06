import sys
import re
import collections
sys.path.append('../resource-gcg/scripts')
import tree
from storestate import Sign, StoreState, getArity   #, getParentContexts, getRchildContexts

def arityGivenCat(cat):
  arity = getArity(cat)
  return arity+1 if cat[0]=='N' and not cat.startswith('N-b{N-aD}') else arity

def getOp(cat,siblingcat,parentcat):
  if '-l' not in cat:                        return 'I'
  if '-lM' in cat:                           return 'M'
  if '-lA' in cat and '\\' in parentcat:     return str(arityGivenCat(parentcat.partition('\\')[2]))
  if '-lA' in cat and '\\' not in parentcat: return str(arityGivenCat(siblingcat))
  sys.stderr.write('WARN: illegal cat in '+cat+'\n')
  return 'I'

def getBaseForm(cat, w):
  baseCat = ""
  m = re.match("(.*)\-xX[^\*]*\*([^\*]*)\*([^ ]*)", cat)
  if m != None and m.group(1) != None and m.group(2) != None and m.group(3) != None:
    if m.group(1)[0] in ["V","B","L","G"]:
      baseCat = "B"+m.group(1)[1:]
    baseW = re.sub(m.group(3)+'$',m.group(2),w) #w[:-len(m.group(3))]+m.group(2)
    return (baseCat, baseW)
  else:
    return (cat, w)


def calccontext ( tr, Cx, t=0, s=0, d=0 ):

  #### at preterminal node...
  if len(tr.ch)==1 and len(tr.ch[0].ch)==0:

    ## increment time step at word encounter...
    t += 1
    ## set fork value for operation after word at t...
    Cx[t,'f']=1-s
    ## calc Cx[t,d,s]...
    if tr.c == ':': tr.c = 'Pk'
    if tr.ch[0].c[0].isdigit(): tr.ch[0].c = '!num!'
    if ',' in tr.ch[0].c: tr.ch[0].c = '!containscomma!'
    category,predicate = getBaseForm ( tr.c, tr.ch[0].c.lower() if all(ord(c) < 128 for c in tr.ch[0].c) else '!loanword!' )
    Cx[t,'p'] = Sign( [ re.sub(':','Pk',category) + ':' + predicate + ('_1' if category.startswith('N') and not category.startswith('N-b') else '_0') ], tr.c ) if tr.c[0]>='A' and tr.c[0]<='Z' else\
                Sign( [ ], re.sub(':','Pk',category) )

    print 'F', ','.join(Cx[t-1].calcForkPredictors()), ':', ('f1' if Cx[t,'f']==1 else 'f0') + '&' + (Cx[t,'p'].sk[0] if len(Cx[t,'p'].sk)>0 else 'bot')
    print 'P', ' '.join(Cx[t-1].calcPretrmCatPredictors(Cx[t,'f'],Cx[t,'p'].sk[0] if len(Cx[t,'p'].sk)>0 else 'bot')), ':', Cx[t,'p'].l
    print 'W', (Cx[t,'p'].sk[0] if len(Cx[t,'p'].sk)>0 else 'bot'), Cx[t,'p'].l, ':', tr.ch[0].c if all(ord(c) < 128 for c in tr.ch[0].c) else '!loanword!'
#    ## if f==1, merge p contexts into b...
#    if Cx[t,'f']==0:
#      Cx[t-1][-1].b.sk += Cx[t,'p'].sk
#      if '"' in Cx[t-1][-1].a.sk: Cx[t-1][-1].a.sk = Cx[t-1][-1].b.sk

  #### at non-preterminal unary node...
  elif len(tr.ch)==1:

    t = calccontext ( tr.ch[0], Cx, t, s, d )

  #### at binary nonterminal node...
  elif len(tr.ch)==2:

    ## traverse left child...
    t = calccontext ( tr.ch[0], Cx, t, 0, d if s==0 else d+1 )

    f = Cx[t,'f']
    j = s
    pretrm = Cx[t,'p']
    ancstr = Cx[t-1].getAncstr(f)
    lchild = Cx[t-1].getLchild(f,pretrm)
    opL    = getOp( tr.ch[0].c, tr.ch[1].c, tr.c )
    opR    = getOp( tr.ch[1].c, tr.ch[0].c, tr.c )
    print 'J', ','.join(Cx[t-1].calcJoinPredictors(f,pretrm)), ':', ('j1' if j==1 else 'j0') + '&' + opL + '&' + opR
    print 'A', ' '.join(Cx[t-1].calcApexCatPredictors(f,j,opL,pretrm)),       ':', re.sub('-l.','',tr.c)
    print 'B', ' '.join(Cx[t-1].calcBrinkCatPredictors(f,j,opL,opR,pretrm,re.sub('-l.','',tr.c))), ':', re.sub('-l.','',tr.ch[1].c)
    Cx[t] = StoreState( Cx[t-1], f, j, opL, opR, re.sub('-l.','',tr.c), re.sub(':','Pk',re.sub('-l.','',tr.ch[1].c)), pretrm )
#    print str(Cx[t-1]) + ' ===(' + str(lchild.sk) + ')==> ' + str(Cx[t])

    ## traverse right child...
    t = calccontext ( tr.ch[1], Cx, t, 1, d )

  return t



for line in sys.stdin:
  # print line
  tr = tree.Tree()
  tr.read ( line )
  Cx = collections.defaultdict(list)
  Cx[0] = StoreState()
  t = calccontext ( tr, Cx )
  print 'J', ','.join(Cx[t-1].calcJoinPredictors(Cx[t,'f'],Cx[t,'p'])), ': j1&S&I'
  # for k in sorted(Cx):
  #   print str(k) + str(Cx[k])
