import sys
import re

def getArity ( cat ):
  while '{' in cat:
    cat = re.sub('\{[^\{\}]*\}','X',cat)
  return len(re.findall('-[abcd]',cat))

################################################################################

class Sign:

  def __init__ ( this, skTemp=None, lTemp=None ):
    this.sk = skTemp  ## set of contexts
    this.l  = lTemp   ## category label

  def getArity ( this ):
    return len(re.findall('-[abcd]',this.l))                                                                  if this.l!=None else\
           max([0] + [int(k[-1]) for k in this.sk if len(k)>=2 and k[-2]=='-' and k[-1]>='0' and k[-1]<='9'])

  def __repr__ ( this ):
    return str(this.sk) + ':' + str(this.l)

################################################################################

class IncompleteSign:

  def __init__ ( this, aTemp=None, bTemp=None ):
    this.a = aTemp  ## apex
    this.b = bTemp  ## brink

  def __repr__ ( this ):
    return str(this.a) + '/' + str(this.b)

################################################################################

class StoreState (list):

  top = Sign([],'T')

  def __init__ ( q_t, q_tdec1=None, f=0, j=0, opL='', opR='', lA='', lB='', pretrm=None ):   # parent=None, rchild=None ):
    if q_tdec1!=None:
      lchild = q_tdec1.getLchild(f,pretrm)
      parent = Sign( ( [ ]                 if j==0 or len(q_tdec1)<2-f else\
                       q_tdec1[f-2].b.sk )  # otherwise
                     + ( [k[:-1]+'1' for k in lchild.sk if k[-1]=='0']               if lchild.l=='N-b{N-aD}' else\
                         lchild.sk                                                   if opL=='I'      else\
                         [k+'-'+opL  for k in lchild.sk if len(k)>=2 and k[-2]!='-'] if opL.isdigit() else\
                         [k[:-1]+'1' for k in lchild.sk if k[-1]=='0'] ),             # opL=='M'
                     lA )
      rchild = Sign( ( parent.sk                                                      if lchild.l=='N-b{N-aD}' else\
                       parent.sk                                                      if opR=='I'      else\
                       [k[:-1]+opR for k in parent.sk if k[-1]=='0']                  if opR.isdigit() else\
                       [k+'-1'     for k in parent.sk if len(k)>=2 and k[-2]!='-'] ),  # opR=='M'
                     lB )
      if j==0 and (opR=='I' or lchild.l=='N-b{N-aD}'): parent.sk = ['\"']
    list.__init__( q_t, [ ]                                                                        if q_tdec1==None else\
                        q_tdec1[:-2] + [ IncompleteSign(Sign(parent.sk,q_tdec1[-2].a.l), rchild) ] if f==0 and j==1 and '\"' in q_tdec1[-2].a.sk and opR!='I' else\
                        q_tdec1[:-2] + [ IncompleteSign(q_tdec1[-2].a,                   rchild) ] if f==0 and j==1 else\
                        q_tdec1[:-1] + [ IncompleteSign(parent,                          rchild) ] if f==0 and j==0 else\
                        q_tdec1[:-1] + [ IncompleteSign(Sign(parent.sk,q_tdec1[-1].a.l), rchild) ] if f==1 and j==1 and '\"' in q_tdec1[-1].a.sk and opR!='I' else\
                        q_tdec1[:-1] + [ IncompleteSign(q_tdec1[-1].a,                   rchild) ] if f==1 and j==1 else\
                        q_tdec1      + [ IncompleteSign(parent,                          rchild) ] )  # f==1 and j==0 

  def getBrink ( q ):
    return q[-1].b        if len(q)>0 else\
           StoreState.top  # len(q)==0

  def getAncstr ( q, f ):
    return q[-2+f].b      if len(q)>=2-f else\
           StoreState.top  # len(q)<2-f

  def getLchild ( q, f, pretrm ):
    return pretrm                               if f==1 else\
           StoreState.top                       if len(q)==0 else\
           q[-1].a                              if '"' not in q[-1].a.sk else\
           Sign(q[-1].b.sk+pretrm.sk,q[-1].a.l)  # '"'   in   q[-1].a.sk


  def calcForkPredictors ( q_tdec1 ):   #, f, pretrm ):
    return [ 'd'+str(len(q_tdec1))+'&'+k_b+'=1' for k_b in (['top'] if len(q_tdec1)==0 else ['bot'] if len(q_tdec1[-1].b.sk)==0 else q_tdec1[-1].b.sk) ]

  def calcJoinPredictors ( q_tdec1, f, pretrm ):   #, parent, rchild ):
    ancstr = q_tdec1.getAncstr ( f )
    lchild = q_tdec1.getLchild ( f, pretrm )
    return [ 'd'+str(len(q_tdec1)+f)+'&'+k_b+'&'+k_l+'=1' for k_b in (ancstr.sk if len(ancstr.sk)>0 else ['top']) for k_l in (lchild.sk if len(lchild.sk)>0 else ['bot']) ]

  def calcPretrmCatPredictors ( q_tdec1, f, k_p_t ):
    return ( str(len(q_tdec1)), str(f), q_tdec1.getBrink().l, k_p_t.partition(':')[0] )

  def calcApexCatPredictors ( q_tdec1, f, j, opL, pretrm ):
    return ( str(len(q_tdec1)), str(f), str(j), opL, q_tdec1.getAncstr(f).l, q_tdec1.getLchild(f,pretrm).l if j==0 else '-' )

  def calcBrinkCatPredictors ( q_tdec1, f, j, opL, opR, pretrm, lParent ):
    return ( str(len(q_tdec1)), str(f), str(j), opL, opR, lParent, q_tdec1.getLchild(f,pretrm).l )


#def getParentContexts ( opL, arityA, skLchild ):
#  return skLchild                                                            if opL=='I' else\
#         [k+'-'+str(arityA+1) for k in skLchild if len(k)>=2 and k[-2]!='-'] if opL=='A' else\
#         [k[:-1]+'1'          for k in skLchild if k[-1]=='0']                # opL=='M'


def getUnkWord ( w ):
  return '!unk!'+('ing' if w.endswith('ing') else\
                  'ed'  if w.endswith('ed') else\
                  's'   if w.endswith('s') else\
                  'ion' if w.endswith('ion') else\
                  'er'  if w.endswith('er') else\
                  'est' if w.endswith('est') else\
                  'ly'  if w.endswith('ly') else\
                  'ity' if w.endswith('ity') else\
                  'y'   if w.endswith('y') else\
                  'al'  if w.endswith('al') else\
                  'cap' if w[0].isupper() else\
                  'num' if w[0].isdigit() else\
                  '')    # otherwise
