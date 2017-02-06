import sys
import re
import collections
sys.path.append('../resource-gcg/scripts')
import tree

for line in sys.stdin:
  Q = [ ]
  f = 0
  j = 0
  p = ''
  w = ''

  def calccontext ( tr, s=1, d=0 ):
    global Q, f, j, p, w

    #### at preterminal node...
    if len(tr.ch)==1 and len(tr.ch[0].ch)==0:
      f = 1 - s
      p = tr.c
      w = tr.ch[0].c

    #### at non-preterminal unary node...
    elif len(tr.ch)==1:
      calccontext ( tr.ch[0], s, d )

    #### at binary nonterminal node...
    elif len(tr.ch)==2:
      ## traverse left child...
      calccontext ( tr.ch[0], 0, d if s==0 else d+1 )
      j = s
      #print( d, f, j, Q )
      #Q = ( Q[:d+f-j-2] if d+f-j>=2 else [ ] ) + ( [ ( Q[d+f-j-1].partition('/')[0] if j==1 else tr.c ) + '/' + tr.ch[1].c ] if d+f-j>=1 else [ ] )
      Q = ( Q[:d-1] if d>=1 else [ ] ) + ( [ ( Q[d-1].partition('/')[0] if j==1 else tr.c ) + '/' + tr.ch[1].c ] if d>=1 else [ ] )
      print ( w + ' ' + p + ' ' + str(f) + ' ' + str(j) + ' ' + ';'.join(Q) )
      ## traverse right child...
      t = calccontext ( tr.ch[1], 1, d )


  # print line
  #print ( line )
  tr = tree.Tree('T',[tree.Tree(),tree.Tree('T')])
  tr.ch[0].read ( line )
  calccontext ( tr )
  #print ( str(f) + ' ' + str(j) + ' ' + ';'.join(Q) + ' ' + p )

