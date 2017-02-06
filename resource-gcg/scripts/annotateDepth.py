import sys
import tree

def annotDepth(t,b='L',d=1):
    if len(t.ch)>0:
        t.c += '-b'+b+'-d'+str(d)
    if len(t.ch)==1:
        annotDepth(t.ch[0], b, d)
    if len(t.ch)==2:
        if b=='L':
            annotDepth(t.ch[0],'L',d)
            annotDepth(t.ch[1],'R',d)
        else:
            annotDepth(t.ch[0],'L',d+1)
            annotDepth(t.ch[1],'R',d)

for s in sys.stdin:
    if s == '\n':
        print ( '' )
    else:
        t = tree.Tree()
        t.read(s)
        annotDepth(t)
        print ( str(t) )

