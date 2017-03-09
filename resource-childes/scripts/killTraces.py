import sys, os, re
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
import tree

def kill_traces(T):
    if len(T.ch) == 1 and len(T.ch[0].ch) == 0:
        if T.ch[0].c.startswith('*'):
            tmp = T
            while len(tmp.p.ch) == 1:
              tmp = tmp.p
            i = tmp.p.ch.index(tmp)
            tmp.p.ch[i] = 'TOKILL'
    else:
        for x in T.ch:
            kill_traces(x)
        done = False
        while not done:
            try:
                T.ch.remove('TOKILL')
            except ValueError:
                done = True
        if len(T.ch) == 0:
            i = T.p.ch.index(T)
            T.p.ch[i] = 'TOKILL'

t = tree.Tree()
for line in sys.stdin:
    if (line.strip() !=''):
        t.read(line)
        kill_traces(t)
        print(t)
