
import re
import sys
sys.path.insert(0, '../resource-gcg/scripts/')
from tree import Tree

def A(q):
    return q.split('/')[0]
def W(q):
    return q.split('/')[1]
def depth(S):
    return ( -1 if '-/-'==S[0] else
             0 if '-/-'==S[1] else
             1 if '-/-'==S[2] else
             2 if '-/-'==S[3] else
             3 )
D = 4

def act(t,d,F,Q,X):
    #sys.stdout.write('t: '+str(t)+' ')
    #print('A',t,d,'f='+F[t],depth(Q[t-1]),Q[t][d],Q[t][d+1],Q[t][D],X[t])
    # if initial element in row, add lex item
    if depth(Q[t])==d-1:
        #print('  lex')
        return Tree(Q[t][D],[X[t]])
    # if result of active transition, add branch, building tree from bottom up
    if ( '1'==F[t] and depth(Q[t-1])==d or
         '0'==F[t] and depth(Q[t-1])==d-1 ):
        #print('  branch',A(Q[t][d]))
        return Tree(A(Q[t][d]),[act(t-1,d,F,Q,X),awa(t,d,F,Q,X)])
    # otherwise recurse backward along active components in row
    else:
        return act(t-1,d,F,Q,X)

def awa(t,d,F,Q,X):
    #print('W',t,d,'f='+F[t+1],depth(Q[t+1]),Q[t+1][d],Q[t+1][d+1],Q[t][D],X[t])
    # if reduction at current depth, add lex item
    if ( '1'==F[t+1] and depth(Q[t])==d ):
        #print('  lex')
        return Tree(Q[t][D],[X[t]])
    # if source of awaited transition, add branch, building tree from bottom up
    if depth(Q[t+1])==d:
        #print('  branch',W(Q[t][d]))
        return Tree(W(Q[t][d]),[act(t,d+1,F,Q,X),awa(t+1,d,F,Q,X)])
    # otherwise recurse forward along awaited components in row
    else:
        return awa(t+1,d,F,Q,X)

T = 0
F = {}
Q = {}
X = {}

for s in sys.stdin:
    #sys.stdout.write(s)
    #sys.stderr.write('s: '+str(s))
    m = re.search('([^ ]+) +([^;]+);+([^ ]+) +([^ \n]*)',s)
    if m is not None:
        (st, hf, hq, x) = m.groups()
        t = int(st)
        T = max(T,t)
        F[t] = hf
        Q[t] = []
        myq = []
        emod = ''
        for e in hq.split(';'):
            # handle when a real semi-colon should be in the output
            e = emod + e
            if e == '':
                #if act of q elem should have a ;, add it
                emod = ';'
            elif myq != [] and myq[-1][-1] == '/':
                # if awa of q elem should have a ;, add it
                emod = ''
                myq[-1] = myq[-1] + ';' + e
            else:
                emod = ''
                myq.append(e)
        for iq,q in enumerate(myq):
            #strip out entity info from efabp
            if iq < D:
                # q elems
                #sys.stderr.write('  q: '+str(q)+'\n')
                qa,qb = q.split('/')
                Q[t].append(qa[:qa.rfind('.')] +'/'+ qb[:qb.rfind('.')])
            elif iq == D:
                # preterminal
                #sys.stderr.write('  p: '+str(q)+'\n')
                Q[t].append(q[:q.rfind('.')])
        X[t] = x
    m = re.search('----------',s)
    if m is not None:
        #print(Q)
        if T<=1: tr = Tree('FAIL',[Tree('fail')]) #act() assumes T-1 != 0
        else:    tr = act(T-1,0,F,Q,X)
        print(tr)
        T=0
        #F = {}
        #Q = {}
        #X = {}
        #exit(0)
