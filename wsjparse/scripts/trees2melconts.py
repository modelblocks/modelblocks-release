import fileinput
import re
import tree
import model
import quants
import argparse
import sys

parser = argparse.ArgumentParser(description='Extract features for maxent learner.')
parser.add_argument('-d', '--debug', default=False, required=False, help='print annotated tree')
parser.add_argument('-p', '--psg', action="store_true", default=False, help='assume psg style grammar')
parser.add_argument('-r', '--reducedFeats', action="store_true", default=False, help='assume not to have maxproj, leftRightChild, parent, grandparent, sibling features')
parser.add_argument('-t', '--treeFile', type=argparse.FileType('r'), required=True, help='cnftrees')
parser.add_argument('-c', '--conjunctionHeadShift', action="store_true", default=False, required=False, help='Shift the head of the conjunction from the last conjunct to the conjunction word')
opts = parser.parse_args()

if opts.psg:
    sys.stderr.write('*** using psg mode ***\n')
else:
    sys.stderr.write('*** using non-psg mode ***\n')

# print tree with referents
def printRefs(t):
    if len(t.ch)==1:
        print('('+t.c+' '+t.e+' '+t.ch[0].c+')', end='')
    else:
        print('('+t.c+' '+t.e+' ', end='')
        for st in t.ch:
            printRefs(st)
        print(') ', end='')

#find the common parent in the 2 lists p1Lst and p2Lst. Also return the index of that common parent
#on p2Lst. p1Lst should be ancestors of proposition. p2Lst should be that of argument. Take the index
#of the common parent minus 1 will be the maximum projection of the proposition on the argument  
def commonParent(p1Lst, p2Lst):
    for p1 in p1Lst:
        for idx, p2 in enumerate(p2Lst):
            if p1 == p2:
                return idx, p2
    print('No crossing ancestors line', ln)
    return -1, None

ancestorsMap = {}
def getAncestors(t):
    if t not in ancestorsMap:
        ancestorsMap[t] = t.getAncestors()
    return ancestorsMap[t]

def getRefList(lst):
    if lst == ['-']:
        return []
    return list(map(lambda x: x.e if isinstance(x, tree.Tree) else x, lst))

# print propositional content of role-label-annotated tree
def interpret_psg(t,args=[],gaps=[],cons=[]):
    props = []

    if opts.debug:
        print(t.c, '('+t.e+')', '&','a =', getRefList(args),', g =', getRefList(gaps), ', c =', getRefList(cons))

    # if gaps not propagated down, add gap referent to args
    if re.search('-g',t.c) != None:
        if quants.none_of ( t.ch, lambda st: re.search('-g',st.c) ):
            if re.search('-gNP|-g.S|-g.E',t.c) != None:
                args += gaps
            else:
                for g in gaps:
                    props.append( (g.e, 1, t.e+'#-') )
            gaps = []

    # if terminal, print prop
    if len(t.ch)==1:
        if gaps != []: args += gaps
        if cons != []: args = cons

        if opts.debug:
            print('Leaf', t.c, '('+t.e+')', '&','a =', getRefList(args))
        if not opts.reducedFeats:
            leftOrRightChild='#0' if t.p == None or t.p.ch[0] == t else '#1' 
            parent = '#' + (t.p.c if t.p else '0')
            grandParent = '#' + (t.p.p.c if t.p and t.p.p else '0') 
            sibling = '#' + ('0' if t.p == None else (t.p.ch[1].c if t.p.ch[0] == t else t.p.ch[0].c))
        else:
            leftOrRightChild='#LR' 
            parent = '#PARENT'
            grandParent = '#GPARENT' 
            sibling = '#SIBLING'
        
        cat = re.sub('\-l.$', '', t.c)
        props.append( (t.e, 0, cat + '#' + (t.ch[0].c if len(t.ch[0].ch)==0 else t.ch[0].ch[0].c if len(t.ch[0].ch[0].ch)==0 else t.ch[0].ch[0].ch[0].c) + leftOrRightChild + parent + grandParent + sibling) )
        if not opts.reducedFeats:
            tAncestors = getAncestors(t)
            for a in range(0,len(args)):
                if args[a]!='-':
                    maxProj = '#'
                    for idx, p in enumerate(tAncestors):
                        if args[a] == p:
                            maxProj += tAncestors[idx-1].sibling().c
                            break
                    else:
                        maxProj += args[a].c
                    props.append( (t.e, a+1, args[a].e + maxProj) )
        else:
            for a in range(0,len(args)):
                if args[a]!='-':
                    props.append( (t.e, a+1, args[a].e + '#MAXPROJ' ) )

    # if non-terminal
    else:
        # define child args,gaps,cons
        child_args = model.ListModel('')
        child_gaps = model.ListModel('')
        child_cons = model.ListModel('')

        # pass accumulated argument referents to head (or conjunct)
        for b in range(0,len(t.ch)):
            if re.search('-l[IC]',t.ch[b].c) != None:
                child_args[b] = args[0:]

        # pass accumulated conjunct referents to head
        for b in range(0,len(t.ch)):
            if re.search('-lI',t.ch[b].c) != None:
                child_cons[b] = cons[0:]

        # pass non-gapped child referent to newly-gapped child
        for b in range(0,len(t.ch)):
            # if child has gap/relpro tag
            if re.search('-g|-oR',t.ch[b].c) != None:
                # if parent has gap/relpro tag and no sibling or sibling has no relpro tag, pass gap fillers down from parent
                if re.search('-g|-oR',t.c) != None and (len(t.ch)<2 or re.search('-oR',t.ch[1-b].c) == None):
                    child_gaps[b] = gaps[0:]
                # otherwise, if sibling not AC gap, add sibling as child's gap filler
                else:
                    if re.search('-gAC',t.ch[1-b].c) == None:
                        child_gaps[b] = [ t.ch[1-b] ]

        # pass new argument referent to head
        for b in range(0,len(t.ch)):
            if re.search ('-lA',t.ch[b].c) != None:
                child_args[1-b] += [ t.ch[b] ]
                if re.search('^(VP|IP|BP|LP|AP|RP|GP|NP|NN)',t.ch[b].c) != None:
                    child_args[b] += [ '-' ] #[ args[0] if len(args)>0 else '-' ]
        # no first argument for non-related phrases
        for b in range(0,len(t.ch)):
            if re.search ('-lN',t.ch[b].c) != None:
                # phrasal projections need placeholder for missing subj; NP.*-oR nominal rel pro needs placeholder so antecedent will always be second arg
                if re.search('(^VP|^IP|^BP|^LP|^AP|^RP|^GP|^NP.*-oR)',t.ch[b].c) != None:
                    child_args[b] += [ '-' ]

        # pass new conjunct referent to head
        for b in range(0,len(t.ch)):
            if re.search ('-lC',t.ch[b].c) != None:
                child_cons[1-b] += [ t.ch[b] ]

        # pass head referent to modifier
        for b in range(0,len(t.ch)):
            if re.search ('-lM',t.ch[b].c) != None  :
                child_args[b] += [ t ]

        # recurse to children
        for b in range(0,len(t.ch)):
            props += interpret_psg(t.ch[b],child_args[b],child_gaps[b],child_cons[b])

        # apply post-hoc .......
        for b in range(0,len(t.ch)):
            # for object control (A1 of embedded clause is A3 of main clause; in addition to embedded clause being A2 of main clause)
            if re.search('^Sto.*-lA|[AI]S.*-lA',t.ch[b].c)!=None:   ## didn't help
                for e1,l1,f1 in props:
                    if e1==t.ch[b].e and l1==1:
                        if (t.ch[1-b].e,3,f1) not in props:
                            props.append ( (t.ch[1-b].e,3,f1) )

    return props

# print propositional content of role-label-annotated tree
def interpret(t,args=[],gaps=[],cons=[]):
    props = []

    if opts.debug:
        print('==>',[t],t.e,t.c,'a=',args,'g=',gaps,'c=',cons)

    # if terminal, print prop
    if len(t.ch)==1:
        if gaps != []: args += gaps
        if cons != []: args = cons
        leftOrRightChild='#0' if t.p == None or t.p.ch[0] == t else '#1' 
        parent = '#' + (t.p.c if t.p else '0')
        grandParent = '#' + (t.p.p.c if t.p and t.p.p else '0') 
        sibling = '#' + ('0' if t.p == None else (t.p.ch[1].c if t.p.ch[0] == t else t.p.ch[0].c))
        
        cat = re.sub('\-l.$', '', t.c)
        props.append( (t.e, 0, cat + '#' + (t.ch[0].c if len(t.ch[0].ch)==0 else t.ch[0].ch[0].c if len(t.ch[0].ch[0].ch)==0 else t.ch[0].ch[0].ch[0].c) + leftOrRightChild + parent + grandParent + sibling) )
        tAncestors = getAncestors(t)
        for a in range(0,len(args)):
            if args[a]!='-':
                maxProj = '#'
                pathToArg = '#'
                shortPta = '#'
                for idx, p in enumerate(tAncestors):
                    if args[a] == p:
                        maxProj += tAncestors[idx-1].sibling().c
                        path = list(map(lambda x: x.c, tAncestors[0:idx+1] + [tAncestors[idx-1].sibling()]))
                        pathToArg += '_'.join( path )
                        shortPta += '_'.join( path[-3:] )
                        break
                    elif args[a].p == p:
                        maxProj += args[a].c
                        path = list(map(lambda x: x.c, tAncestors[0:idx+1] + [args[a]]))
                        pathToArg += '_'.join( path )
                        shortPta += '_'.join( path[-3:] )
                        break;
                else:
                    print("Unknown maxProj on line:", ln, file=sys.stderr)
                    maxProj += args[a].c
                    pathToArg += '_'.join( list(map(lambda x: re.sub('\-l[ACIMN]$', '', x.c), tAncestors[0:2] + [args[a]])))
                    shortPta = pathToArg
                props.append( (t.e, a+1, args[a].e + maxProj) )

    # if non-terminal
    else:
        # define child args,gaps,cons
        child_args = model.ListModel('')
        child_gaps = model.ListModel('')
        child_cons = model.ListModel('')

        # if gaps not propagated down, add gap referent to args
        if quants.none_of ( t.ch, lambda st: re.search('-g',st.c) ):
            args += gaps
            gaps = []

        # pass accumulated argument referents to head (or conjunct)
        for b in range(0,len(t.ch)):
            if re.search('-l[IC]',t.ch[b].c) != None:
                child_args[b] = args[0:]

        # pass accumulated conjunct referents to head
        for b in range(0,len(t.ch)):
            if re.search('-lI',t.ch[b].c) != None:
                child_cons[b] = cons[0:]

        # pass non-gapped child referent to newly-gapped child
        for b in range(0,len(t.ch)):
            if re.search('-g',t.ch[b].c) != None:
                if re.search('-g',t.c) != None:
                    child_gaps[b] = gaps[0:]
                else:
                    child_gaps[b] = [ t.ch[1-b] ]

        # pass new argument referent to head
        for b in range(0,len(t.ch)):
            if re.search ('-lA',t.ch[b].c) != None:
                child_args[1-b] += [ t.ch[b] ]
                if re.search('^(Spro|VP|ADJP|ADVP|PP)',t.ch[b].c) != None:
                    if t.ch[1-b].c[0]=='V':
                        child_args[b] = args[0:]
                    else:
                        child_args[b] += [ '-' ]

        # pass new conjunct referent to head
        for b in range(0,len(t.ch)):
            if re.search ('-lC',t.ch[b].c) != None:
                child_cons[1-b] += [ t.ch[b] ]

        # pass head referent to modifier
        for b in range(0,len(t.ch)):
            if re.search ('-lM',t.ch[b].c) != None  :
                child_args[b] += [ t ]

        # recurse to children
        for b in range(0,len(t.ch)):
            props += interpret(t.ch[b],child_args[b],child_gaps[b],child_cons[b])

        # apply post-hoc .......
        for b in range(0,len(t.ch)):
            # for Spro arguments to prepositions
            if ( re.search('^[^N]',t.c)!=None
                 and len(t.ch[b].ch)>1 and re.search('^Spro.*-lA',t.ch[b].ch[1].c)!=None ):
                for e1,l1,f1 in props:
                    if e1==t.e and l1==1:
                        if (t.ch[b].ch[1].e, 1, f1) not in props:
                            props.append ( (t.ch[b].ch[1].e, 1, f1) )
            # for Spro/ADJP arguments
            if re.search('-lI',t.c)!=None and re.search('^(Spro|ADJP).*-lA',t.ch[b].c)!=None:
                for e2,l2,f2 in props:
                    if e2==t.ch[1-b].e and l2==1:
                        if (t.ch[b].e,1,f2) not in props:
                            props.append ( (t.ch[b].e,1,f2) )
            # for Spro modifiers
            if re.search('^Spro.*-lM',t.ch[b].c)!=None:
                for e2,l2,f2 in props:
                    if e2==t.ch[1-b].e and l2==1:
                        if (t.ch[b].e,1,f2) not in props:
                            props.append ( (t.ch[b].e,1,f2) )
            # for object control (A1 of embedded clause is A3 of main clause; in addition to embedded clause being A2 of main clause)
            if re.search('^Sto.*-lA',t.ch[b].c)!=None:   ## didn't help
                for e1,l1,f1 in props:
                    if e1==t.ch[b].e and l1==1:
                        if (t.ch[1-b].e,3,f1) not in props:
                            props.append ( (t.ch[1-b].e,3,f1) )

    return props

def shiftConjunctionHeadToConjWord(mels):
    for idx, (e,l,f) in enumerate(mels):
        if l==0 and f[0:2] == 'CC': #found conjunction head
            lastConjunctIdx = -1
            lastConjunct = None
            for (e1,l1,f1) in mels:
                if e1==e and l1!=0 and int(f1.split('#')[0][1:]) > lastConjunctIdx:
                    lastConjunct = f1
                    lastConjunctIdx = int(f1.split('#')[0][1:])
            if lastConjunct:
                for idx, (e2, l2, f2) in enumerate(mels):
                    if e2 != e and f2==lastConjunct:
                        mels[idx] = e2,l2,e + '#' + ''.join(lastConjunct.split('#')[1:])
    return mels


def keyFunc(elf):
    return int(elf[0][1:])*1000 + (int(elf[2][1:].split('#')[0]) if elf[1] != 0 else 0)

            

ln = 0
for line in opts.treeFile:
    ln += 1
    t = tree.Tree()
    t.read(line)
    ancestorsMap = {}
    t.e = ''
    t.setRefs(0)
    if opts.debug:
        printRefs(t)

    if (opts.psg):
        P = interpret_psg(t)
        if opts.conjunctionHeadShift:
            P = shiftConjunctionHeadToConjWord(P)
    else:
        P = interpret(t)
    
    P = sorted(P, key=keyFunc)
    for p in P:
        print(p[0]+'/'+str(p[1])+'/'+p[2], end=' ')
    print()
