import os, sys, re, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
import model, tree

argparser = argparse.ArgumentParser('''
Takes a linetrees file and headedness model and produces most probable dependency graphs.
''')
argparser.add_argument('model', nargs=1, help='Probability model (head child given category).')
argparser.add_argument('-d', '--debug', dest='debug', action='store_true', help='Debug mode: Print tree before dependency graph.')
args, unkown = argparser.parse_known_args()

heads = {}
deps = {}

def preterms(t):
    if preterm(t):
        return [t]
    x = []
    for ch in t.ch:
        x += preterms(ch)
    return x

def preterm(t):
    return len(t.ch) == 1 and t.ch[0].ch == []

def get_deps(t, ix, words):
    if preterm(t):
        deps[t] = ix
        return(words.index(t) + 1)
    heads[t] = max(t.ch, key = lambda x: head_model[t.c][x])
    if args.debug:
        heads[t].c = 'HEAD:' + heads[t].c + '->' + str(ix)
    children = t.ch[:]
    head = children.pop(children.index(heads[t]))
    headix = get_deps(head, ix, words)
    for ch in children:
        get_deps(ch, headix, words)
        if args.debug:
            ch.c += '->' + str(headix)
    return(headix)

with open(args.model[0], 'r') as m:
    head_model = model.CondModel('R')
    for line in m:
        head_model.read(line)

t = tree.Tree()

for line in sys.stdin:
    heads = {}
    deps = {}
    t.read(line)
    preterminals = preterms(t)
    get_deps(t, 0, preterminals)
    preterminals.insert(0, tree.Tree('X', [tree.Tree('ROOT', [])]))
    if args.debug:
        print(t)
    for i in range(1, len(preterminals)):
        print('X(' + preterminals[deps[preterminals[i]]].ch[0].c + '-' + str(deps[preterminals[i]]) + ', ' + str(preterminals[i].ch[0].c) + '-' + str(i) + ')')
    print('')

 
