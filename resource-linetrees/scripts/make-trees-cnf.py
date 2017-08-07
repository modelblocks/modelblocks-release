import os, sys, re, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
import tree

argparser = argparse.ArgumentParser('''
Takes a linetrees file and binarizes it to CNF.
''')
args, unkown = argparser.parse_known_args()

heads = {}
deps = {}

def preterm(t):
    return len(t.ch) == 1 and t.ch[0].ch == []

def binarize(t):
  children = t.ch[:]
  if len(children) > 2:
    t2 = tree.Tree()
    t2.c = t.c
    t2.ch = children[1:]
    t2.l = 0
    t2.r = 1
    t2.p = t
    t.ch = [binarize(children[0])] + [binarize(t2)]
    t.l = 0
    t.r = 1
  elif len(children) == 2:
    t.ch = [binarize(children[0])] + [binarize(children[1])]
  elif len(children) == 1 and not preterm(t):
    return(binarize(t.ch[0]))
  return(t)

t = tree.Tree()

for line in sys.stdin:
    heads = {}
    deps = {}
    t.read(line)
    print(binarize(t))

 
