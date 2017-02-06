import sys
import collections
sys.path.append('../resource-gcg/scripts')
import tree

def makedepterm ( t, DepDict, w=0 ):
  for st in t.ch:
    w = makedepterm ( st, DepDict, w )
  if len(t.ch) == 1 and len(t.ch[0].ch) == 0:
    w += 1
    t.ch[0].c = '_'.join(DepDict[w])
    if t.ch[0].c == '': t.ch[0].c = '-'
  return w

for depline in open(sys.argv[1]):
  DepDict = collections.defaultdict(list)
  DepList = depline.split()
  for dep in DepList:
    src = dep.partition(',')[0]
    DepDict[int(src[0:2])].append ( dep )

  treeline = sys.stdin.readline()
  t = tree.Tree()
  t.read ( treeline )
  makedepterm ( t, DepDict )
  print t
