
import re
import sys

def calcInheritances ( C, G, v ):
  ## memoization...
  if v in C: return C[v]
  ## calculate local predicate context...
  C[v] =  [G[v,'0']+',0'] if (v,'0') in G else []                                        # from predicate itself.
  C[v] += [G.get((u,'0'),'???')+','+l for u,l in G if G[u,l]==v and l>='1' and l<='9']   # from impinging predicates.
  ## conjunction inheritance...
  if (v,'c') in G:    C[v] += calcInheritances( C, G, G[v,'c'] )
  ## restriction inheritance due to predicative noun phrase...
  if (v,'r') in G:    C[v] += calcInheritances( C, G, G[v,'r'] )
  return C[v]


## for each graph...
for line in sys.stdin:

  ## initialize dicts...
  C = { }
  G = { }
  V = { }

  ## construct dependencies...
  for dep in line.split():
    if ',' not in dep:
      sys.stderr.write('WARNING: Improperly formed dependency: '+dep+'\n')
      continue
    src,lbl,dst = dep.split(',')
    G [ src, lbl ] = dst
    if lbl>='1' and lbl<='9' and dst != '-':
      V [ dst[0:2]+'s' ] = 1

  ## print...
  for v in sorted(V):
    print ( v + ' ' + ' '.join(calcInheritances(C,G,v)) )
  print ( )
