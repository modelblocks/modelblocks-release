import re
import sys

## reads in this format: <word> <preterm sign> <fork decision> <join decision> <apex sign>/<brink sign>

Q = [ ]
for line in sys.stdin:
  m = re.search('^(.*) (.*) (.*) (.*) (.*)',line)
  if m is not None:
    (w, p, f, j, q) = m.groups()
    if f=='1': Q.append('')
    if j=='1': Q = Q[:-1]
    if len(Q)>0: Q[-1] = q
    print( w + ' ' + p + ' ' + f + ' ' + j + ' ' + ';'.join(Q) )

