import sys
from itertools import chain, combinations

def powerset(iterable):
    xs = list(iterable)
    return chain.from_iterable(combinations(xs,n) for n in range(len(xs)+1))

preds = sys.argv[1:]
if len(preds) == 1:
    preds = preds.split('_')
pset = powerset(preds)

out = []

for s in pset:
    out_cur = []
    for p in preds:
        if p not in s:
            out_cur.append('~%s' %p)
        else:
            out_cur.append('%s' %p)
    out_cur = '_'.join(out_cur)
    out.append(out_cur)

out = ' '.join(out)

print(out)

