import sys, re

spill = re.compile('(.+)S[0-9]+$')

cli = sys.argv[1].strip().split('_')

preds = []
restrdomain = ''
for i in xrange(len(cli)):
    s = cli[i]
    if s in ['-a', '-A', '-x']:
        P = cli[i+1].split('+')
        for p in P:
            if p not in preds:
                preds.append(p)
    elif s == '-R':
        restrdomain = 'scripts/' + cli[i+1] + '.restrdomain.txt'

if restrdomain != '':
    with open(restrdomain, 'rb') as f:
        for line in f.readlines():
            if line.strip() != '' and not line.startswith('#'):
                col = line.strip().split()[1]
                if not col in preds:
                    preds.append(col)

for i, pred in enumerate(preds):
    if pred.startswith('fut'):
        pred = pred[3:]
    if pred.startswith('cum'):
        pred = pred[3:]
    if spill.match(pred):
        pred = spill.match(pred).group(1)
    preds[i] = pred

print('+'.join(preds))


