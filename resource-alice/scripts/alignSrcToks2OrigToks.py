import sys

srcPath, origPath = sys.argv[1:3]

punc = ["-LRB-", "-RRB-", ",", "-", ";", ":", "\'", "\'\'", '\"', "`", "``", ".", "!", "?", "-RRB-", ",", "-", ";", ":", ".", "!", "?"]

def depuncWrd(wrd):
    for p in punc:
        wrd = wrd.replace(p, '')
    return wrd

with open(srcPath, 'rb') as f:
    srcTokMeasures = []
    header = True
    for line in f.readlines():
        if header:
            header = False
            continue
        tmp = line.strip().split()
        srcTokMeasures.append((tmp[0], float(tmp[1])))

srcTokMeasures.append((None, srcTokMeasures[-1][-1]))

with open(origPath, 'rb') as f:
    sents = []
    for line in f.readlines():
        sents.append(line.strip().split())

x = srcTokMeasures.pop(0)
w = ''
last_time = x[1]
sentid = 0
out = []
while sentid < len(sents):
    sent = sents[sentid]
    i = 0
    while i < len(sent):
        wrd = sent[i]
        if wrd not in punc:
            wrd_tmp = depuncWrd(w+wrd).lower()
            if depuncWrd(x[0]).lower() == wrd_tmp:
               out.append((w+wrd,sentid,x[1]))
               w = ''
               last_time = x[1]
               x = srcTokMeasures.pop(0)
            elif depuncWrd(x[0]).lower().startswith(wrd_tmp):
               w += wrd
            else:
               assert False, 'Word mismatch occurred in sentid %d: source = %s; original = %s' %(sentid, depuncWrd(x[0]).lower(), wrd_tmp)
        else:
            out.append((wrd, sentid, last_time))
        i += 1
    sentid += 1

assert x[0] == None

print('word sentid timestamp')
for r in out:
    print(' '.join([str(x) for x in r]))
             

