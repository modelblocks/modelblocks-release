import sys, re, argparse
argparser = argparse.ArgumentParser('''
Calculates various center-embedding fields on %.nopunc.tss.fromlinetree.semprocdecpars and outputs tokmeasures using tokenization provided by %.linetoks (stdin).
''')
argparser.add_argument('decpars', nargs=1, help='%.nopunc.tss.fromlinetree.semprocdecpars file to use for embedding calculations.')
args, unknown = argparser.parse_known_args()

pronouns = ['I', 'he', 'she', 's/he', 'it', 'we', 'they', 'you', 'me', 'him', 'her', 'us', 'them', \
        'who', 'whom', 'which', 'that', 'myself', 'himself', 'herself', 'itself', 'ourselves', 'themselves', 'oneself']

rows = []
disc_ref = []
disc_ref_v = []
store_start_ix = []
noF_start_ix = [0]
yesJ_start_ix = [0]
nofork = False
yesjoin = False
i = 0
s = re.compile('----(.*)$')
w = re.compile('W .* ([^ ]+)$')
f = re.compile('F .* : f([01])')
j = re.compile('J .* : j([01])')
p = re.compile('P .* ([^ ]+)$')
frag = re.compile(' *[^\]]+\]:([^\/]+)\/*[^\]]+\]:([^\/]+)')
r = re.compile('====(.*)$')

print('word embddepth dc dcv ' \
    + 'endembd embdlen embdlendr embdlendrv ' \
    + 'noF noFlen noFlendr noFlendrv ' \
    + 'yesJ ' \
    + 'reinst reinstlen reinstlendr reinstlendrv ' \
    + 'Ad AdPrim Bd BdPrim Bdm1 Bdm1Prim')

def dr(pos, word):
    if pos.startswith('V') or ((pos.startswith('N') and not pos.startswith('N-b{N-aD}')) and not word in pronouns):
        return 1
    else:
        return 0

def drv(pos, word):
    if pos.startswith('V'):
        return 2
    elif pos[0] in ['V','G','B','L'] or ((pos.startswith('N') and not pos.startswith('N-b{N-aD}')) and not word in pronouns):
        return 1
    else:
        return 0

def next_tok(buffer):
    line = decpars.readline()
    while line and not line.startswith('----'):
        line = decpars.readline()
    return(line)

def get_primitive_cat(s):
    if s == 'N-b{N-aD}':
        return 'D'
    if s[0] in ['B', 'G', 'L']:
        return 'Vnon'
    if s[0] in ['N', 'V', 'A', 'R', 'D', 'X', 'S']:
        return s[0]
    if s == 'null':
        return s
    return 'O'    
 
def process_tok(buffer, line, i):
    global nofork
    global yesjoin

    storestr = s.match(line.strip()).group(1)
    while line and not line.startswith('F '):
       line = buffer.readline()
    assert line, 'Badly-formed input'
    F = int(f.match(line.strip()).group(1))
    while line and not line.startswith('P '):
        line = buffer.readline()
    assert line, 'Badly-formed input'
    pos = p.match(line.strip()).group(1)
    while line and not line.startswith('W '):
        line = buffer.readline()
    assert line, 'Badly-formed input'
    word = w.match(line.strip()).group(1)
    while line and not line.startswith('J '):
       line = buffer.readline()
    assert line, 'Badly-formed input'
    J = int(j.match(line.strip()).group(1))

    disc_ref.append(dr(pos, word))
    disc_ref_v.append(drv(pos, word))

    store = [x for x in storestr.strip().split(';') if x != '']
    if len(store) > 0:
      Ad = frag.match(store[-1]).group(1)
      Bd = frag.match(store[-1]).group(2)
      if len(store) > 1:
        Bdm1 = frag.match(store[-2]).group(2)
      else:
        Bdm1 = 'null'
    else:
      Ad = 'null'
      Bd = 'null'
      Bdm1 = 'null'
    depth = len(store)
    endembd = 0
    embdlen = 0
    embdlendr = 0
    embdlendrv = 0
    noFstart = None
    noFlen = 0
    noFlendr = 0
    noFlendrv = 0
    yesJstart = None
    yesJlen = 0
    yesJlendr = 0
    yesJlendrv = 0
    reinstlen = 0
    reinstlendr = 0
    reinstlendrv = 0
    while len(store_start_ix) > depth + 1:
        store_start_ix.pop()
        noF_start_ix.pop()
    while depth > len(store_start_ix):
        store_start_ix.append(i)
        noF_start_ix.append(i)
    if depth < len(store_start_ix):
        start = store_start_ix.pop()
        noFstart = noF_start_ix.pop()
        embdlen = i - start
        embdlendr = sum(disc_ref[start:i])
        embdlendrv = sum(disc_ref_v[start:i])
        endembd = 1
    noF = int(nofork)
    if noF == 1:
        if noFstart == None:
            noFstart = noF_start_ix[-1]
        noFlen = i - noFstart
        noFlendr = sum(disc_ref[noFstart:i])
        noFlendrv = sum(disc_ref_v[noFstart:i])
        noF_start_ix[-1] = i
    yesJ = int(yesjoin)
    #if yesJ == 1:
    #    if yesJstart == None:
    #        yesJstart = yesJ_start_ix[-1]
    #    yesJlen = i - yesJstart
    #    yesJlendr = sum(disc_ref[yesJstart:i])
    #    yesJlendrv = sum(disc_ref_v[yesJstart:i])
    #    yesJ_start_ix[-1] = i
    reinst = int(noF == 1 or endembd == 1)
    if reinst == 1:
        reinstlen = max(noFlen, embdlen)
        reinstlendr = max(noFlendr, embdlendr)
        reinstlendrv = max(noFlendrv, embdlendrv)
    nofork = F != 1
    yesjoin = J == 1
#    print(' '.join([str(x) for x in [word, endembd, noF, store_start_ix, noF_start_ix, yesJ]]))
    return(word, depth, \
           endembd, embdlen, embdlendr, embdlendrv, \
           noF, noFlen, noFlendr, noFlendrv, \
           yesJ, \
           reinst, reinstlen, reinstlendr, reinstlendrv, \
           Ad, Bd, Bdm1, \
           i+1)

with open(args.decpars[0], 'rb') as decpars:
    line = next_tok(decpars)
    while line:
        word, depth, \
              endembd, embdlen, embdlendr, embdlendrv, \
              noF, noFlen, noFlendr, noFlendrv, \
              yesJ, \
              reinst, reinstlen, reinstlendr, reinstlendrv, \
              Ad, Bd, Bdm1, \
              i = process_tok(decpars, line, i)
        AdPrim = get_primitive_cat(Ad)
	BdPrim = get_primitive_cat(Bd)
	Bdm1Prim = get_primitive_cat(Bdm1)
        if len(rows) > 0:
            rows[-1] += [endembd, embdlen, embdlendr, embdlendrv, \
                         noF, noFlen, noFlendr, noFlendrv, \
                         yesJ, \
                         reinst, reinstlen, reinstlendr, reinstlendrv, \
                         Ad, AdPrim, Bd, BdPrim, Bdm1, Bdm1Prim]
        rows.append([word, depth, disc_ref[i-1], disc_ref_v[i-1]])
        line = next_tok(decpars)
 
rows[-1] += [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'null', 'null', 'null', 'null', 'null', 'null']
depth = 0

for line in sys.stdin:
    for word in line.split():
        if len(rows) > 0 and word.replace('[','!').replace(']','!') == rows[0][0]:
            # Matches a non-punc word in the decpars input
            print(' '.join(str(x) for x in rows[0]))
            row = rows.pop(0)
            depth = row[1]
            Ad = row[-6]
            AdPrim = row[-5]
            Bd = row[-4]
            BdPrim = row[-3]
            Bdm1 = row[-2]
            Bdm1Prim = row[-1]
        else:
            # Is a punctuation token
            print(str(word) + ' ' + str(depth) + ' 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ' + Ad + ' ' + AdPrim + ' ' + Bd + ' ' + BdPrim + ' ' + Bdm1 + ' ' + Bdm1Prim)
assert len(rows) == 0, 'The *.semprocdecpars file was not fully consumed during printing. Check to make sure that there are no mismatches between wordforms in the *.linetoks file and the *.semprocdecpars file.'
