import sys, re, argparse
sys.setrecursionlimit(1500)

## Command-line processing
argparser = argparse.ArgumentParser('''
Calculates various center-embedding fields on %.nopunc.tss.fromlinetree.semprocdecpars and outputs tokmeasures using tokenization provided by %.linetoks (stdin).
''')
argparser.add_argument('decpars', nargs=1, help='%.nopunc.tss.fromlinetree.semprocdecpars file to use for embedding calculations.')
argparser.add_argument('-d', '--debug', action='store_true', help='Print verbose debugging log instead of data table.')
args, unknown = argparser.parse_known_args()

## Global variables
i = 0
pronouns = ['I', 'he', 'she', 's/he', 'it', 'we', 'they', 'you', 'me', 'him', 'her', 'us', 'them', \
        'who', 'whom', 'which', 'that', 'myself', 'himself', 'herself', 'itself', 'ourselves', 'themselves', 'oneself']
depth_modes = ['any', 'allnolo', 'cc', 'min']
length_measures = ['len', 'dr', 'drv']

## Regex searches
s = re.compile('----(.*)$')
w = re.compile('W .* ([^ ]+)$')
f = re.compile('F .* ([01])[^ ]*$')
j = re.compile('J .* ([01])[^ ]*$')
p = re.compile('P .* ([^ ]+)$')
frag = re.compile(' *\[[^\]]*\]:([^\/;]+)\/ *\[[^\]]*\]:([^\/;]+)')
#frag = re.compile(' *[^\]]+\]:([^\/]+)\/*[^\]]+\]:([^\/]+)')
r = re.compile('====(.*)$')
maxsplit = re.compile('[/;]')

## Class definitions
class StoreNode(object):
    def __init__(self, t, f, j, lab, edgetype='A', dr=0, drv=0, prev=None, next=None, above=None, below=None):
        self.t = t
        self.f = f
        self.j = j
        self.lab = lab
        self.edgetype = edgetype
        self.dr = dr
        self.drv = drv
        self.prev = prev
        self.next = next
        self.above = above
        self.below = below
        self.measures = ['len', 'dr', 'drv']

        self.d = {}
        for mode in depth_modes: 
            self.d[mode] = self.depth(mode)
    
    def __str__(self):
        out = self.lab if self.lab != None else '<EMPTY>'
        if self.above != None:
            out = str(self.above) + ' <--' + self.edgetype + ' ' + out
        else:
            out = '%s F:%s J:%s DR:%s DRV:%s ' %(self.t, self.f, self.j, self.dr, self.drv) + out
            if self.prev != None:
                out = str(self.prev.bottom()) + '\n' + out
        return out

    def retrievalLen(self, mode, measure, definition):
        n = self
        noF = n.f == 0
        endembd = n.endembd(mode) == 1
        length = 0
        
        if definition == 'noF':
            if not noF:
                return 0
        elif definition == 'embd':
            if not endembd:
                return 0
        elif definition == 'reinst':
            if n.f == 1 and n.endembd(mode) == 0:
                return 0
        else:
            raise ValueError('Length definition "%s" not supported.' %definition)

        if measure == 'len':
            length += 1
        elif measure == 'dr':
            length += n.dr
        elif measure == 'drv':
            length += n.drv
        else:
            raise ValueError('Length measure type "%s" not supported' %measure)
        
        d = n.d[mode]
        prev = n.top().prev
        if prev == None:
            return length

        n = prev
        if endembd:
            d += 1
        while n.d[mode] < d and n.below != None:
            n = n.below
        while n.d[mode] > d and n.above != None:
            n = n.above
        prev = n.top().prev
        
        stop = n.d[mode] != d or n == None
        if not stop and (definition == 'noF' or (definition=='reinst' and not endembd)):
            stop = n.f==0 and not n.endembd(mode)==1
        while not stop:
            if measure == 'len':
                length += 1
            elif measure == 'dr':
                length += n.dr
            elif measure == 'drv':
                length += n.drv
            n = prev
            if n != None:
                prev = n.top().prev
                while n.d[mode] < d and n.below != None:
                    n = n.below
                while n.d[mode] > d and n.above != None:
                    n = n.above
            stop = n == None or n.d[mode] != d
            if not stop and (definition == 'noF' or (definition=='reinst' and not endembd)):
                stop = n.f==0 and not n.endembd(mode)==1
        return length
            

    def depth(self, mode='any'):
        d = 0
        n = self
        if self.below != None:
            belowedgetype = self.below.edgetype
        else:
            belowedgetype = 'B'
        while n != None:
            if mode == 'any':
                d += 1
            elif mode == 'allnolo':
                if n.above != None and \
                     ((n.edgetype == 'A' and belowedgetype == 'B') \
                   or (n.edgetype == 'A' and belowedgetype == 'A') \
                   or (n.edgetype == 'B' and belowedgetype == 'B')):
                    d += 1
            elif mode == 'cc':
                if n.edgetype == 'A' and n.above != None:
                    d += 1
            elif mode == 'Bonly':
                if n.edgetype == 'B' and n.above != None:
                    d += 1
            elif mode == 'min':
                if n.edgetype == 'A' and belowedgetype == 'B':
                    d += 1
            else:
                raise ValueError('Mode "%s" for depth calculation is not supported' %mode)
            belowedgetype = n.edgetype
            n = n.above
        return d

    def get_ab(self):
        ad = None
        bd = None
        n = self
        if self.below != None:
            belowedgetype = self.below.edgetype
        else:
            belowedgetype = 'B'
        while n != None and (ad == None or bd == None):
            if belowedgetype == 'A' and n.edgetype == 'B' or n.above == None:
                ad = n.lab
            elif belowedgetype == 'B' and n.edgetype == 'A' or n.above == None:
                bd = n.lab
            belowedgetype = n.edgetype
            n = n.above
        adm1 = None
        bdm1 = None
        while n != None and (adm1 == None or bdm1 == None):
            if belowedgetype == 'A' and n.edgetype == 'B' or n.above == None:
                adm1 = n.lab
            elif belowedgetype == 'B' and n.edgetype == 'A':
                bdm1 = n.lab
            belowedgetype = n.edgetype
            n = n.above
         
        return ad if ad else 'null', \
               bd if bd else 'null', \
               adm1 if adm1 else 'null', \
               bdm1 if bdm1 else 'null'
                
    def top(self):
        n = self
        while n.above != None:
            n = n.above
        return n

    def bottom(self):
        n = self
        while n.below != None:
            n = n.below
        return n

    def endembd(self, mode):
        dt = self.d[mode]
        if self.prev != None:
            dtm1 = self.prev.bottom().d[mode]
            out = int(dt < dtm1)
        else:
            out = 0
        return out

    def startembd(self, mode):
        dt = self.d[mode]
        if self.prev != None:
            dtm1 = self.prev.bottom().d[mode]
            out = int(dt > dtm1)
        else:
            out = 1
        return out


## Method definitions
def next_tok(buffer):
    line = decpars.readline()
    while line and not line.startswith('N '):
        line = decpars.readline()
    return(line)

def process_tok(buffer, line, i, prev, debug=True):
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
    DR = dr(pos, word)
    DRV = drv(pos, word)
    while line and not line.startswith('J '):
       line = buffer.readline()
    assert line, 'Badly-formed input'
    J = int(j.match(line.strip()).group(1))
    while line and not (line.startswith('----') or line.startswith('TREE ')):
       line = buffer.readline()
    if line and line.startswith('----'):
        storestr = s.match(line.strip()).group(1)
        node = parse_store(storestr, i, F, J, prev, DR, DRV)
        i += 1
        prev = node.top()
    else:
        node = StoreNode(i, F, J, None, prev=prev, dr=DR, drv=DRV)
        i = 0
        prev = None
    
    preds = [] 
    preds += [word, DR, DRV, 1-F, J]

    embddepthAny, embddepthAllnolo, embddepthCC, embddepthMin = [node.d[mode] for mode in depth_modes]
    preds += [embddepthAny, embddepthAllnolo, embddepthCC, embddepthMin]
    
    endembdAny, endembdAllnolo, endembdCC, endembdMin = [node.endembd(mode) for mode in depth_modes]
    preds += [endembdAny, endembdAllnolo, endembdCC, endembdMin]
    
    startembdAny, startenmbdAllnolo, startembdCC, startembdMin = [node.startembd(mode) for mode in depth_modes]
    preds += [startembdAny, startenmbdAllnolo, startembdCC, startembdMin]
    
    noFlen, noFdr, noFdrv = [node.retrievalLen('min', measure, 'noF') for measure in length_measures]
    preds += [noFlen, noFdr, noFdrv]
    
    embdlen, embddr, embddrv = [node.retrievalLen('min', measure, 'embd') for measure in length_measures]
    preds += [embdlen, embddr, embddrv]
    
    reinstlen, reinstdr, reinstdrv = [node.retrievalLen('min', measure, 'reinst') for measure in length_measures]
    preds += [reinstlen, reinstdr, reinstdrv]

    Ad, Bd, Adm1, Bdm1 = node.get_ab()
    AdPrim = get_primitive_cat(Ad)
    BdPrim = get_primitive_cat(Bd)
    Bdm1Prim = get_primitive_cat(Bdm1)
    Adm1Prim = get_primitive_cat(Adm1)
    preds += [Ad, AdPrim, Bd, BdPrim, Bdm1, Bdm1Prim]
    
    if debug:
        print('-'*50)
        print(word)
        print(node)
        print('embddepthAny embddepthAllnolo embddepthCC embddepthMin')
        print('%s %s %s %s' %(embddepthAny, embddepthAllnolo, embddepthCC, embddepthMin))
        print('endembdAny endembdAllnolo endembdCC endembdMin')
        print('%s %s %s %s' %(endembdAny, endembdAllnolo, endembdCC, endembdMin))
        print('startembdAny startenmbdAllnolo startembdCC startembdMin')
        print('%s %s %s %s' %(startembdAny, startenmbdAllnolo, startembdCC, startembdMin))
        print('noFlen noFdr noFdrv')
        print('%s %s %s' %(noFlen, noFdr, noFdrv))
        print('embdlen embddr embddrv') 
        print('%s %s %s' %(embdlen, embddr, embddrv))
        print('reinstlen reinstdr reinstdrv')
        print('%s %s %s' %(reinstlen, reinstdr, reinstdrv))
        print('Ad AdPrim Bd BdPrim Adm1 Adm1Prim Bdm1 Bdm1Prim')
        print('%s %s %s %s %s %s %s %s' %(Ad, AdPrim, Bd, BdPrim, Adm1, Adm1Prim, Bdm1, Bdm1Prim))
        print('-'*50)
            
    return i, prev, preds 

def parse_store(storestr, t, f, j, prev=None, DR=0, DRV=0):
    cur = ''
    edgetype = 'A'
    lab = None
    above = None
    node = None
    while storestr != '':
        char = storestr[0]
        storestr = storestr[1:]
        if char == '/' or char == ';':
            lab = cur.split(':')[-1]
            node = StoreNode(t, f, j, lab, edgetype=edgetype, dr=DR, drv=DRV, above=above, prev=prev)
            if above != None:
                above.below = node
            if prev != None:
                prev = prev.below
            above = node
            cur = ''
            if char == ';':
                edgetype = 'B'
            else:
                edgetype = 'A'
        else:
            cur += char

    return node
    
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

def get_primitive_cat(s):
    if s == 'N-b{N-aD}':
        return 'D'
    if s == 'null' or s == '':
        return 'null'
    if s[0] in ['B', 'G', 'L']:
        return 'Vnon'
    if s[0] in ['N', 'V', 'A', 'R', 'D', 'X', 'S']:
        return s[0]
    return 'O'




## Main program

header = []
punc_tok = []
header += 'word', 'dr', 'drv'
punc_tok += ['0'] * 2
header += 'noF', 'yesJ'
punc_tok += ['0'] * 2
header += 'embddepthAny', 'embddepthAllnolo', 'embddepthCC', 'embddepthMin'
punc_tok += ['0'] * 4
header += 'endembdAny', 'endembdAllnolo', 'endembdCC', 'endembdMin'
punc_tok += ['0'] * 4
header += 'startembdAny', 'startenmbdAllnolo', 'startembdCC', 'startembdMin'
punc_tok += ['0'] * 4
header += 'noFlen', 'noFdr', 'noFdrv'
punc_tok += ['0'] * 3
header += 'embdlen', 'embddr', 'embddrv'
punc_tok += ['0'] * 3
header += 'reinstlen', 'reinstdr', 'reinstdrv'
punc_tok += ['0'] * 3
header += 'Ad', 'AdPrim', 'Bd', 'BdPri', 'Bdm1', 'Bdm1Prim'
punc_tok += ['null'] * 6
print(' '.join(header))
    

with open(args.decpars[0], 'rb') as decpars:
    dpars_line = next_tok(decpars)
    ltoks_line = sys.stdin.readline()
    prev = None
    i, prev, preds = process_tok(decpars, dpars_line, i, prev, debug=args.debug)
    while ltoks_line and dpars_line:
        for wrd in ltoks_line.strip().split():
            if preds is not None and wrd == preds[0]:
                print(' '.join([str(P) for P in preds]))
                if dpars_line:
                    i, prev, preds = process_tok(decpars, dpars_line, i, prev, debug=args.debug)
                    dpars_line = next_tok(decpars)
                else:
                    preds = None
            else: ## punctuation token
                print(' '.join([wrd] + punc_tok))
        ltoks_line = sys.stdin.readline()
    if preds is not None:
        print(' '.join([str(P) for P in preds]))


