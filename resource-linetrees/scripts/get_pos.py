import sys, argparse
sys.path.append('../resource-gcg/scripts/')
import tree

argparser = argparse.ArgumentParser('''
Prints a CoNLL style table of word and PoS tags from linetrees input.
''')
argparser.add_argument('-i', '--noindex', dest='i', action='store_true', help='Remove trace indices from PoS tags in PTB-style trees')
args, unknown = argparser.parse_known_args()

t = tree.Tree()
ix = 1
pos_count = 0
pos_map = {}

def preterm(t):
    return len(t.ch) == 1 and len(t.ch[0].ch) == 0

def num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def print_pos(t):
    global ix, pos_count
    if args.i:
        tlist = t.c.split('-')
        if num(tlist[-1]):
            t.c = '-'.join(tlist[:-1])
    if preterm(t):
        if t.c in pos_map:
            tag = pos_map[t.c]
        else:
            pos_count += 1
            pos_map[t.c] = pos_count
            tag = pos_count
        print(str(ix) + '\t' + t.ch[0].c + '\t' + t.ch[0].c + '\t' + str(tag) + '\t' + str(tag) + '\t_\t_\t_\t_')
        ix += 1
    for c in t.ch:
        print_pos(c)

for line in sys.stdin:
    if line.strip() != '':
        ix = 1
        t.read(line)
        print_pos(t)
        print('')
        
