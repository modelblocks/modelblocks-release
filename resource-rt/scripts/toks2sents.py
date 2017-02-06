import sys, argparse

argparser = argparse.ArgumentParser('''
Outputs sentences from line-separated RT tokens and a target sentence tokenization.
''')
argparser.add_argument('targ_sents_file', action='store', nargs=1, help='Target sentence tokenization file')
argparser.add_argument('-w', '--word', dest='w', action='store', nargs=1, default=['word'], help='Name of field containing word strings')
args, unknown = argparser.parse_known_args()

def split_notok_sents(notokwords, toksents):
    target = notokwords.pop(0)
    for s in toksents:
        out = ''
        cur = ''
        for w in s.split():
            cur += w
            assert len(cur) <= len(target), '%s expected, %s provided.' % (target, cur)
            if cur == target:
                if out == '':
                    out = cur
                else:
                    out += ' ' + cur
                cur = ''
                if len(notokwords) > 0:
                    target = notokwords.pop(0)
        print out

header = sys.stdin.readline().split()
wordix = header.index(args.w[0])

toks = []
for line in sys.stdin:
    toks.append(line.split())

words = []
for i in range(len(toks)):
    words.append(toks[i][wordix])

with open(args.targ_sents_file[0], 'rb') as file:
    sents = file.readlines()

split_notok_sents(words, sents)
