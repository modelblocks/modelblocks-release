# python preprocess_gigaword.py Gigaword_File
# strips xml from gigaword and renders it in plaintext, line-delimited sents
import sys

with open(sys.argv[1], 'r') as f:
    lines = f.readlines()

corpus = []
sent = []
INTEXT = False
for line in lines:
    sline = line.strip()
    if sline == '':
        continue
    elif sline == "<TEXT>":
        #begun a story, so prep for content
        INTEXT = True
    elif sline == "</TEXT>":
        #completed a story, so prep for next metadata
        INTEXT = False
    elif INTEXT:
        if sline == "</P>":
            #completed a sentence
            corpus.append(' '.join(sent))
            sent = []
        elif sline[0] == '<':
            #tag line
            continue
        else:
            #content line
            sent.append(sline)
sys.stdout.write('\n'.join(corpus)+'\n')
