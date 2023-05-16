import sys
import pandas as pd
import argparse

argparser = argparse.ArgumentParser('''
Add Natural Stories document ID's to item-level tokenization
''')
argparser.add_argument('src', help='Path to source tokenization table <NSDIR>/naturalstories_RTS/all_stories.tok')
args = argparser.parse_args()

docnames = [None, 'Boar', 'Aqua', 'MatchstickSeller', 'KingOfBirds', 'Elvis', 'MrSticky', 'HighSchool', 'Roswell', 'Tulips', 'Tourettes'] 
name2discid = {docnames[i+1]: i for i in range(10)}

docix_seq = pd.read_csv(args.src, sep='\t').item.values

i = 0
si = 0

word = []
sentid = []
sentpos = []
docid = []

for line in sys.stdin:
    sp = 1
    words = line.strip().split()
    for w in words:
        word.append(w)
        sentid.append(si)
        sentpos.append(sp)
        docid.append(docnames[docix_seq[i]])
        sp += 1
        i += 1
    si += 1

out = pd.DataFrame({'word': word, 'sentid': sentid, 'sentpos': sentpos, 'docid': docid})
out['discid'] = out.docid.map(name2discid)
out['discpos'] = out.groupby('discid').cumcount() + 1

out.to_csv(sys.stdout, index=False, sep=' ')
