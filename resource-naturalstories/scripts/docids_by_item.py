import sys
import pandas as pd
import argparse

argparser = argparse.ArgumentParser('''
Add Natural Stories document ID's to item-level tokenization
''')
argparser.add_argument('src', help='Path to source tokenization table <NSDIR>/naturalstories_RTS/all_stories.tok')
args = argparser.parse_args()

docnames = [None, 'Boar', 'Aqua', 'MatchstickSeller', 'KingOfBirds', 'Elvis', 'MrSticky', 'HighSchool', 'Roswell', 'Tulips', 'Tourettes'] 

docix_seq = pd.read_csv(args.src, sep='\t').item.values

sentid = 0
i = 0

print('word sentid sentpos docid')

for line in sys.stdin:
    sentpos = 1
    words = line.strip().split()
    for word in words:
        print('%s %s %s %s' %(word, sentid, sentpos, docnames[docix_seq[i]]))
        sentpos += 1
        i += 1

    sentid += 1
