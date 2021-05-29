import sys
import re

'''
Convert linetrees to linetoks by finding all words.  Print !ARTICLE delims without modification
'''

with open(sys.argv[1],'r') as iff:
    lines = iff.readlines()

for line in lines:
    if line=="!ARTICLE\n":
        print(line.strip())
    else:
        words = re.findall(' ([^ \)]+)\)', line) #space, one or more non spaces or close parens, close paren
        print(" ".join(words))
