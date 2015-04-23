#python tagPunctuation.py
# Adds fields detailing what punctuation is observed in stdin

import sys

HEADER = True
wordix = 0
for line in sys.stdin.readlines():
  if HEADER:
    sys.stdout.write(line[:-1]+' period comma colon semicolon apostrophe nonmorphapos quote exclamation question lpar rpar\n')
    for i,w in enumerate(line.split()):
      if w == 'word':
        wordix = i
        break
    HEADER = False
  else:
    PERIOD = '0'
    COMMA = '0'
    COLON = '0'
    SEMICOLON = '0'
    APOSTROPHE = '0'
    QUOTEAPOS = '0'
    QUOTE = '0'
    EXCLAMATION = '0'
    QUESTION = '0'
    LPAR = '0'
    RPAR = '0'
    sline = line.strip().split()
    if '.' in sline[wordix]:
      PERIOD = '1'
    if ',' in sline[wordix]:
      COMMA = '1'
    if ':' in sline[wordix]:
      COLON = '1'
    if ';' in sline[wordix]:
      SEMICOLON = '1'
    if "'" in sline[wordix]:
      APOSTROPHE = '1'
    if "'" in (sline[wordix][0],sline[wordix][-1]): #this could miss if ' isn't the final char: 'blue'.
      QUOTEAPOS = '1'
    if '"' in sline[wordix]:
      QUOTE = '1'
    if '!' in sline[wordix]:
      EXCLAMATION = '1'
    if '?' in sline[wordix]:
      QUESTION = '1'
    if '(' in sline[wordix]:
      LPAR = '1'
    if ')' in sline[wordix]:
      RPAR = '1'
    sys.stdout.write(line[:-1]+' '+PERIOD+' '+COMMA+' '+COLON+' '+SEMICOLON+' '+APOSTROPHE+' '+QUOTEAPOS+' '+QUOTE+' '+EXCLAMATION+' '+QUESTION+' '+LPAR+' '+RPAR+'\n')
