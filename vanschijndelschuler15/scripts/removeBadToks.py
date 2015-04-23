#python removeBadtoks.py
# Removes unparsed or unfixated regions from stdin
# Also realigns the various subject files to ensure consistent column placement

import sys
import re

# For any sentences with failed parses, this flag forces omission of the entire sentence
# If False, this flag allows complexity metrics up to the point of failure
OMIT_FAILED= False

# This flag forces omission of any sentences that contain unknown words
OMIT_UNKS= False

newfile = re.compile('subject word')

DEBUG = False

HEADER = True
fdurix = 0
parsedix = 0
unkix = 0
sentidx = 0
goldsentidx = 0
goldheader = []
output = []
failedsents = []
colorder = {}
for line in sys.stdin.readlines():
  if newfile.match(line):
    sline = line.strip().split()
    for i,w in enumerate(sline):
      if w == 'fdur':
        fdurix = i
      elif w == 'parsed':
        #figure out where the parsed key is
        parsedix = i
      elif w == 'unknown':
        unkix = i
      elif w == 'sentid':
        sentidx = i
        if HEADER:
          goldsentidx = i
    if HEADER:
      goldheader = sline    
      sys.stdout.write(line)
      HEADER = False
    #need to create new alignment ordering
    colorder = dict([(c,i) for i,c in enumerate(sline)])
    if DEBUG:
      sys.stderr.write('parsedix = '+str(parsedix)+'\n')
  else:
    sline = line.strip().split()
    if sline == []:
      #empty line
      continue
    if DEBUG:
      sys.stderr.write(str(sline[parsedix])+ ' : '+str(sline)+'\n')
    if float(sline[fdurix]) == 0: #unfixated region
      continue
    elif bool(int(sline[parsedix])) == False: #unparsed region (parse failure)
      if OMIT_FAILED:
        failedsents.append(sline[sentidx])
      continue
    elif OMIT_UNKS and bool(int(sline[unkix])) == True: #region contains unknown words
      continue
    else:
      outputline = []
      for col in goldheader:
        #output columns according to first header order
        outputline.append(sline[colorder[col]])
      output.append(' '.join(outputline)+'\n')

for line in output:
  if line.split()[goldsentidx] not in failedsents:
    sys.stdout.write(line)
