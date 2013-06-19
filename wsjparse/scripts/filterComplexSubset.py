#python3 (-v) filterComplexSubset.py subsetFile dataFile
# Filters out all lines of complexity data that have a positive filter flag as well as any sentences not in the subsetFile

import sys

#########################
#
# Filter
#
#########################
sys.stderr.write('Filtering output\n')

INVERSE = False

if sys.argv[1][0] == '-':
  if 'v' in sys.argv[1]:
    INVERSE = True
  subF = sys.argv[2]
  datF = sys.argv[3]
else:
  subF = sys.argv[1]
  datF = sys.argv[2]

subset = []

with open(subF,'r') as subsetFile:
  for line in subsetFile:
    if line.strip() != '':
      subset.append(int(line.strip()))

file = open(datF,'r')
try:
  for line in file.readlines():
    pass
  file.close()
  file = open(datF,'r')
except:
  file = open(datF,'r',encoding='latin-1')

ABORT = False
FIRST = True
keys = {}

sentix = 1
sentpos = -1
subj = 'sa'
for line in file.readlines():
  sline = line.split()
  if FIRST:
    for ix in range(len(sline)):
      keys[sline[ix]] = ix
  else:
    if sline[keys['subject']] != subj:
      sentix = 1
      subj = sline[keys['subject']]
    elif int(sline[keys['sentpos']]) < sentpos:
      sentix += 1
    sentpos = int(sline[keys['sentpos']])
  if FIRST or (int(sline[keys['filter']]) == 0 and int(sline[keys['notfix']]) == 0):#if line isn't filtered, output it
    if FIRST or (not INVERSE and sentix in subset) or (INVERSE and sentix not in subset): #only output sentences from desired subset
      for i in range(len(sline)):
        if i not in (keys['filter'],keys['notfix']): #do not output filter flags
          sys.stdout.write(sline[i]+' ')
      sys.stdout.write('\n')
  FIRST = False

file.close()
