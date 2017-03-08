#python querydataframe.py --input FILE [--i colheader ...]
# each --i argument specifies a column header whose column will be output to stdout
# outputs columns in order specified by input file

import sys

targetcols = []
OPTS = {}
for aix in range(1,len(sys.argv)):
  if len(sys.argv[aix]) < 2 or sys.argv[aix][:2] != '--':
    #filename or malformed arg
    continue
  elif aix < len(sys.argv) - 1 and len(sys.argv[aix+1]) > 2 and sys.argv[aix+1][:2] == '--':
    #missing filename, so simple arg
    OPTS[sys.argv[aix][2:]] = True
  else:
    if sys.argv[aix][2:] == 'i':
      targetcols.append(sys.argv[aix+1])
    else:
      OPTS[sys.argv[aix][2:]] = sys.argv[aix+1]

headerix = []
if 'input' in OPTS:
  with open(OPTS['input'],'r') as f:
    data = [l.strip().split() for l in f.readlines()]
else:
  data = [l.strip().split() for l in sys.stdin.readlines()]

for hix in range(len(data[0])):
  if data[0][hix] in targetcols:
    headerix.append(hix)

output = []
for line in data:
  outputline = []
  for ix in headerix:
    outputline.append(line[ix])
  output.append(' '.join(outputline))

sys.stdout.write('\n'.join(output)+'\n')
