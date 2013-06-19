#python3 filterComplex.py dataFile
# Filters out all lines of complexity data that have a positive filter flag

import sys

#########################
#
# Filter
#
#########################
sys.stderr.write('Filtering output\n')

file = open(sys.argv[1],'r')#,encoding='latin-1')
try:
  for line in file.readlines():
    pass
  file.close()
  file = open(sys.argv[1],'r')
except:
  file = open(sys.argv[1],'r',encoding='latin-1')

ABORT = False
FIRST = True
ix = 0
for line in file.readlines():
  sline = line.split()
  if FIRST:
    while ix < len(sline):
      if sline[ix] == 'filter':
        break
      else:
        ix += 1
    if ix == len(sline):
      #no filter column in datafile; abort!
      ABORT = True
  if ABORT:
    sys.stdout.write(line)
    FIRST = False
    continue
  if FIRST or (int(sline[ix]) == 0 and int(sline[ix+1]) == 0):#if line isn't filtered, output it
    for i in range(0,len(sline)):
      if i != ix and i != ix+1: #do not output filter flags
        sys.stdout.write(sline[i]+' ')
    sys.stdout.write('\n')
  FIRST = False

file.close()
