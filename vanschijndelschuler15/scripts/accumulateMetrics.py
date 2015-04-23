#python accumulateMetrics.py
# Sums metrics in regions to output cummetrics

import sys
import re

DEBUG = False
header = []

metrics = {}

newfile = re.compile('subject word')

for line in sys.stdin.readlines():
  sline = line.strip().split()
  if newfile.match(line):
    #header case
    if DEBUG:
      header = sline
    for i,h in enumerate(sline):
      lenh = len(h)
      if lenh <= 3 or ('start' != h[:5] and h[:3] not in ['end','cum','unk'] and h[-3:] not in ['fix','len','pos'] and 'id' != h[-2:] and h not in ['subject','word','parsed']):
        #not a boolean metric
        metrics[i] = ['cum'+h,0]

    sys.stdout.write(line[:-1]+' ')
    for i in sorted(metrics):
      #output cumuheader
      sys.stdout.write(metrics[i][0]+' ')
    sys.stdout.write('\n')
  else:
    #data case
    if float(sline[2]) ==  0:
      #not fixated, so still in a saccade (not the end of a region)
      for i,w in enumerate(sline):
        #if the word wasn't fixated, there's incomplete info
        sys.stdout.write(w+' ')
        hix = i
        
        if DEBUG:
          sys.stderr.write(' :: '+str(i)+'['+ header[i]+']/'+str(hix))
        if hix in metrics.keys():
          #only accumulate floats (bools are preaccumulated by xptimetoks)
          if DEBUG:
            sys.stderr.write('['+metrics[hix][0]+']')
          metrics[hix][1] += float(w)
      if DEBUG:
        sys.stderr.write('\n\n')
      sys.stdout.write('\n')
    else:
      #end of a region
      for i,w in enumerate(sline):
        if i in metrics.keys():
          #only accumulate floats (bools are preaccumulated by xptimetoks)
          metrics[i][1] += float(w)
        sys.stdout.write(str(sline[i])+' ')
      for i in sorted(metrics):
        #output cummetrics and reset them
        sys.stdout.write(str(metrics[i][1])+' ')
        metrics[i][1] = 0
      sys.stdout.write('\n')
