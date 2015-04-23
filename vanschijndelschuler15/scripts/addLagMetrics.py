#python addLagMetrics.py
# Adds lag metrics... obviously
# adds metrics from previous line to current line as lag metrics (should probably be done after filtration to account for spillover effects)

import sys
import re

metrics = {}
nextmetrics = {}

newfile = re.compile('subject word')

for line in sys.stdin.readlines():
  sline = line.strip().split()
  if newfile.match(line):
    #header case
    for i,h in enumerate(sline):
      if h not in ['subject','word','parsed']:
        #find all viable lag metrics
        metrics[i] = ['lag'+h,'0']

    sys.stdout.write(line[:-1]+' ')
    for i in sorted(metrics):
      #output lagheader
      sys.stdout.write(metrics[i][0]+' ')
    sys.stdout.write('\n')
  else:
    #data case
    for i,w in enumerate(sline):
      if i in metrics.keys():
        nextmetrics[i] = w
      sys.stdout.write(sline[i]+' ')
    for i in sorted(metrics):
      #output lagmetrics and reset them
      sys.stdout.write(metrics[i][1]+' ')
      metrics[i][1] = nextmetrics[i]
      nextmetrics[i] = 0
    sys.stdout.write('\n')
