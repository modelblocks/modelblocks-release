import sys
import re

num = 0
for line in sys.stdin:
  num += 1
  m = re.match("^([^ ]+) : ([^ ]+)$", line)
  if m != None:
    feat = m.group(1).strip()
    label = m.group(2).strip()
    feat = feat.replace("=1", "")
    featList = re.split(",", feat)
#    print label
  print str(num)+"\t"+label+"\t"+" ".join(featList)


