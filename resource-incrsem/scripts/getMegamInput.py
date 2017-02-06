import sys
import re
import pickle

labelD = {}

num = 0
for line in sys.stdin:
  
  m = re.match("^([^ ]+) : ([^ ]+)$", line)
  if m != None:
    feat = m.group(1).strip()
    label = m.group(2).strip()
    if label not in labelD:
      num += 1
      labelD[label] = num    
      
    feat = feat.replace("=1", "")
    featList = re.split(",", feat)
#    print label
  print str(labelD[label]) + "\t" + " ".join(featList)

pickle.dump(labelD, open("FmodelLabel.pkl", "wb"))
#print len(labelD)
