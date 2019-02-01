import sys
import os
import re
import collections
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
import tree
import gcgtree

for line in sys.stdin:
  print( gcgtree.GCGTree(line.strip()) ) #strip to prevent adding additional blank lines

