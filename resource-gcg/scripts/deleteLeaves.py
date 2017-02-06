# Replaces leaves from a linetrees file with the string 'LEAF'

import sys
sys.path.append('../resource-gcg/scripts/')
import tree

def replaceLeaves(t):
    for x in t.ch:
        if len(x.ch) == 0:
            x.c = 'LEAF!'
        else:
            replaceLeaves(x)
            
for line in sys.stdin:
    if line.strip() !='':
        inputTree = tree.Tree()
        inputTree.read(line)
        replaceLeaves(inputTree)
        print inputTree