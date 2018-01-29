import re
import tree
import pickle
#import editdist
import sys

#generating linetrees for Evalb testing
#pull out parses from grammarTester
#match ccg and gcg
#unlabelize the categories

def readDicFromFile(f):
    strDic = {}
    for line in f:
        t = tree.Tree()
        t.read(line)
        rawS = re.sub("\*[^ ]*", "", t.leaf())
        s = re.sub(" +", " ",rawS)
        if s not in strDic:
            strDic[s] = line.strip()
    return strDic

def readStrFromTree(line):
    s = ''
    t = tree.Tree()
    t.read(line)
    rawS = re.sub("\*[^ ]*", "", t.leaf())
    s = re.sub(" +", " ",rawS)
    return s

def getDicFromFile(f):
    dic = {}
    for line in file(f):
        t = tree.Tree()
        t.read(line)
        s = re.sub(" +", " ",t.leaf())
        dic[s] = line
    return dic


inputF = open(sys.argv[1], 'r')
srcF = open(sys.argv[2], 'r')

inputS = readDicFromFile(inputF)
srcS = readDicFromFile(srcF)
#print(inputS)
print("There are ", len(inputS), "unique trees")
optTrees = []
optF1 = open(sys.argv[3], 'w')
optF2 = open(sys.argv[4], 'w')


for line in inputS:
    if line in srcS:
        optTrees.append(srcS[line])
        optF1.write(inputS[line]+"\n")
        optF2.write(srcS[line]+"\n")
    else:
        print(inputS[line])
    
optF1.close()
optF2.close()

print(len(optTrees))
