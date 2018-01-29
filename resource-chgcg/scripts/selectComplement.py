import re
import tree
import pickle
#import editdist
import sys

#generating linetrees for Evalb testing
#pull out parses from grammarTester
#match ccg and gcg
#unlabelize the categories

def readStrFromFile(f):
    strList = {}
    for line in f:
        t = tree.Tree()
        t.read(line)
        rawS = re.sub("\*[^ ]*", "", t.leaf())
        s = re.sub(" +", " ",rawS)
        if s not in strList:
            strList[s]=line.strip()
    return strList

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

def listSents(f):
    listS = []
    for line in f:
        if line not in listS:
            listS.append(line)
    return listS

smallF = open(sys.argv[1], 'r')
bigF = open(sys.argv[2], 'r')

#smallS = listSents(smallF)
#bigS = listSents(bigF)
smallS = readStrFromFile(smallF)
bigS = readStrFromFile(bigF)
extra = open(sys.argv[3], 'w')
diffS = []
for s in bigS:
    if s not in smallS:
        diffS.append(s)
        extra.write(bigS[s]+"\n")
extra.close()


print("number of sentences in small file", len(smallS))
print("number of sentences in big file", len(bigS))

#diffS = set(bigS)-set(smallS)
#print("number of unique sentences in small", len(set(smallS)))
#print("number of unique sentences in big", len(set(bigS)))
print("number of sentences in extra = big-small file", len(diffS))


#check = set(smallS)-set(bigS)
#checkF = open("checkThese.linetrees", "w")
#for s in list(check):
#    checkF.write(s)
#checkF.close()
#
#overlap = set(smallS).intersection(set(bigS))
#print("this many in test are found in big", len(overlap))
#
#extra = open(sys.argv[3], 'w')
#for s in list(diffS):
#    extra.write(s)
#extra.close()
