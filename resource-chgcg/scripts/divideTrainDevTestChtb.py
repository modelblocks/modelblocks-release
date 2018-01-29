import pickle 
import tree
import re
import sys

def readStrFromTree(f):
    strDic = {}
    for line in file(f):
        t = tree.Tree()
        t.read(line)
        s = re.sub(" +", " ",t.leaf())
        strDic[s]=line
    return strDic



chtb = sys.argv[1]

devSent = readStrFromTree("genmodel/chgcg.same.dev.linetrees")    
testSent = readStrFromTree("genmodel/chgcg.same.test.linetrees")
trainSent = readStrFromTree("genmodel/chgcg.same.train.linetrees")    

print "dev", len(devSent), "test", len(testSent)
print "train", len(trainSent)


devFile = open("genmodel/dev.raw.sent","wb")
testFile = open("genmodel/test.raw.sent", "wb")
for s in devSent:
    devFile.write(s+'\n')
devFile.close()

for s in testSent:
    testFile.write(s+'\n')
testFile.close()
    
chtbTrain = open(sys.argv[2],"wb")
chtbDev = open(sys.argv[3],"wb")
chtbTest = open(sys.argv[4],"wb")

for line in file(chtb):
    t = tree.Tree()
    t.read(line)
    st = re.sub(" +"," ", t.leaf())
    stList = re.split(" ", st)
    newList = []
    for w in stList:
        if "*" not in w:
            newList.append(w)
    newSt = " ".join(newList)
#    print newSt
    if newSt in devSent:
        chtbDev.write(line)
    elif newSt in testSent:
        chtbTest.write(line)
    elif newSt in trainSent:
        chtbTrain.write(line)

chtbDev.close()
chtbTest.close()
chtbTrain.close()

