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


dev = "genmodel/chccg.dev.linetrees"
train = "genmodel/chccg.train.linetrees"
# the original test set
test = "genmodel/chccg.test.linetrees"
# the additional test set
#test1 = "../genmodel/chccg.test1.linetrees"
gcg = sys.argv[1]

devSent = readStrFromTree(dev)    
testSent = readStrFromTree(test)
#test1Sent = readStrFromTree(test1)    
trainSent = readStrFromTree(train)    

print "dev", len(devSent), "test", len(testSent)
print "train", len(trainSent)

    
gcgDev = open("genmodel/chgcg.same.dev.linetrees","wb")
gcgTest = open("genmodel/chgcg.same.test.linetrees","wb")
#gcgTest1 = open("../genmodel/chgcg.same.test1.linetrees","wb")
gcgTrain = open("genmodel/chgcg.same.train.linetrees","wb")

ccgSameDev = open("genmodel/chccg.same.dev.linetrees","wb")
ccgSameTest = open("genmodel/chccg.same.test.linetrees","wb")
#ccgSameTest1 = open("../genmodel/chccg.same.test1.linetrees","wb")
ccgSameTrain = open("genmodel/chccg.same.train.linetrees","wb")

extraGcg = open("genmodel/chgcg.extra.linetrees", "wb") 
notFound = 0
for line in file(gcg):
    t = tree.Tree()
    t.read(line)
    st = re.sub(" +"," ", t.leaf())
    if st in devSent:
        gcgDev.write(line)
        ccgSameDev.write(devSent[st])
    elif st in testSent:
        gcgTest.write(line)
        ccgSameTest.write(testSent[st])
    elif st in trainSent:
        gcgTrain.write(line)
        ccgSameTrain.write(trainSent[st])
    else:
        notFound += 1
        extraGcg.write(line)
        
gcgDev.close()
gcgTest.close()
gcgTrain.close()

ccgSameDev.close()
ccgSameTest.close()
ccgSameTrain.close()
print notFound
