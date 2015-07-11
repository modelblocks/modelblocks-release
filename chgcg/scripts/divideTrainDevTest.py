import pickle 
import tree
import re

def readStrFromTree(f):
    strDic = {}
    for line in file(f):
        t = tree.Tree()
        t.read(line)
        s = re.sub(" +", " ",t.leaf())
        strDic[s]=line
    return strDic


dev = "../genmodel/chccg.dev.linetrees"
train = "../genmodel/chccg.train.linetrees"
# the original test set
test = "../genmodel/chccg.test.linetrees"
# the additional test set
#test1 = "../genmodel/chccg.test1.linetrees"
gcg = "../genmodel/annot-chgcg.linetrees"

devSent = readStrFromTree(dev)    
testSent = readStrFromTree(test)
#test1Sent = readStrFromTree(test1)    
trainSent = readStrFromTree(train)    

#devSent = pickle.load(open("dev.raw.sent.pkl"))
#testSent = pickle.load(open("test.raw.sent.pkl"))
#trainSent = pickle.load(open("train.raw.sent.pkl"))
print "dev", len(devSent), "test", len(testSent)
#print "test1", len(test1Sent),
print "train", len(trainSent)

#pickle.dump(devSent,open("dev.raw.sent.pkl", "wb"))
#pickle.dump(testSent,open("test.raw.sent.pkl", "wb"))
#pickle.dump(trainSent,open("train.raw.sent.pkl", "wb"))

devFile = open("../genmodel/dev.raw.sent","wb")
testFile = open("../genmodel/test.raw.sent", "wb")
for s in devSent:
    devFile.write(s+'\n')
devFile.close()

for s in testSent:
    testFile.write(s+'\n')
testFile.close()
    
gcgDev = open("../genmodel/chgcg.same.dev.linetrees","wb")
gcgTest = open("../genmodel/chgcg.same.test.linetrees","wb")
#gcgTest1 = open("../genmodel/chgcg.same.test1.linetrees","wb")
gcgTrain = open("../genmodel/chgcg.train.linetrees","wb")

ccgSameDev = open("../genmodel/chccg.same.dev.linetrees","wb")
ccgSameTest = open("../genmodel/chccg.same.test.linetrees","wb")
#ccgSameTest1 = open("../genmodel/chccg.same.test1.linetrees","wb")
ccgSameTrain = open("../genmodel/chccg.same.train.linetrees","wb")

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
#    elif st in test1Sent:
#        gcgTest1.write(line)
#        ccgSameTest1.write(test1Sent[st])
    elif st in trainSent:
        gcgTrain.write(line)
        ccgSameTrain.write(trainSent[st])

gcgDev.close()
gcgTest.close()
#gcgTest1.close()
gcgTrain.close()

ccgSameDev.close()
ccgSameTest.close()
#ccgSameTest1.close()
ccgSameTrain.close()
