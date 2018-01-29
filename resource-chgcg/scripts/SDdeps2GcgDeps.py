import dependency
import DepsTools
import re
import sys

# take a sd dep and minus the word idx by 1
def decreaseWordIdx(d):
#    print(d)
    m = re.match("(^.*)-(.*)", d.head)
    if m.group() != None:
        headW = m.group(1)
        headIdx = int(m.group(2))-1
#        print("head", headW, headIdx)
        newHead = headW+"-"+str(headIdx)
    n = re.match("(^.*)-(.*)",  d.child)
    if n.group() != None:
        childW = n.group(1)
        childIdx = int(n.group(2))-1
#        print("child", childW, childIdx)
        newChild = childW+"-"+str(childIdx)
    
    newD = dependency.Dependency()
    newD.label = d.label
    newD.head = newHead
    newD.child = newChild
#    print(newD)
    return(newD)
    
##Match sd deps to gcg deps

sdF = open(sys.argv[1], 'r', encoding="utf8")
sdDeps = DepsTools.readDepsFromFile(sdF)
#print(len(sdDeps))
#gcgDeps = DepsTools.readDepsFromFile(sys.argv[2])
##for f in sdDeps:
#    for d in f:
#        print(d)
#    break



depList = []
sdeps = []
for s in sdDeps:
    for dp in s:
        d = decreaseWordIdx(dp)
        if d.label == "nsubj":
            newD = dependency.Dependency()
            newD.label = "1"
            newD.head = d.head
            newD.child = d.child
            sdeps.append(newD)
        elif d.label == "dobj":
            newD = dependency.Dependency()
            newD.label = "2"
            newD.head = d.head
            newD.child = d.child
            sdeps.append(newD)
        elif d.label == "rcmod":
            verb = d.child
            noun = d.head
            rcty = DepsTools.TypeRcmod(dp,s)
            # only subject --> object rel 
            if rcty == "s":
                newD = dependency.Dependency()
                newD.label = "2"
                newD.head = verb
                newD.child = noun
#                print("s", newD)
                sdeps.append(newD)

            # only object --> subject rel
            elif rcty == "o":
                newD = dependency.Dependency()
                newD.label = "1"
                newD.head = verb
                newD.child = noun
                sdeps.append(newD)
#                print("o", newD)

            # no sbj and obj -- sbj rel
            elif rcty == "n":
                newD = dependency.Dependency()
                newD.label = "1"
                newD.head = verb
                newD.child = noun
                sdeps.append(newD)
#                print("n", newD)

                
        else:
            sdeps.append(d)
    depList.append(sdeps)
    sdeps = []
            
#outF = open(sys.argv[2], 'w', encoding="utf8")
for s in depList:
    for d in s:
        print(d)
    print()

