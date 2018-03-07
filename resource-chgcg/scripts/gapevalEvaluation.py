import sys
import dependency
from DepsTools import readDepsFromFile

def addCorFile(deps):
    newDeps = []
    for s in deps:
        sDeps = addCor(s)
        newDeps.append(sDeps)
    return newDeps

def addCor(ds):
    newdeps = []
    for d in ds:
        newdeps.append(d)
    for d in ds:
        if d.label == "cor" or d.label == "conj":
            c1 = d.head
            c2 = d.child
            for i in ds:
                if c1 == i.head:
                    newD = dependency.Dependency()
                    newD.label = i.label
                    newD.head = c2
                    newD.child = i.child
                    newdeps.append(newD)

                elif c2 == i.head:
                    newD = dependency.Dependency()
                    newD.label = i.label
                    newD.head = c1
                    newD.child = i.child
                    newdeps.append(newD)
                elif c1==i.child:
                    newD = dependency.Dependency()
                    newD.label = i.label
                    newD.head = i.head
                    newD.child = c2
                    newdeps.append(newD)
                elif c2==i.child:
                    newD = dependency.Dependency()
                    newD.label = i.label
                    newD.head = i.head
                    newD.child = c1
                    newdeps.append(newD)
    return newdeps

def unlabelize(deps):
    newDeps = []
    for s in deps:
        sDeps = []
        for d in s:
            newd = (d.head, d.child)
            sDeps.append(newd)
        newDeps.append(sDeps)
        sDeps = []
    return newDeps

def undirection(deps):
    newDeps = []
    for s in deps:
        sDeps = []
        for d in s:
            newd = set([d.head, d.child])
            sDeps.append(newd)
        newDeps.append(sDeps)
        sDeps = []
    return newDeps


def evaluDeps(keyDeps, autoDeps):
    count = 0
    score = 0
    if len(keyDeps) == len(autoDeps):
        for i in range(0, len(keyDeps)):
            if len(keyDeps[i]) > 1:
                for d in keyDeps[i]:
                    count += 1
                    for m in set(autoDeps[i]):
                        if d.is_same(m):
                            score += 1
            else:
                for d in keyDeps[i]:
                    count += 1
                    for m in autoDeps[i]:
                        if d.is_same(m):
                            score += 1
    else:
        for i in range(0, len(keyDeps)):
            print(keyDeps[i])
            print(autoDeps[i])
            print()
            break
    return (count, score, score/count)


def evaluList(keyDeps, autoDeps):
    count = 0
    score = 0
    if len(keyDeps) == len(autoDeps):
        for i in range(0, len(keyDeps)):
            if len(keyDeps[i]) > 1:
                for d in keyDeps[i]:
                    count += 1
                    if d in autoDeps[i]:
                        score += 1
            else:
                for d in keyDeps[i]:
                    count += 1
                    if d in autoDeps[i]:
                        score += 1
    else:
        for i in range(0, len(keyDeps)):
            print(keyDeps[i])
            print(autoDeps[i])
            print()
            break
    return (count, score, score/count)


keyF = open(sys.argv[1], "r")
autoF = open(sys.argv[2], "r")

keyDeps = addCorFile(readDepsFromFile(keyF))
autoDeps = addCorFile(readDepsFromFile(autoF))



print()
print("--------------------------------------------------")
print("Gapeval of", sys.argv[2])
print("# of sentencs in key deps", len(keyDeps))
print("# of sentences in auto deps",  len(autoDeps))


#score = 0
#count = 0
#if len(keyDeps) == len(autoDeps):
#    for i in range(0, len(keyDeps)):
#        if len(keyDeps[i]) > 1:
#            for d in keyDeps[i]:
#                count += 1
#                for m in set(addCor(autoDeps[i])):
#                    if d.is_same(m):
#                        score += 1
#        else:
#            for d in keyDeps[i]:
#                count += 1
#                for m in autoDeps[i]:
#                    if d.is_same(m):
#                        score += 1
#else:
#    for i in range(0, len(keyDeps)):
#        print(keyDeps[i])
#        print(autoDeps[i])
#        print()
#        break

count = 0
score = 0
recall = 0
print("Labeled evaluation")
count, score, recall = evaluDeps(keyDeps, autoDeps)
print("Total # of deps", count)
print("# of correctly predicted" , score)
print("Recall", score/count)
print()


unlabelKey = unlabelize(keyDeps)
unlabelAuto = unlabelize(autoDeps)
count = 0
score = 0
recall = 0

print("UnLabeled evaluation")
count, score, recall = evaluList(unlabelKey, unlabelAuto)
print("Total # of deps", count)
print("# of correctly predicted" , score)
print("Recall", score/count)
print()


undirKey = undirection(keyDeps)
undirAuto = undirection(autoDeps)
count = 0
score = 0
recall = 0

print("Undirectioned evaluation")
count, score, recall = evaluList(undirKey, undirAuto)
print("Total # of deps", count)
print("# of correctly predicted" , score)
print("Recall", score/count)
print("--------------------------------------------------------------------")
print()

