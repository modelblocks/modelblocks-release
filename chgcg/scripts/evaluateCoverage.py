import sys
import pickle
import tree


#26062 trees in CTB
ip = sys.argv[1]
lineNum = 0
cat = []
for line in file(ip):
    lineNum += 1
    t = tree.Tree()
    t.read(line)
    for c in t.cats():
        if c not in cat:
            cat.append(c)
print "Number of categories in Chinese Penn Treebank", len(cat)
print lineNum
#pickle.dump(cat, open("chtbCats.pkl","wb"))
                                    

#cats = pickle.load(open("chtbCats.pkl"))
cat.remove("PU")
cat.remove("-NONE-")

#print len(cats)
inF = sys.argv[2]
unannotF = open(sys.argv[3],"wb")
annotF = open(sys.argv[4],"wb")
stopNum = sys.argv[5]

diffSen = [179, 503, 3205, 3362, 4746, 5345, 6312, 6313, 7752, 10934, 18670]
unannotList = []
annotList =[]
lineNum = 0
annot = 0
for line in file(inF):
#    print line
    lineNum += 1
#    print lineNum
    if lineNum in diffSen:
        continue
    if lineNum%1000 == 0:
        print str(lineNum)+" lines have been processed."
    t = tree.Tree()
    t.read(line)
    newCats = t.cats()
    newCat = []
    for c in newCats:
        if c not in newCat:
            newCat.append(c)
    if "" in newCat:
        newCat.remove("")

    inter = set(cat).intersection(set(newCat))
#    print len(inter), inter
    if len(inter) == 0:
        annot += 1
        annotList.append(lineNum)
        annotF.write(line)
    else:
        unannotList.append(lineNum)
    if lineNum == int(sys.argv[5]):
        break


tbline = 0
for line in file(sys.argv[1]):
    tbline += 1
    if tbline in unannotList:
        unannotF.write(line)
annotF.close()
unannotF.close()

print "number of total sentences", lineNum
print "number of fully annotated",annot
print "annotation rate", float(annot)/lineNum
print "unannotated trees", len(unannotList)
