import sys
import tree
import pickle

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

print len(cat)
print lineNum
pickle.dump(cat, open("chtbCats.pkl","wb"))
