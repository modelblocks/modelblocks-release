import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
import tree

print("word syncat")
for line in sys.stdin:
    if (line.strip() !='') and (line.strip()[0] != '%'):
        T = tree.Tree()
        T.read(line)
        for word, cat in zip(T.words(), T.syncats()):
            print(word + " " + cat)
