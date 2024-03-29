import sys, os
#from itertools import izip
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
import tree

# File containing stripped linetrees
trees = open(sys.argv[1], 'r')
# File containing all words in Dundee corpus
words = open(sys.argv[2], 'r')
#curSent = words.next().split()
curSent = next(words).split()

def plugWords(T):
    global curSent
    if len(T.ch) == 0:
        if len(curSent) > 0:
            T.c = curSent.pop(0)
        else:
#            curSent = words.next().split()
            curSent = next(words).split()
            T.c = curSent.pop(0)
    else:
        for x in T.ch:
            plugWords(x)

T = tree.Tree()

# test = '(S-lS (S (Q-bA Are) (A-lA (N-lA (N tourists) (A-aN-lM (A-aN-v enticed) (R-aN-lM (R-aN-bN by) (N-lA (R-aN-x-lM (R-aN-x these)) (N attractions))))) (A-aN (A-aN-bN threatening) (N-lA (D-lA (D their)) (N-aD (A-aN-x-lM (A-aN-x very)) (N-aD existence)))))) (. ?))'
# testwrds = 'Are tourists enticed by these attractions threatening their very existence ?'
# T.read(test)
# plugWords(T, testwrds.split())
# print T

for line in trees:
    T.read(line)
    plugWords(T)
    print( T )

words.close()
trees.close()
