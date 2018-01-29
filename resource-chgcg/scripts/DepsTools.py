import re
import dependency

#read dependencies from file
#[[deps of s1], [deps of s2],...]
def readDepsFromFile(f):
    fdeps = []
    sDeps = []
    for line in f:
#        print(line)
        if line != "\n":
            d = dependency.Dependency()
            d.read(line.strip())
            #print(d.label)
            sDeps.append(d)
        else:
            fdeps.append(sDeps)
            sDeps = []
    return fdeps
                                            

# take a rcmod dep and the dep list where it occurs to return its type: so, s, o, n.
# s=sbj, o=obj, n=none
def TypeRcmod(rc, ds):
    v = rc.child
    s = False
    o = False
    for d in ds:
        if d.head == v and d.label == "nsubj":
            s = True
        if d.head == v and d.label == "dobj":
            o = True
    if s == True and o == True:
        return "so"
    elif s == True and o == False:
        return "s"
    elif s == False and o == True:
        return "o"
    else:
#        print(rc)
        return "n"
            
