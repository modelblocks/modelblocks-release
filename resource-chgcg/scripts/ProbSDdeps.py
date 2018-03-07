import dependency
import DepsTools
import re
import sys

##calcuate the most likely gcg dep match sd deps

sdF = open(sys.argv[1], 'r', encoding="utf8")
sdDeps = DepsTools.readDepsFromFile(sdF)
print(len(sdDeps))
gcgF = open(sys.argv[2], 'r', encoding="utf8")
gcgDeps = DepsTools.readDepsFromFile(gcgF)
print(len(gcgDeps))
#for f in sdDeps:
#    for d in f:
#        print(d)
#    break

rc = {}
for s in sdDeps:
#    print(s)
    for d in s:
        if d.label == "rcmod":
            rcty = DepsTools.TypeRcmod(d, s)
            if rcty not in rc:
                rc[rcty] = 1
            else:
                rc[rcty] = rc[rcty] + 1

print(rc)

depType = {}

if len(sdDeps) == len(gcgDeps):
    for i in range(0, len(sdDeps)-1):
        #print(i)
        for d in sdDeps[i]:
            if d.label == "rcmod":
                rcty = DepsTools.TypeRcmod(d, sdDeps[i])
                verb = d.child
                noun = d.head
                print("re",d)
                for gd in gcgDeps[i]:
                    if gd.head == verb and gd.child == noun:
                        print("found match", gd)
                        #label = gd.label
                        dt = rcty+"_"+gd.label
#                        print(dt)
                        if dt not in depType:
                            depType[dt] = 1
                        else:
                            depType[dt] = depType[dt] + 1


else:
    print(len(sdDeps), len(gcgDeps))

for k,v in sorted(depType.items(), key=lambda x:x[1]):
    print("%s:%d" %(k,v))

sTotal = 0
soTotal = 0
oTotal = 0
nTotal = 0

for k in depType:
    if re.split("_", k)[0] == "s":
        sTotal = sTotal+ depType[k]
    elif re.split("_", k)[0] == "o":
        oTotal = oTotal+ depType[k]
    elif re.split("_", k)[0] == "so":
        soTotal = soTotal+ depType[k]
    else:
        nTotal = nTotal + depType[k]

print("s: ", sTotal)
print("o: ", oTotal)
print("so: ", soTotal)
print("n: ", nTotal)

for k in depType:
    if re.split("_", k)[0] == "s":
        print(k, depType[k], depType[k]/sTotal)
    elif re.split("_", k)[0] == "o":
        print(k, depType[k], depType[k]/oTotal)
    elif re.split("_", k)[0] == "so":
        print(k, depType[k], depType[k]/soTotal)
    else:
        print(k, depType[k], depType[k]/nTotal)
    
            

