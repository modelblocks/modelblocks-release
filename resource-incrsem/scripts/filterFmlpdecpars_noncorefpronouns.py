import sys
import re

#remove F model features where antecedent is null and k is a pronoun k such as f[01]&Top?&X-xX:he_1
#he, him, his, she, her, they, them, their, its
#multiple models for different subtypes of coreference?

#F depth syncat kvec kvec kvec corefOFF goldfek 
#fek is of format: F&E&K,  where k is cat:predarg
#e.g., 
# F 1 S [R-aN-bN:at_1] [Bot] [N-aD:!unk!_1][N-aD:!unk!_2] 0 1&&N:they_1
logfile = open("filterFmodel.log", "w+")

count = 0
for line in sys.stdin:
    #match null-antecedent predicting pronounk F training data
    _, _, _, _, _, corefOFF, K = line.split(" ")
    k = K.split("&")[2].strip()
    rem = re.match(r'[DN]:([hH]e|[hH]im|[Hh]is|[Ss]he|[Hh]er|[Tt]hey|[Tt]hem|[Tt]heir|[Ii]t|[Ii]ts|[Ii]tself|[Tt]hemselves|[Tt]hemself|[Hh]erself|[Hh]imself)_[01]', k)
    #rem = re.match(r'F.*acorefOFF=1.* : f[01]&&[DN]-PRTRM:([hH]e|[hH]im|[Hh]is|[Ss]he|[Hh]er|[Tt]hey|[Tt]hem|[Tt]heir|[Ii]t|[Ii]ts|[Ii]tself|[Tt]hemselves|[Tt]hemself|[Hh]erself|[Hh]imself)_[01].*', line)
    if corefOFF == "1" and rem is not None:
        logfile.write("found non-coreferent pronoun line to omit: {}\n".format(rem.group(0)))
        count += 1
    else:
        sys.stdout.write(line) 

logfile.write("total non-coref pronoun lines found: {}\n".format(count))

#TODO filter these from N model as well? not obvious how to implement since there is no word information to compare against...
