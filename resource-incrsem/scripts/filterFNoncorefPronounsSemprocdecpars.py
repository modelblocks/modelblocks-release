import sys
import re

#remove F model features where antecedent is null and k is a pronoun k such as f[01]&Top?&X-xX:he_1
#he, him, his, she, her, they, them, their, its
#multiple models for different subtypes of coreference?

#example F model line - "F <comma-sepfeatvals> : fresponse = count
#F d0&tT=1,d0&&Top&=1,acorefON=1,d0&&&N-aD-PRTRM:!unk!_1=1,d0&&&N-aD-PRTRM:!unk!_1=1,d0&&&N-aD-PRTRM:!unk!_1=1,d0&&&N-aD-PRTRM:!unk!_1=1,d0&&&N-PRTRM:!unk!_1=1,d0&&&N-PRTRM:i_1=1,d0&&&N-PRTRM:he_1=1,d0&&&N-PRTRM:he_1=1,d0&&&FAIL-PRTRM:!unk!_0=1,d0&&&D-PRTRM:!unk!_0=1,d0&&&N-aD-PRTRM:!unk!er_1=1 : f1&&N-PRTRM:he_1 = 1
logfile = open("filterFmodel.log", "w+")

count = 0
for line in sys.stdin:
    #match null-antecedent predicting pronounk F training data
    rem = re.match(r'F.*acorefOFF=1.* : f[01]&&N-PRTRM:(he|him|his|she|her|they|them|their|it|its)_1.*', line)
    if rem is not None:
        logfile.write("found non-coreferent pronoun line to omit: {}\n".format(rem.group(0)))
        count += 1
    else:
        sys.stdout.write(line) 

logfile.write("total non-coref pronoun lines found: {}\n".format(count))

#TODO filter these from N model as well? not obvious how to implement since there is no word information to compare against...
