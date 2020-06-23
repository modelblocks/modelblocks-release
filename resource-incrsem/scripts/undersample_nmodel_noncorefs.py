import sys
import random

undersample_rate = .2


for line in sys.stdin:
    if line.endswith("1 : 0\n"): #negative example of coreference.  corefON, label=false 
        if random.random() < undersample_rate:
            print(line.strip())
    else:
        print(line.strip())
