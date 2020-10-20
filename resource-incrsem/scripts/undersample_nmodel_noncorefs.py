import sys
import random
import configparser

#undersample_rate = .2 #print only 20% of negative coref data
#undersample_rate = 1.0 #off
config = configparser.ConfigParser(allow_no_value=True)
config.read(sys.argv[1])
undersample_rate = config["NModel"].getfloat("UndersampleNRate")

for line in sys.stdin:
    if line.endswith("1 : 0\n"): #negative example of coreference.  corefON, label=false 
        if random.random() < undersample_rate:
            print(line.strip())
    else:
        print(line.strip())
