#
# a script to go from berkeley's .grammar file format to our CKY parser's format

import re
import sys


#startsyms = 1

for line in sys.stdin:
        arr = line.split(' ')
        if len(arr) == 4:
                arr[0] = arr[0].replace("&","-")
                arr[2] = arr[2].replace("&","-")
                #unary rule
                if(arr[0] == "ROOT_0"):
                        if(arr[2] != "ROOT_0"):
                                print("Cr : %s = %s" % (arr[2],arr[3]), end='')
                else:
                        if arr[0] != arr[2]:
                                print("Cu %s : %s = %s" % (arr[0],arr[2],arr[3]), end='')
        elif len(arr) == 5:
                # replace rules like :_0 with CN_0
                arr[0] = arr[0].replace("&","-") #### DELETED, SHOULD BE HANDLED IN CORPUS PROC ####
                arr[2] = arr[2].replace("&","-") #.replace(":","CN")
                arr[3] = arr[3].replace("&","-") #.replace(":","CN")
                # binary rule
                print("CC %s : %s %s = %s" % (arr[0],arr[2],arr[3],arr[4]), end='')
