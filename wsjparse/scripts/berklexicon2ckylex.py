#
# Simple script to go from berkeley's lexicon files to 
# our own CKY format
#
# usage:
# cat berk_lexicon | scripts/berklexicon2ckylex.py


import re
import sys

for line in sys.stdin:
        match = re.search('([^ ]+) ([^ ]+) \[(.*)\]', line)
        if match is not None:
                (m1,m2,m3) = match.group(1,2,3)
                m1 = m1.replace("&","-")
                #m2 = m2.replace("&","-")
                #### COMMENTED OUT B/C SHOULD BE HANDLED IN CORPUS PRE-PROC
                #if m1 == ':':
                #        # why you would use ':' for a production is beyond me, we will
                #        # replace the production : with CN
                #        m1 = "CN"
                #if m2.find(":") != -1:
                #        # ':' inside of a lexical item is also trouble, we will replace it with !colon!
                #        m2 = m2.replace(":","!colon!")
                i = 0
                for x in m3.split(","):
                        print("X %s_%d : %s = %s" % (m1,i, m2, x))
                        i+=1
