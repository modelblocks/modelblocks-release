import sys
import re
import pdb

def is_failtree(linetree):
    #perl -pe 's/.*\([^a-z]+ .*//'
    if re.search(r'.*\([^a-z]+ .*', linetree) is not None:
        return True
    else:
        return False

def gen_failtree(words):
    #use failnode labels?
    assert len(words) > 0 
    #if ("" not in words):
    #    pdb.set_trace()
    FTAG = "FAIL"
    if len(words) == 1: #base case
        return "({} {})".format(FTAG,words[0]) #(X word)
    else: #recursion
        return "({} {} {})".format(FTAG, gen_failtree([words[0]]), gen_failtree(words[1:])) #(X (X word) recursion)

for line in sys.stdin:
    if is_failtree(line):
        #get words from failed bracketed linetree
        words = [x[1:-1] for x in re.findall(r' [^\) ]+\)', line)] #TODO test this. should match things with a space, then anynumber of non-paren chars, then a close paren.
        #generate right-branching failtree, keeping words.
        print(gen_failtree(words)) 
    else:
        print(line[:-1])
    
