import sys
import re
import pdb
import random

DOCSTART = "0001:(U !ARTICLE)"
LOGFILE = open("numberededitabletrees2conll.log","w+")

class CorefTracker:
    def __init__(self):
        self.wordlist = [] #e.g., [('word1','0101'),('word2','0102'),...]
        self.chains = {} #e.g., {'0102':['0102','0112',...],...}
        self.article_id = 0
        self.filehandle = sys.stdout

    def printFooter(self):
        print("#end document",file = self.filehandle)

    def printHeader(self):
        print("#begin document ({});".format(self.article_id), file = self.filehandle)

    def articleIncrement(self):
        self.article_id += 1

    def clear(self):
        self.wordlist = []
        self.chains = {}
        #self.filehandle.close()

    def getWord(self, line):
        match = re.search(r"\s([^\) ]+)\)+$",line) #space, anything not a space or close paren, close paren
        assert match is not None
        return match.group(1)

    def getId(self, line):
        return str(self.article_id).zfill(3) + line.split(":")[0] #assumes zero-padded to 4 places, but annotation is sometimes not - fix annotation

    def addWord(self, line):
        id = line.split(":")[0] #assumes 2 digit sentid, 2 digit word id.
        id = str(self.article_id).zfill(3) + id #prepend article id
        word = self.getWord(line)
        self.wordlist += [(id,word)]

    def findChainId(self, annot):
        #find if it's in an existing chain
        for chainid in self.chains:
            if annot in self.chains[chainid]:
                return chainid
        return None

    def processCoref(self, line):
        #check for anaphoricity
        result = re.search(r".*-[nm]([0-9]+) ", line)
        if result is not None:
            annot = str(self.article_id).zfill(3) + result.group(1).zfill(4)
            chainid = self.findChainId(annot)
            if chainid is not None:
                self.chains[chainid] += [self.getId(line)]#append current id to that chain
            else: #create new chain
                self.chains[annot] = [annot,self.getId(line)]

    def printCoNLL(self):
        if self.wordlist is [] or self.article_id == 0:
            return
        else:
            self.printHeader()
            for item in self.wordlist:
                word = item[1]
                wordnum = item[0][-2:]
                corefid = self.findChainId(item[0])
                if corefid == None:
                    corefid = "-"
                else:
                    corefid = "("+corefid+")"
                print("{}\t{}\t\t{}\t0\t{}".format(item[0], word, wordnum, corefid), file=self.filehandle)
            self.printFooter()

ct = CorefTracker()

with open(sys.argv[1],'r') as iff:
    for line in iff:
        if line.startswith(DOCSTART):
            print("found article boundary",file=LOGFILE)
            print("coref tracker contains {} chains".format(len(ct.chains)),file=LOGFILE)
            ct.printCoNLL() #print chains for previous article
            ct.clear()
            ct.articleIncrement()
            continue
        ct.addWord(line)
        ct.processCoref(line)
    ct.printCoNLL()
    ct.filehandle.close()
    LOGFILE.close()



