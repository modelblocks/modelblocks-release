import sys
import re
import pdb

class Counter(object): #ummm, why not just use an integer here?
    def __init__(self,start=1):
        self.count = start
    def __call__(self):
        val = self.count
        self.count += 1
        return val

def invertChains(chains):
    inverted = {}
    for item in chains.items():
        inverted[item[1]] = item[0] 
    return inverted
    
def findChainId(idx, chains):
    for item in chains.items():
        if idx in item[1]:
            return item[0]
    return None

with open(sys.argv[1],'r') as iff:
    lines = iff.readlines()

data = {} #{article_num:[(word,corefidx),...],...}
chains = {} #{article_num: {cid:[idx,idx2,...],...}, ...}
cc = Counter()
article_idx = 0

for line in lines:
    if line.startswith("word pos"):
        continue
    if line.startswith("!ARTICLE"): 
        article_idx += 1
        continue
    try:
        word = line.split(" ")[0]
        corefidx = line.split(" ")[-2].strip()
        #data += [(word, corefidx)]
        if article_idx not in data:
            data[article_idx] = []
        data[article_idx].append((word, corefidx))
    except:
        pdb.set_trace()


#accumulate coref chains, assigning arbitrary chain id and create list of participating indices.
#chains = {article_num: {chain_id: [idx1,idx2,...], ...}, ...}
for article in data:
    if article not in chains:
        chains[article] = {}
    for i, datum in enumerate(data[article]):
        offset = abs(int(datum[1]))
        if offset != 0:
            pointsto = i - offset
            cid = findChainId(pointsto, chains[article])
            if cid is not None: #if pointsto is in another chain already
                chains[article][cid] += [i] #add current location to existing chain
            else:
                #get new id, add pointsto and i to it
                chains[article][cc()] = [pointsto, i]

for article in data:
    print("#begin document ({});".format(article))
    for i, datum in enumerate(data[article]):
        cid = findChainId(i,chains[article])
        word = datum[0]
        if cid is not None:
            cid = "("+str(cid)+")"
        else:
            cid = "-"
        print("{}\t0\t0\t0\t{}".format(word,cid))
    print("#end document")

