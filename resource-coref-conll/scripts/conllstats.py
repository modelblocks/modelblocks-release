import sys
import pdb
import numpy as np

#generate coreference chain descriptive statistics like mean chain length, distance to last mention, etc.

with open(sys.argv[1], 'r') as iff:
    lines = iff.readlines()

chains = {} #{docid:{cid:[idx1,idx2,...],...},...}
DOC_START_PREFIX = "#begin document" 
DOC_END = "#end document"
NULLSYMBOL = "-"

doc_id = 0
for lineno, line in enumerate(lines): 
    if line.startswith(DOC_START_PREFIX):
        doc_id += 1
        chains[doc_id] = {}
        continue
    if line.startswith(DOC_END):
        continue
    fields = line.strip().split("\t")
    try:
        word = fields[1]
        corefid = fields[5]
    except:
        pdb.set_trace()
    if corefid != NULLSYMBOL:
        if corefid in chains[doc_id]:
            chains[doc_id][corefid].append(lineno) #add to existing chain
        else:
            chains[doc_id][corefid] = [lineno] #create new chain

print(chains[1])

distances = [] #distances between mentions
num_mentions = [] #each chain's length
num_chains = [] #chains in a doc

#for doc in chains:
for doc in [1]:
    #mentions in a chain
    #distances between mentions
    num_chains.append(len(chains[doc]))
    for cid in chains[doc]:
        num_mentions.append(len(chains[doc][cid]))
        for x in range(1,len(chains[doc][cid])-1):
            distance = chains[doc][cid][x] - chains[doc][cid][x-1]
            distances.append(distance)
            #if distance > 15:
            #    pdb.set_trace()
            
        
#print("distances: {}".format(distances))
print("word distance between mentions - mean: {}, median: {}, min/max: {}/{}, stddev: {}".format(np.mean(distances),np.median(distances),np.min(distances),np.max(distances),np.std(distances)))

#print("coref chain length - mean: {}, min/max:{}/{}, stddev: {}".format(mean,min,max,stdddev))
#print("num chains in a doc - mean: {}, min/max:{}/{}, stddev: {}".format(mean,min,max,stdddev))
