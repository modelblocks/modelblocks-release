#python calc_perplexity.py < infile > outfile
# computes the perplexity of a grammar on a dataset
# infile: line-delimited log-probs of observations from a test dataset
# outfile: the perplexity computed from the observation probabilities in infile

import math
import sys

norm = 0
xent = 0
for line in sys.stdin.readlines():
    sline = line.strip()
    if sline != []:
        norm += 1
        xent += math.log(math.exp(float(sline)),2)
print 'cross entropy:',-xent, -xent/norm
print 'perplexity:',2**(-1*xent/norm)
