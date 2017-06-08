import sys
import random
import numpy
import collections
import math

numY     = int(sys.argv[1])
NUMITERS = int(sys.argv[2])
alpha    = float(sys.argv[3])
beta     = float(sys.argv[4])
refstart,refend = (sys.argv[5],('_'+sys.argv[6])) if len(sys.argv) > 6 else (sys.argv[5]+'_','_')
sys.stderr.write( 'numY = '     + str(numY)+'\n' )
sys.stderr.write( 'NUMITERS = ' + str(NUMITERS)+'\n' )
sys.stderr.write( 'alpha = '    + str(alpha)+'\n' )
sys.stderr.write( 'beta = '     + str(beta)+'\n' )
sys.stderr.write( 'refstart = ' + str(refstart)+'\n' )
sys.stderr.write( 'refend = '   + str(refend)+'\n' )

#totcounts = 0.0
#totcountsY = [0.0] * numY
countsK = collections.defaultdict(float)

## load refconts...
RC = { }
for line in sys.stdin:
  ls = line.split()
  if len(ls)>1 and ls[0].endswith('s') and [] != [True for k in ls[1:] if k.startswith(refstart) and k.endswith(refend)]:
    RC[ls[0]] = ls[1:]
    for k in ls[1:]: countsK[k] += 1
numK = len(countsK)
sys.stderr.write( 'numK = ' + str(numK) + '\n' )

'''
## remove rare words...
for r in RC:
  for k in RC[r]:
    if countsK[k]<100000:
      RC[r].remove(k)
'''

for k in RC.values()[0:10]:
  sys.stderr.write( str(k) + '\n' )

'''
RC = { str(k)+'s' : v for k,v in enumerate( [['N-aD:a_1','b']]*600 + [['N-aD:c_1','d']]*400 ) }
for k in RC:
  for v in RC[k]:
    countsK[v]+=1
#sys.stderr.write( str(RC) )
'''
totcountsYforR = { r : ([0.0] * numY) for r in RC }
totcountsKgivY = [ { k : 0.0 for k in countsK } for y in range(0,numY) ]

lgprBest = -1000000000.0
iterBest = 0

## for each iter...
for i in range(0,NUMITERS):

  ## counts for current iteration
  countsY     = [0.0] * numY
  countsYforR = { r : ([0.0] * numY) for r in RC }
  countsKgivY = [ { k : 0.0 for k in countsK } for y in range(0,numY) ]

  lgpr = 0.0

  ## skip first iter as no model yet...
  if i>0:
    ## for each ref...
    for r in RC:

      ## for each context of ref...
      for k in RC[r]:

        ## calc distrib...
#        modYpost = [ (modY[y] * numpy.prod([ modKgivY[y][k] for k in RC[r] ])) for y in range(0,numY) ]
        modYpost = [ (modYforR[r][y] * modKgivY[y][k]) for y in range(0,numY) ]
        lgpr += math.log(sum(modYpost[y] for y in range(0,numY)))
        tot = sum(modYpost)
        modYpost = [ v/tot for v in modYpost ]

        ## sample hidd var from distrib...
        y = numpy.random.choice(numY,p=modYpost)

        ## increment counts...
        countsY[y] += 1.0
        countsYforR[r][y] += 1.0
        totcountsYforR[r][y] += 1.0
        for k in RC[r]:
          countsKgivY[y][k] += 1.0
          totcountsKgivY[y][k] += 1.0

  sys.stderr.write( 'iteration=' + str(i) + ' : logprob=' + str(lgpr) + ' ' + str(countsY) + '\n' )
  sys.stderr.flush()

  '''
  ## bail after 10 iters without improvement...
  if lgpr > lgprBest or i<=1:
    lgprBest,iterBest = lgpr,i
#    totcountsYforR = { r : ([0.0] * numY) for r in RC }
#    totcountsKgivY = [ { k : 0.0 for k in countsK } for y in range(0,numY) ]
  if i > iterBest+10: break
  '''

  ## use first half of iters for burn-in...
  if i < NUMITERS/2:
    totcountsYforR = { r : ([0.0] * numY) for r in RC }
    totcountsKgivY = [ { k : 0.0 for k in countsK } for y in range(0,numY) ]

  ## sample prior from Dirichlet (uneven prior moves larger counts to end)...
  modYforR = { }
  for r in RC:
    modYforR[r] = [ random.gammavariate(a+alpha,1.0) for a in countsYforR[r] ]
    denom = sum(modYforR[r])
    modYforR[r] = [ v/denom for v in modYforR[r] ]

  ## sample likelihood from Dirichlet...
  modKgivY = [{}] * numY
  for y in range(0,numY):
    modKgivY[y] = { k : random.gammavariate(a+beta,1.0) for k,a in countsKgivY[y].iteritems() }
    denom = sum(modKgivY[y].values())
    modKgivY[y] = { k : v/denom for k,v in modKgivY[y].iteritems() }

## final printout of prior...
for r in RC:
  for y in range(0,numY):
    print 'Y ' + r +' : ' + str(y) + ' = ' + str(totcountsYforR[r][y]/sum(totcountsYforR[r]))

## final printout of likelihood...
for y in range(0,numY):
  tot = sum(totcountsKgivY[y][k] for k in totcountsKgivY[y])
  for k,p in sorted(totcountsKgivY[y].items(), key=lambda value: value[1], reverse=True):
    if p > 0.0:
      print 'K ' + str(y) + ' : ' + k + ' = ' + str(p/tot)



