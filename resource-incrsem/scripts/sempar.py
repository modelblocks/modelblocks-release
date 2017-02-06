import sys
import re
from storestate import Sign, StoreState, getUnkWord
import numpy
import scipy.sparse
import collections
import math

################################################################################

class IndexMap(dict):
  def __missing__ ( this, key ):
    this[key] = len(this)
    return this[key]

################################################################################

indexDK = IndexMap()
indexFK = IndexMap()
datF  = [ ]
modelW = collections.defaultdict(list)
modelP = collections.defaultdict(dict)
modelJ = { }
modelA = collections.defaultdict(dict)
modelB = collections.defaultdict(dict)

sys.stderr.write( 'reading ' + sys.argv[1] + '...\n' )

## read model...
for line in open(sys.argv[1],'r'):
  id = line[0]
  try:
    predictor,response,weight = re.split(' [:=] ',line[2:-1])
  except Exception as e:
    sys.stderr.write( 'ERROR ON INPUT: ' + line )
  if id=='F': datF.append((indexDK[predictor],indexFK[response],float(weight)))
  if id=='W': modelW[response].append((predictor.partition(' ')[0],predictor.partition(' ')[2],float(weight)))
  if id=='P': modelP[tuple(predictor.split())][response] = float(weight)
  if id=='J': modelJ[tuple(predictor.split() + [response])] = float(weight)
  if id=='A': modelA[tuple(predictor.split())][response] = float(weight)
  if id=='B': modelB[tuple(predictor.split())][response] = float(weight)

sys.stderr.write( 'read.' )

matF = numpy.zeros((len(indexFK),len(indexDK)))

for pred,resp,wght in datF:
  matF[resp,pred]=wght

sys.stderr.write( 'loaded.\n' )


## for each sentence...
for line in sys.stdin:

  ## initialize beam...
  Beam = [ [ (0.0,StoreState()) ] ]

  ## for each word...
  for w_t in line.split():

    print( 'WORD: ' + w_t )

    ## add current time step to beam...
    Beam.append([])

    #sys.stderr.write ( ' ' + w_t )

    ## for each hypoth storestate...
    for lgpr_q_tdec1,q_tdec1 in Beam[-2]:

      ## calculate fprob denominator...
      fpredictors = [indexDK[feat[:-2]] for feat in q_tdec1.calcForkPredictors() if feat[:-2] in indexDK]
      for fpredr in q_tdec1.calcForkPredictors():
        print( '    fpredr:' + fpredr )
      fresponses  = numpy.exp ( matF * scipy.sparse.csc_matrix ( ( numpy.ones(len(fpredictors)), \
                                                                   (numpy.array(fpredictors), numpy.zeros(len(fpredictors))) ), \
                                                                 (matF.shape[1],1) ) )
      fnorm = sum( fresponses )

      ## for each lemma (context + label)...
      for k_p_t,l_p_t,probwgivkl in modelW.get(w_t,modelW[getUnkWord(w_t)]):

        ## for each no-fork or fork...
        for f in [0,1]:

          ## calc lex prob...
          fresponse = indexFK['f'+str(f)+'&'+k_p_t] if 'f'+str(f)+'&'+k_p_t in indexFK else indexFK['f'+str(f)+'&bot']
          probFork = float(fresponses[fresponse])/fnorm * modelP[q_tdec1.calcPretrmCatPredictors(f,k_p_t)].get(l_p_t,0.0) * probwgivkl

          ###print( '  ---> ' + k_p_t + ' ' + ','.join(q_tdec1.calcPretrmCatPredictors(f,k_p_t)) + ' ' + str(probFork) )

          if probFork>0.0:

            print( '      f: ' + str(f)+'&'+k_p_t + ' ' + str(float(fresponses[fresponse])) + ' / ' + str(fnorm) + ' * ' + str(modelP[q_tdec1.calcPretrmCatPredictors(f,k_p_t)].get(l_p_t,0.0)) + ' * ' + str(probwgivkl) + ' = ' + str(probFork) )

            ScoreJoin = { }

            ## calculate existing context sets...
            ancstr = q_tdec1.getAncstr(f)
            pretrm = Sign([k_p_t],l_p_t)
            lchild = q_tdec1.getLchild(f,pretrm)
            arityA = ancstr.getArity()
            jpredictors = q_tdec1.calcJoinPredictors(f,pretrm)
            ### for jpredr in jpredictors:
            ###   print( 'jpredr:'+ jpredr )

            ## for each no-join or join...
            for j in [0,1]:
              ###logscoreJ = sum( [ modelJ[feat[:-2]][val] for feat in jpredictors if feat[:-2] in modelJ for val in modelJ[feat[:-2]] ] )

              ## for each left and right child operation, compute left and right child context sets...
              for opL in ['1','2','3','I','M']:
                apredictors = q_tdec1.calcApexCatPredictors(f,j,opL,pretrm)
                for lA in modelA.get(apredictors,[]):   #LabelParent[ancstr.l,lchild.l,opL]:
                  #parent = Sign ( getParentContexts(opL,arityA,lchild), lA )
                  for opR in ['1','2','3','I','M']:
                    logscoreJ = sum( [ modelJ.get((feat[:-2],'j'+str(j)+'&'+opL+'&'+opR),0.0) for feat in jpredictors] )
                    bpredictors = q_tdec1.calcBrinkCatPredictors(f,j,opL,opR,pretrm,lA)
                    for lB in modelB.get(bpredictors,[]):   #LabelRchild[parent.l,lchild.l,rchild.l]:
                      #rchild = Sign ( getRchildContexts(opR,arityA,parent), lB )

                      ## calculate syn prob...
                      scoreJoin = math.exp(logscoreJ) * modelA[apredictors][lA] * modelB[bpredictors][lB]
                      if scoreJoin>0.0:
                        q_t = StoreState ( q_tdec1, f, j, opL, opR, lA, lB, pretrm )    # parent, rchild )
                        print( '        ' + str(q_tdec1) + ' ==(' + str(f) + ',' + str(j) + ',' + opL + ',' + opR + ',' + str(pretrm) + ')==> ' + str(q_t) + ' l' + str(lgpr_q_tdec1) + ' f' + str(math.log(probFork)) + ' j' + str(logscoreJ) + ' a' + str(math.log(modelA[apredictors][lA])) + ' b' + str(math.log(modelB[bpredictors][lB])) + ' =' + str(math.log(scoreJoin)) )
                        ScoreJoin[j,opL,opR,lA,lB] = (scoreJoin, q_t)

            jnorm = sum( [ scoreJoin for (_,(scoreJoin,q_t)) in ScoreJoin.items() ] )
            print( '      -- n' + str(math.log(jnorm) if jnorm>0.0 else -1000000) )
            for j,opL,opR,lA,lB in ScoreJoin:
              scoreJoin,q_t = ScoreJoin[j,opL,opR,lA,lB]
              logprob = lgpr_q_tdec1 + math.log(probFork) + math.log(scoreJoin) - math.log(jnorm)
              Beam[-1].append ( (logprob, q_t) )

    Beam[-1] = sorted(Beam[-1],reverse=True)[:10]
    sys.stderr.write( w_t + ' ' + str(len(Beam[-1])) + '\n' )

    for prob,ss in Beam[-1]:
      print( ' ' + str(ss) + ' = ' + str(prob) )
    sys.stdout.flush()

  sys.stderr.write('\n')


