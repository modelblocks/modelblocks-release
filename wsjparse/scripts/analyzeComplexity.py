#python analyzecomplexity.py [-vlnbrR[fFg]wLB] complexFile origFile eyeFile eventFile [fwngramFile] [bwngramFile] [bodFile] [lexFile]
#v = verbose, l = lexicon, n = ngram calculation, b = frankbod'11 dataset, r = remove first and last word of each sentence,
#R = remove first and last word of each line
#f = first fixaton, F = first pass-first fixation, g = first pass fixation (gaze duration)
#w = whitelist everything (modulo other cli args) [leaves in words with multicaps and words containing punctuation]
#L = lag metrics, B = include B and dist complexity metrics

import sys
import re
import math
from model import Model, CondModel
from string import punctuation

#if len(sys.argv) < 5:
#  print("Incorrect args")
#  print("python analyzeComplexity.py [-vlnbr[fFg]LB] complexFile origFile eyeFile eventFile [ngramFile] [bodFile] [lexFile]")
#  raise Exception("Incorrect args")

if '-' == sys.argv[1][0]:
  OPTS = sys.argv[1][1:]
else:
  OPTS = ''

fixmetric = { #determines which fixation metric is used for the analysis
          'FirstPassFirstFix':False, #This only counts first pass fixations and only counts the duration of the first fixation
          'FirstPass':False, #This sums the duration after first fixation until the eye travels past that point
          'GoPast':False, #This sums the duration after first fixation until the eye travels outside this region
          'GoPastFirst':False, #This sums go past durations and first pass complexity
          'FirstFix':False} #This counts the first fixation of all words that were fixated upon

if 'g' in OPTS:
  fixmetric['GoPastFirst'] = True #activate a fixation metric
elif 'G' in OPTS:
  fixmetric['GoPastFirst'] = True
elif 'f' in OPTS:
  fixmetric['FirstFix'] = True #activate a fixation metric
elif 'F' in OPTS:
  fixmetric['FirstPassFirstFix'] = True #activate a fixation metric
else:
  fixmetric['FirstPass'] = True #activate FirstPass fixation by default

if 'n' in OPTS:
  NGRAMS = True #boolean denoting whether ngram results will be reported (slows down execution)
else:
  NGRAMS = False

if 'l' in OPTS:
  LEXICON = True #boolean denoting whether to develop separate lexicon metrics (unk thresholding, etc)
else:
  LEXICON = False

if 'v' in OPTS:
  VERBOSE = True #boolean denoting whether to output debugging information
else:
  VERBOSE = False

if 'r' in OPTS:
  #boolean denoting whether to rm first and last word of each sent
  RMFIRSTLASTOFSENT = True
else:
  RMFIRSTLASTOFSENT = False

if 'R' in OPTS:
  #boolean denoting whether to rm first and last word of each line
  RMLINEENDS = True
else:
  RMLINEENDS = False

if 'b' in OPTS:
  #boolean denoting whether to attempt to use the FrankBod'11 dataset or to calculate a dataset from scratch using their description
  FRANKBOD11 = True
else:
  FRANKBOD11 = False

if 'L' in OPTS:
  #boolean denoting whether to output lag metrics
  LAG = True
else:
  LAG = False

if 'B' in OPTS:
  #boolean denoting whether to output B and distance metrics (filler-gap, semantic roles)
  #only works with efabp
  BMET = True
else:
  BMET = False

if 'w' in OPTS:
  #a boolean denoting whether all terms should be whitelisted (modulo other commandline args)
  WHITEOUT = True
else:
  WHITEOUT = False

#running in efabp or no?
if OPTS == '':
  complexFname = sys.argv[1]
else:
  complexFname = sys.argv[2]

if re.search('efabp',complexFname) == None:
  EFABP = False
  BMET = False #override BMET since they don't exist
else:
  EFABP = True

if re.search('dundee',complexFname) != None:
  DUNDEE = True #Flag to look for quirks of Dundee
else:
  DUNDEE = False
  LAG = False #override lag metrics since they won't be calculated properly outside of eyetracked data

restart = re.compile('^start')
refinal = re.compile('\<FIN')
reskip = re.compile('[Tt]ime:')
multicap = re.compile('[A-Z].*[A-Z]')

if DUNDEE:
  try:
    if OPTS == '':
      SubjectLabel = re.search('\.(s[a-j])\.eyedata$',sys.argv[3]).group(1)
    else:
      SubjectLabel = re.search('\.(s[a-j])\.eyedata$',sys.argv[4]).group(1)
    CCOMPLEX = False #A separate flag from DUNDEE to allow future modifications to handle other eyetracked corpora
  except:
    CCOMPLEX = True
    LAG = False
else:
  CCOMPLEX = True
  LAG = False

#########################
#
# Definitions
#
#########################

def backConstruct(inlist):
  tmpoutlist = []
  outlist = []
  outstring = ''
  for i in range(len(inlist)-1,-1,-1):
    outstring = inlist[i]['word'].replace('\"','\'')+outstring
    outlist.append(outstring)
  return outlist

def unparse(incorp):
  # if a sentence failed to parse (notified at end of sentence),
  #  mark sentence as a parse failure
  for i in reversed(range(len(incorp)-1)):
    incorp[i]['parsed'] = False
    if incorp[i]['sentpos'] == 1:
      return True

#########################
#
# Load Ngrams
#
#########################
if NGRAMS:
  sys.stderr.write('Loading ngrams\n')

  if CCOMPLEX:
    ngramFile = open(sys.argv[4],'r',encoding='latin-1')
  else:
    ngramFile = open(sys.argv[6],'r',encoding='latin-1')

  unigrams = Model('U')
  w_giv_min_u = Model('UBF')
  w_giv_min = CondModel('BF')
  w_giv_min_bak = Model('UBBF')
  w_giv_plus_u = Model('UBB')
  w_giv_plus = CondModel('BB')
  w_giv_plus_bak = Model('UBBB')

  for ix, line in enumerate(ngramFile.readlines()):
    unigrams.read(line)
    w_giv_min.read(line)
    w_giv_plus.read(line)
    w_giv_min_bak.read(line)
    w_giv_plus_bak.read(line)
    if ix % 1000000 == 0:
      sys.stderr.write('  Loaded '+str(ix)+'\n')
  ngramFile.close()

#########################
#
# Load FrankBod'11 Dataset
#
#########################

whitelist = Model('WL')

if FRANKBOD11:
  sys.stderr.write('Loading FrankBod11 dataset\n')
  if NGRAMS:
    fbFile = open(sys.argv[7],'r',encoding='latin-1')
  else:
    fbFile = open(sys.argv[6],'r',encoding='latin-1')

  fbFile.readline()

  for line in fbFile.readlines():
    sline = [i for i in re.split('[, ]',line) if i != '']
    if SubjectLabel == 'sa' and sline[3] != 's1':
      continue
    elif SubjectLabel == 'sb' and sline[3] != 's2':
      continue
    elif SubjectLabel == 'sc' and sline[3] != 's3':
      continue
    elif SubjectLabel == 'sd' and sline[3] != 's4':
      continue
    elif SubjectLabel == 'se' and sline[3] != 's5':
      continue
    elif SubjectLabel == 'sf' and sline[3] != 's6':
      continue
    elif SubjectLabel == 'sg' and sline[3] != 's7':
      continue
    elif SubjectLabel == 'sh' and sline[3] != 's8':
      continue
    elif SubjectLabel == 'si' and sline[3] != 's9':
      continue
    elif SubjectLabel == 'sj' and sline[3] != 's10':
      continue

    whitelist[int(sline[4][1:])] = 1

  fbFile.close()

#########################
#
# Load Lexicon
#
#########################

lexicon = Model('L')
if LEXICON:
  sys.stderr.write('Loading lexicon\n')
  if CCOMPLEX:
    lexFile = open(sys.argv[5],'r',encoding='latin-1')
  else:
    lexFile = open(sys.argv[7],'r',encoding='latin-1')

  for line in lexFile.readlines():
    lexicon.read(line)
  lexFile.close()

#########################
#
# Load Original File
#
#########################

if DUNDEE and not CCOMPLEX:
  corpus = []

  sys.stderr.write('Loading original text\n')

  fnum = 0 #id of file being processed

  refilebreak = re.compile('^WORD ')

  if OPTS == '':
    origFile = open(sys.argv[2],'r',encoding='latin-1')
  else:
    origFile = open(sys.argv[3],'r',encoding='latin-1')

  prevwnum = 0 #id of previous word processed
  wnumoffset = 0

  for line in origFile.readlines():
    #beginning processing a new file
    if refilebreak.match(line) != None:
      fnum += 1
      wnumoffset += prevwnum
      continue
    sline = line.split()

    word = sline[0]
    #snum = int(sline[2])
    #lnum = int(sline[3])
    lw = int(sline[4]) #wnum relative to current line
    #wnum = sline[5] #wnum relative to current snum
    wnum = int(sline[12]) #wnum relative to current fnum

    if not FRANKBOD11:
      #blacklist first and last word in each line
      if RMLINEENDS and lw == 1: #blacklist first word in each line
        whitelist[wnum+wnumoffset] = 0.0
        if not (wnum == 1 and fnum == 1): #if word is not first word of first file, blacklist previous word (last of previous line)
          whitelist[wnum+wnumoffset-1] = 0.0
          corpus[wnum+wnumoffset-2]['inuse'] = False #corpus is one shorter than whitelist, so -2 is really -1
      elif not WHITEOUT and (multicap.search(word) != None or ( #throw out words with more than one capital letter
           [p for p in punctuation+'1234567890' if p in word] != [] )): #throw out words that contain a non-letter
        whitelist[wnum+wnumoffset] = 0.0
      else:
        whitelist[wnum+wnumoffset] = 1.0

    #build corpus from non-blacklisted, non-unk'ed words
    if whitelist[wnum+wnumoffset] != 0.0 and LEXICON and lexicon[word] != 0.0:
      corpus.append({'word':word,'inuse':True,'parsed':False})
    elif whitelist[wnum+wnumoffset] != 0.0 and not LEXICON:
      corpus.append({'word':word,'inuse':True,'parsed':False})
    else:
      corpus.append({'word':word,'inuse':False,'parsed':False})

    prevwnum = wnum

  origFile.close()

  if not FRANKBOD11 and (RMLINEENDS or RMFIRSTLASTOFSENT):
    corpus[-1]['inuse'] = False #blacklist final word
else: #build ccomplex
  whitelist = True
  corpus = []

  sys.stderr.write('Loading original text\n')

  fnum = 0 #id of file being processed

  refilebreak = re.compile('^WORD ')

  if OPTS == '':
    origFile = open(sys.argv[2],'r',encoding='latin-1')
  else:
    origFile = open(sys.argv[3],'r',encoding='latin-1')

  for line in origFile.readlines():
    #beginning processing a new file
    if refilebreak.match(line) != None:
      fnum += 1
      continue
    for i,w in enumerate(line.split()):
      if not WHITEOUT and (multicap.search(w) != None or ( #throw out words with more than one capital letter
           [p for p in punctuation+'1234567890' if p in w] != [] )): #throw out words that contain a non-letter
        whitelist = False
      else:
        whitelist = True

      #build corpus from non-blacklisted, non-unk'ed words
      if whitelist and LEXICON and lexicon[w] != 0.0:
        corpus.append({'word':w,'file':fnum,'counts':True,'inuse':True,'parsed':False})
      elif whitelist and not LEXICON:
        corpus.append({'word':w,'file':fnum,'counts':True,'inuse':True,'parsed':False})
      else:
        corpus.append({'word':w,'file':fnum,'counts':True,'inuse':False,'parsed':False})
  origFile.close()

#########################
#
# Load Complexity
#
#########################

#To add new complexity metrics:
# Make src/parser-x-MODEL.cpp output them,
# Add them to meas,
# Add to measmap appropriate mappings from model output to complexity nicks,
# Finally, if the final metric output by the model changes, update stopmetric with the new metric nick

sys.stderr.write('Loading complexity measures\n')

complexFile = open(complexFname,'r',encoding='latin-1')

#drill to start of output
while True:
  line = complexFile.readline()
  match = restart.match(line)
  if match != None:
    break

word = ''

measmap = { 'Total Surprisal':'totsurp',\
    'Lexical Surprisal':'lexsurp',\
    'Syntactic Surprisal':'synsurp',\
    'Entropy Reduction':'entred',\
    'Embedding Depth':'embedep',\
    'Embedding Difference':'embedif',\
    'F-L- Cost':'fmlm',\
    'F+L- Cost':'fplm',\
    'F-L+ Cost':'fmlp',\
    'F+L+ Cost':'fplp',\
    'Depth 1 F+L+ Cost':'d1fplp',\
    'Depth 1 F+L- Cost':'d1fplm',\
    'Depth 1 F-L+ Cost':'d1fmlp',\
    'Depth 1 F-L- Cost':'d1fmlm',\
    'Depth 2 F+L+ Cost':'d2fplp',\
    'Depth 2 F+L- Cost':'d2fplm',\
    'Depth 2 F-L+ Cost':'d2fmlp',\
    'Depth 2 F-L- Cost':'d2fmlm',\
    'Depth 3 F+L+ Cost':'d3fplp',\
    'Depth 3 F+L- Cost':'d3fplm',\
    'Depth 3 F-L+ Cost':'d3fmlp',\
    'Depth 3 F-L- Cost':'d3fmlm',\
    'Depth 4 F+L+ Cost':'d4fplp',\
    'Depth 4 F+L- Cost':'d4fplm',\
    'Depth 4 F-L+ Cost':'d4fmlp',\
    'Depth 4 F-L- Cost':'d4fmlm',\
    'Distance F-L+ Cost':'Dfmlp',\
    'B Add':'badd',\
    'B+ Cost':'bp',\
    'B Storage':'bsto',\
    'B CDR':'bcdr',\
    'B- Cost':'bm',\
    'Depth B+ Cost':'dbp',\
    'Depth B- Cost':'dbm',\
    'Distance B+ Cost':'Dbp',\
    'F-L-Badd':'fmlmba',\
    'F-L-Bcdr':'fmlmbc',\
    'F-L-BNil':'fmlmbo',\
    'F-L-B+':'fmlmbp',\
    'F-L+Badd':'fmlpba',\
    'F-L+Bcdr':'fmlpbc',\
    'F-L+BNil':'fmlpbo',\
    'F-L+B+':'fmlpbp',\
    'F+L-Badd':'fplmba',\
    'F+L-Bcdr':'fplmbc',\
    'F+L-BNil':'fplmbo',\
    'F+L-B+':'fplmbp',\
    'F+L+Badd':'fplpba',\
    'F+L+Bcdr':'fplpbc',\
    'F+L+BNil':'fplpbo',\
    'F+L+B+':'fplpbp',\
}

meas = {'totsurp':0.0,\
    'lexsurp':0.0,\
    'synsurp':0.0,\
    'entred':0.0,\
    'embedep':0.0,\
    'embedif':0.0,\
    'badd':0.0,\
    'bp':0.0,\
    'bsto':0.0,\
    'bcdr':0.0,\
    'bm':0.0,\
    'dbp':0.0,\
    'dbm':0.0,\
    'Dbp':0.0,\
    'fmlm':0.0,\
    'fplm':0.0,\
    'fmlp':0.0,\
    'fplp':0.0,\
    'd1fplp':0.0,\
    'd1fplm':0.0,\
    'd1fmlp':0.0,\
    'd1fmlm':0.0,\
    'd2fplp':0.0,\
    'd2fplm':0.0,\
    'd2fmlp':0.0,\
    'd2fmlm':0.0,\
    'd3fplp':0.0,\
    'd3fplm':0.0,\
    'd3fmlp':0.0,\
    'd3fmlm':0.0,\
    'd4fplp':0.0,\
    'd4fplm':0.0,\
    'd4fmlp':0.0,\
    'd4fmlm':0.0,\
    'Dfmlp':0.0,\
    'fmlmba':0.0,\
    'fmlmbc':0.0,\
    'fmlmbo':0.0,\
    'fmlmbp':0.0,\
    'fmlpba':0.0,\
    'fmlpbc':0.0,\
    'fmlpbo':0.0,\
    'fmlpbp':0.0,\
    'fplmba':0.0,\
    'fplmbc':0.0,\
    'fplmbo':0.0,\
    'fplmbp':0.0,\
    'fplpba':0.0,\
    'fplpbc':0.0,\
    'fplpbo':0.0,\
    'fplpbp':0.0,\
    'wdelta':1.0,\
    'parsed':True\
}

finalIx = len(corpus)-1
ixmod = 0

wnum = 0
cix = 0
wix = 0
midword = False
sentpos = 0

if EFABP:
  stopmetric = 'fplpbp'
else:
  stopmetric = 'd4fplp'

#iterate over complexity output until completion
complexArray = complexFile.readlines()
for lineix in range(0,len(complexArray)-2):
  line = complexArray[lineix]
  if refinal.search(line) != None:
    sentpos = 0
    ixmod += 1
    continue
  elif reskip.search(line) != None:
    try:
      if line.split()[-2] == '0':
        unparse(corpus)
    except:
      pass
    ixmod += 1
    continue

  try:
    if DUNDEE and corpus[wnum]['word'] == '.':
      #Fix for Dundee error where there's a lone extra '.' after a sentence (right before 'purists').
      corpus[wnum]['inuse'] = False
      wnum += 1
  except:
    sys.stderr.write(str(corpus[-1])+'\n')
    sys.stderr.write('Line: '+str(line)+'\n')
    sys.stderr.write('wnum '+str(wnum)+'/'+str(len(corpus))+'\n')
    raise

  if ':' == line[0] or len(line.split(':')) == 1 or len(line.strip().split(':')[1].split()) == 3: #next word chunk
    word += line.split()[-2]
    continue
  else:
    if line.split(':')[0] not in measmap.keys(): #ignore things you don't understand
      continue
    if midword:
      meas[measmap[line.split(':')[0]]] += float(line.split()[-1])
    else:
      meas[measmap[line.split(':')[0]]] = float(line.split()[-1])

  #stopping condition
  if measmap[line.split(':')[0]] == stopmetric:
    if cix + len(word) == wix + len(corpus[wnum]['word']): #finished with word
      sentpos += 1
      meas['sentpos'] = sentpos
      meas['nrchar'] = len(word)

      if NGRAMS:
        #unigrams
        meas['uprob'] = unigrams[corpus[wnum]['word']]
        #using srilm #(logprobs are natural logs)
        #fwbigrams
        if sentpos == 1:
          if w_giv_min['<s>'][corpus[wnum]['word']] == 0.0:
            if w_giv_min_u[corpus[wnum]['word']]*w_giv_min_bak['<s>'] == 0.0:
              meas['fwprob'] = math.log(10**-200)
            else:
              meas['fwprob'] = math.log(w_giv_min_u[corpus[wnum]['word']]*w_giv_min_bak['<s>'])
          else:
            meas['fwprob'] = math.log(w_giv_min['<s>'][corpus[wnum]['word']])
        elif w_giv_min[corpus[wnum-1]['word']][corpus[wnum]['word']] == 0.0:
          if w_giv_min_u[corpus[wnum]['word']]*w_giv_min_bak[corpus[wnum-1]['word']] == 0.0:
            meas['fwprob'] = math.log(10**-200)
          else:
            meas['fwprob'] = math.log(w_giv_min_u[corpus[wnum]['word']]*w_giv_min_bak[corpus[wnum-1]['word']])
        else:
          meas['fwprob'] = math.log(w_giv_min[corpus[wnum-1]['word']][corpus[wnum]['word']])

        #bwbigrams
        if (wnum != 0 and sentpos == 1):
          if w_giv_plus['<s>'][corpus[wnum-1]['word']] == 0.0: #<s> rather than </s> because w_giv_plus is calculated from a reversed corpus
            if w_giv_plus_u[corpus[wnum-1]['word']]*w_giv_plus_bak['<s>'] == 0.0:
              corpus[wnum-1].update({'bwprob':math.log(10**-200)})
            else:
              corpus[wnum-1].update({'bwprob':math.log(w_giv_plus_u[corpus[wnum-1]['word']]*w_giv_plus_bak['<s>'])})
          else:
            corpus[wnum-1].update({'bwprob':math.log(w_giv_plus['<s>'][corpus[wnum-1]['word']])})
        elif wnum != 0 and w_giv_plus[corpus[wnum]['word']][corpus[wnum-1]['word']] == 0.0:
          if w_giv_plus_u[corpus[wnum-1]['word']]*w_giv_plus_bak[corpus[wnum]['word']] == 0.0:
            corpus[wnum-1].update({'bwprob':math.log(10**-200)})
          else:
            corpus[wnum-1].update({'bwprob':math.log(w_giv_plus_u[corpus[wnum-1]['word']]*w_giv_plus_bak[corpus[wnum]['word']])})
        elif wnum != 0:
          corpus[wnum-1].update({'bwprob':math.log(w_giv_plus[corpus[wnum]['word']][corpus[wnum-1]['word']])})
        elif wnum == finalIx: #last item in corpus
          if w_giv_plus['<s>'][corpus[wnum]['word']] == 0.0: #<s> rather than </s> because w_giv_plus is calculated from a reversed corpus
            if w_giv_plus_u[corpus[wnum]['word']]*w_giv_plus_bak['<s>'] == 0.0:
              meas['bwprob'] = math.log(10**-200)
            else:
              meas['bwprob'] = math.log(w_giv_plus_u[corpus[wnum]['word']]*w_giv_plus_bak['<s>'])
          else:
            meas['bwprob'] = math.log(w_giv_plus['<s>'][corpus[wnum]['word']])
      corpus[wnum].update(meas)

      cix += len(word)
      word = ''
      wix += len(corpus[wnum]['word'])
      wnum += 1
      midword = False
    else:
      midword = True

complexFile.close()

#########################
#
# Find missing events (in eyefile)
#
#########################

if DUNDEE and not CCOMPLEX:
  sys.stderr.write('Loading eyetracked text (once)\n')

  fnum = 0 #id of current file being processed

  prevfx = 0 #id of previous fixation event
  prevwnum = [0,0] #(id of previous word fixated, id of previous fixated word)

  possiblefx = Model("PF") #stores the possible fixation ids for each file (some are buried in nests of blinking and so are not first pass)
  maxfx = Model("MF") #stores the maximum fixation event id for each file
  gazetimes = Model("GT")
  gazewords = Model("GW")

  if OPTS == '':
    eyeFile = open(sys.argv[3],'r',encoding='latin-1')
  else:
    eyeFile = open(sys.argv[4],'r',encoding='latin-1')

  for line in eyeFile.readlines():
    #restart file count
    if refilebreak.match(line) != None:
      maxfx[fnum] = prevfx
      fnum += 1
      prevwnum = [-1,-1]
      continue

    sline = line.split()
    word = sline[0]
    snum = int(sline[1])
    lnum = int(sline[2])
    olen = int(sline[3]) #length of word (including punc) #NB: this is 0 if no fixation occurred!
    wlen = int(sline[4])
    xpos = sline[5]
    wnum = int(sline[6]) #wnum relative to fnum
    fdur = sline[7]
    oblp = sline[8]
    wdlp = sline[9]
    fxno = int(sline[10])
    txfr = int(sline[11])

    gazetimes[fnum,fxno] = max(0.0,float(fdur)) #Store all fixation durations for calculating gaze duration
    gazewords[fnum,fxno] = wnum

    if wnum <= prevwnum[1]: #Skips blinks and leads only first fixations of a word to be considered
      continue

    if fxno != 0:
      possiblefx[str( (fnum,fxno) )] = 1.0
      prevfx = fxno
      prevwnum[1] = wnum
    prevwnum[0] = wnum

  maxfx[fnum] = prevfx
  eyeFile.close()

  #########################
  #
  # Queue up Cumulative Metrics
  #
  #########################

  if BMET:
    cummetrics = ['totsurp','synsurp','lexsurp','entred','embedep','embedif',\
                 'badd','bcdr','bp','bsto','bm','dbp','dbm','Dbp','wdelta',\
                 'fmlm','fplm','fmlp','fplp',\
                 'fmlmba','fmlmbc','fmlmbo','fmlmbp','fmlpba','fmlpbc','fmlpbo','fmlpbp',\
                 'fplmba','fplmbc','fplmbo','fplmbp','fplpba','fplpbc','fplpbo','fplpbp',\
                 'd1fplp','d1fplm','d1fmlp','d1fmlm',\
                 'd2fplp','d2fplm','d2fmlp','d2fmlm',\
                 'd3fplp','d3fplm','d3fmlp','d3fmlm',\
                 'd4fplp','d4fplm','d4fmlp','d4fmlm',\
                 'Dfmlp']
  else:
    cummetrics = ['totsurp','synsurp','lexsurp','entred','embedep','embedif','wdelta',\
                 'fmlm','fplm','fmlp','fplp',\
                 'd1fplp','d1fplm','d1fmlp','d1fmlm',\
                 'd2fplp','d2fplm','d2fmlp','d2fmlm',\
                 'd3fplp','d3fplm','d3fmlp','d3fmlm',\
                 'd4fplp','d4fplm','d4fmlp','d4fmlm']

  if NGRAMS:
    cummetrics += ['uprob','fwprob','bwprob']

  #########################
  #
  # Load Eyetracking
  #
  #########################

  sys.stderr.write('Loading eyetracked text (twice)\n')

  meas.update({'sentpos':0,'totsurp':1.0,'lexsurp':0.0,'synsurp':1.0,'entred':0.0,\
               'embedep':0.0,'embedif':0.0,'nrchar':1,'locfreq':1.0,'wdelta':0.0,'fdur':0.0})

  fnum = 0 #id of current file being processed

  if OPTS == '':
    eyeFile = open(sys.argv[3],'r',encoding='latin-1')
  else:
    eyeFile = open(sys.argv[4],'r',encoding='latin-1')

  burned = Model('B') #stores each fxno that has been traversed
  searching = 1 #the fxno currently being sought out

  wnumoffset = -1 #offset to cat all files together (to handle FrankBod11 numbering)
  prevwnum = [0,0] #(id of previous word, id of previous fixated word)
  sought = [0,0] #(id of previously found sought fixation event, id of word for that fixation event)
  sook = 0

  offsets = Model('O')

  #iterate over eyetracking output until completion
  for line in eyeFile.readlines():
    #restart file count
    if refilebreak.match(line) != None:
      offsets[fnum] = prevwnum[0]
      wnumoffset += prevwnum[0]
      fnum += 1
      prevwnum = [-1,-1]
      burned.clear()
      searching = 1
      corpus[sought[1]]['nextfdur'] = 0.0
      sought = [0,0]
      continue
    sline = line.split()
    word = sline[0]
    snum = int(sline[1])
    lnum = int(sline[2])
    olen = int(sline[3]) #length of word (including punc) #NB: this is 0 if no fixation occurred!
    wlen = int(sline[4])
    xpos = sline[5]
    wnum = int(sline[6]) #wnum relative to fnum
    fdur = sline[7]
    oblp = sline[8]
    wdlp = sline[9]
    fxno = int(sline[10])
    txfr = int(sline[11]) #local text frequency

    burned[fxno] = True

    if wnum <= prevwnum[1]: #Skips blinks and leads only first fixations of a word to be considered
      continue

    corpus[wnum+wnumoffset]['fxno'] = fxno
    corpus[wnum+wnumoffset]['locfreq'] = txfr

    if fixmetric['FirstFix']:
      corpus[wnum+wnumoffset]['fdur'] = float(fdur)
      corpus[wnum+wnumoffset]['prevfdur'] = 0.0
      corpus[wnum+wnumoffset]['nextfdur'] = 0.0

    #if we have passed the fixation we were searching for, store the fixation (in case we are currently on it) and look for the next
    swap = True
    while burned[searching] != 0.0 or (possiblefx[str( (fnum,searching) )] == 0.0 and searching <= maxfx[fnum]):
      if swap:
        sook = searching
        swap = False
      searching += 1

    if fxno == sook:
      if fixmetric['FirstPassFirstFix']:
        corpus[wnum+wnumoffset]['fdur'] = float(fdur)
        corpus[wnum+wnumoffset]['prevfdur'] = 0.0
        corpus[wnum+wnumoffset]['nextfdur'] = 0.0
      elif fixmetric['GoPastFirst']:
        corpus[wnum+wnumoffset]['fdur'] = float(fdur)+sum(gazetimes[fnum,fx] for fx in range(fxno,min(searching,maxfx[fnum])) if corpus[int(gazewords[fnum,fx])+wnumoffset]['inuse'] and corpus[int(gazewords[fnum,fx])+wnumoffset]['parsed'])
        for metric in cummetrics:
          for wix in range(sought[1],wnum+wnumoffset+1):
            if corpus[wix]['parsed'] and corpus[wix]['inuse']:
              try:
                corpus[wnum+wnumoffset]['cum'+metric] += corpus[wix][metric]
              except:
                corpus[wnum+wnumoffset]['cum'+metric] = corpus[wix][metric]
        corpus[wnum+wnumoffset]['prevfdur'] = 0.0
        corpus[wnum+wnumoffset]['nextfdur'] = 0.0
      elif fixmetric['FirstPass']:
        corpus[wnum+wnumoffset]['fdur'] = float(fdur)
        for fx in range(fxno,min(searching,maxfx[fnum])):
          if gazewords[fnum,fx] < sought[1] or gazewords[fnum,fx] > wnum:
            break
          corpus[wnum+wnumoffset]['fdur'] += float(gazetimes[fnum,fx])
        for metric in cummetrics:
          for wix in range(sought[1],wnum+wnumoffset+1):
            if corpus[wix]['parsed'] and corpus[wix]['inuse']:
              try:
                corpus[wnum+wnumoffset]['cum'+metric] += corpus[wix][metric]
              except:
                corpus[wnum+wnumoffset]['cum'+metric] = corpus[wix][metric]
        corpus[wnum+wnumoffset]['prevfdur'] = float(fdur)+sum(gazetimes[fnum,fx] for fx in range(sought[0],fxno))
        if sought[1] != 0:
          corpus[sought[1]]['nextfdur'] = float(fdur)+sum(gazetimes[fnum,fx] for fx in range(fxno,min(searching,maxfx[fnum])))
      sought = [fxno,wnum+wnumoffset]
    
      #current word is fixated
      corpus[wnum+wnumoffset]['counts'] = True

      if wnum+wnumoffset != 0: #Not first word of corpus
        if corpus[wnum+wnumoffset-1]['counts']: #prev word was fixated
          corpus[wnum+wnumoffset]['previsfix'] = int(True)
          corpus[wnum+wnumoffset-1]['nextisfix'] = int(True)
        else: #prev word wasn't fixated immediately previous
          corpus[wnum+wnumoffset]['previsfix'] = int(False)
          corpus[wnum+wnumoffset-1]['nextisfix'] = int(False) #Correct in the temporal sense rather than the sequential sense
    else: #NB: Note that nextisfix and previsfix are only calculated for first pass fixations regardless of if FirstFix is being used
      if not fixmetric['FirstFix']:
        corpus[wnum+wnumoffset]['fdur'] = 0.0 #force first pass metrics to discount all non-first pass fixations
        corpus[wnum+wnumoffset]['prevfdur'] = 0.0
        corpus[wnum+wnumoffset]['nextfdur'] = 0.0

      try:
        if corpus[wnum+wnumoffset]['counts'] == True:
          pass
      except:
        corpus[wnum+wnumoffset]['counts'] = False
      corpus[wnum+wnumoffset-1]['nextisfix'] = int(False)
      corpus[wnum+wnumoffset]['previsfix'] = int(False)

    if DUNDEE and word == '.':
      #Fix for Dundee error where there's a lone extra '.' after a sentence (right before 'purists').
      corpus[wnum+wnumoffset].update(meas)

    if fxno != 0:
      prevwnum[1] = wnum
    prevwnum[0] = wnum

  corpus[sought[1]]['nextfdur'] = 0.0
  corpus[0]['previsfix'] = int(False)
  corpus[-1]['nextisfix'] = int(False)

  eyeFile.close()

  #########################
  #
  # Load Event Data
  #
  #########################

  sys.stderr.write('Loading event data\n')

  fnum = 0 #id of current file being processed

  if OPTS == '':
    eventFile = open(sys.argv[4],'r',encoding='latin-1')
  else:
    eventFile = open(sys.argv[5],'r',encoding='latin-1')

  burned.clear() #stores each wnum that has been traversed

  wnumoffset = -1 #offset to cat all files together (to handle FrankBod11 numbering)

  #iterate over eyetracking output until completion
  for line in eventFile.readlines():
    #restart file count
    if refilebreak.match(line) != None:
      wnumoffset += offsets[fnum]
      fnum += 1
      burned.clear()
      continue
    sline = line.split()
    word = sline[0]
    snum = int(sline[1])
    lnum = int(sline[2])
    olen = int(sline[3]) #length of word (including punc) #NB: this is 0 if no fixation occurred!
    wlen = int(sline[4])
    xpos = sline[5]
    wnum = int(sline[6]) #wnum relative to fnum
    fdur = sline[7]
    oblp = sline[8]
    wdlp = int(sline[9])
    laun = int(sline[10])
    txfr = sline[11]

    if lnum <= 0.0 or burned[wnum] != 0.0: #Blink and non-first fixation
      continue
    burned[wnum] = True

    corpus[wnum+wnumoffset]['laundist'] = laun
    corpus[wnum+wnumoffset]['landpos'] = wdlp

    if DUNDEE and word == '.':
      #Fix for Dundee error where there's a lone extra '.' after a sentence (right before 'purists').
      corpus[wnum+wnumoffset].update({'laundist':0,'landpos':1,'inuse':False})

  eventFile.close()
else:
  #not using eye tracking data
  cummetrics = []

#########################
#
# Output Compiled Data
#
#########################

#NB: Excludes non-fixated data
bakfreq = 0

if LAG:
  baselagmetrics = ['sentpos','nrchar','previsfix','nextisfix','laundist','landpos','locfreq','bakfreq']
  lagmetrics = {}
  for m in baselagmetrics: #add non-cummetrics to lagmetrics
    lagmetrics[m] = 0
  lagmetrics['noncumuprob'] = 0.0
  lagmetrics['noncumfwprob'] = 0.0
  lagmetrics['noncumbwprob'] = 0.0
  lagmetrics['noncumtotsurp'] = 0.0
  lagmetrics['noncumembedep'] = 0.0
  for m in cummetrics:
    lagmetrics[m] = 0 #add cummetrics to lagmetrics

sys.stderr.write('Writing output\n')

for i,w in enumerate(corpus):
  #sys.stdout.write(str(w)+'\n') #!!!
  if w['inuse'] and w['counts'] and w['parsed']:
    repix = i
    break

if DUNDEE and not CCOMPLEX:
  sys.stdout.write('subject ')
sys.stdout.write('word filter notfix sentpos nrchar ')
if DUNDEE and not CCOMPLEX:
  sys.stdout.write('locfreq bakfreq fdur ')

badmetrics = ['subject','word','filter','notfix','sentpos','nrchar','locfreq','bakfreq','parsed','counts','inuse','fdur','fxno']

cumpatt = re.compile('cum')
if BMET:
  metrics = [k for k in corpus[repix].keys() if k not in badmetrics and cumpatt.match(k) == None ]
else:
  bpatt = re.compile('b[aocspm]')
  metrics = [k for k in corpus[repix].keys() if k not in badmetrics and \
                                                not (bpatt.search(k) != None or 'D' in k or cumpatt.match(k) != None)]
  

noncummetrics = ['totsurp','embedep','uprob','fwprob','bwprob']

for m in metrics:
  sys.stdout.write(m+' ')
for m in cummetrics:
  sys.stdout.write('cum'+m+' ')
#if NGRAMS:
#  sys.stdout.write('logwordprob prevlogprob logforwprob logbackprob ')

if LAG:
  for m in baselagmetrics:
    sys.stdout.write('lag'+m+' ')
  for m in noncummetrics:
    sys.stdout.write('lag'+m+' ')
  for m in cummetrics:
    sys.stdout.write('lagcum'+m+' ')
#  if NGRAMS:
#    sys.stdout.write('laglogwordprob laglogforwprob laglogbackprob ')

sys.stdout.write('\n')

for i,w in enumerate(corpus):
    blackout = False
    notfix = False

    outstr = ''
    if DUNDEE and not CCOMPLEX:
      outstr = outstr + str(SubjectLabel)+' '
    outstr = outstr + w['word']+' '

    if w['inuse'] == False:
      if VERBOSE:
        outstr = w['word']+' '
        if DUNDEE and not CCOMPLEX:
          outstr = str(SubjectLabel)+' '+outstr+'('+str(w['fxno'])+') '+str(w['previsfix'])+' '+str(w['nextisfix'])+' '
        if LEXICON and lexicon[w['word']] == 0.0:
          outstr = outstr + '[Lex Filtered]'+' '
        outstr = outstr +'\n'
        sys.stdout.write(outstr)
        continue
      else: 
        blackout = True
    if w['parsed'] == False:
      #if a sentence wasn't parsed, don't try to process it
      continue
    if not w['counts']:
      #if a word wasn't fixated, don't bother outputting it
      notfix = True
    if RMFIRSTLASTOFSENT and (w['sentpos'] == 1 or (i < finalIx and corpus[i+1]['sentpos'] == 1) or i == finalIx):
      #ignore first/last words of sents
      blackout = True

    if not CCOMPLEX:
      if i > 0:
        bakfreq = corpus[i-1]['locfreq']
      else:
        bakfreq = 0.0
      w['bakfreq'] = bakfreq

    outstr = outstr + str(int(blackout))+' '
    outstr = outstr + str(int(notfix))+' '
    if VERBOSE and not CCOMPLEX:
      outstr = outstr + '('+str(w['fxno'])+') '
    outstr = outstr + str(w['sentpos'])+' '
    outstr = outstr + str(w['nrchar'])+' '

    if not notfix:
      if not CCOMPLEX:
        outstr = outstr + str(w['locfreq'])+' '
        outstr = outstr + str(w['bakfreq'])+' '
        outstr = outstr + str(w['fdur'])+' '

      if not blackout:
        for m in metrics:
          outstr = outstr + str(w[m])+' '
        for m in cummetrics:
          outstr = outstr + str(w['cum'+m])+' '
#        if NGRAMS:
#          outstr = outstr + str(w['uprob']) + ' '
#          outstr = outstr + str(w['pprob']) + ' '
#          outstr = outstr + str(w['fwprob']) + ' '
#          outstr = outstr + str(w['bwprob']) + ' '

        if LAG:
          #output lag metrics
          for m in baselagmetrics:
            outstr = outstr + str(lagmetrics[m])+' '
          for m in noncummetrics:
            outstr = outstr + str(lagmetrics['noncum'+m])+' '
          for m in cummetrics:
            outstr = outstr + str(lagmetrics[m])+' '
#          if NGRAMS:
#            outstr = outstr + str(lagmetrics['logprob'])+' '
#            outstr = outstr + str(lagmetrics['logforwprob'])+' '
#            outstr = outstr + str(lagmetrics['logbackprob'])+' '

      #update lagmetrics
      if LAG:
        if blackout:
          for metric in lagmetrics:
            lagmetrics[metric] = 0.0
        else:
          for metric in cummetrics:
            lagmetrics[metric] = w['cum'+metric]
          if NGRAMS:
            lagmetrics['noncumuprob'] = w['uprob']
            lagmetrics['noncumfwprob'] = w['fwprob']
            lagmetrics['noncumbwprob'] = w['bwprob']
          lagmetrics['noncumtotsurp'] = w['totsurp']
          lagmetrics['noncumembedep'] = w['embedep']
          for m in baselagmetrics:
            lagmetrics[m] = w[m]

    outstr = outstr +'\n'
    sys.stdout.write(outstr)
