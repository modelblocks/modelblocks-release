#python calcngramprobtoks.py ngramFile textFile 

from __future__ import division
import math
import sys
sys.path.insert(1, '../resource-gcg/scripts/') #grant access to model.py 
from model import Model,CondModel

ngramfileix = 1
textfileix = 2

endofsentpunc = ['.','!','?']

#########################
#
# Definitions
#
#########################

def chainruleprobs(word):
  #decompose ngram probabilities via chain-rule
  fwprob = 0
  bwprob = 0

  #Chain-rule unigram probability
  rootend = 0
  noroot = True
  for cix in range(len(word)):
    if noroot and word[cix] not in endofsentpunc+['\'','"']:
      #find the end of the actual (tokenized) word
      rootend = cix
      continue
    if noroot:
      #first non-root character: we'll catch this below
      noroot = False
      continue
    #Throw in the chain-rule probability of each bit of punctuation followed by the next
    if w_giv_min[word[cix-1]][word[cix]] == 0.0:
      if w_giv_min_u[word[cix]]*w_giv_min_bak[word[cix-1]] == 0.0:
        fwprob += math.log(10**-200)
      else:
        fwprob += math.log(w_giv_min_u[word[cix]]*w_giv_min_bak[word[cix-1]])
    else:
      fwprob += math.log(w_giv_min[word[cix-1]][word[cix]])
    #Throw in the chain-rule probability of each bit of punctuation preceding the other
    if w_giv_plus[word[cix]][word[cix-1]] == 0.0:
      if w_giv_plus_u[word[cix-1]]*w_giv_plus_bak[word[cix]] == 0.0:
        bwprob += math.log(10**-200)
      else:
        bwprob += math.log(w_giv_plus_u[word[cix-1]]*w_giv_plus_bak[word[cix]])
    else:
      bwprob += math.log(w_giv_plus[word[cix]][word[cix-1]])

  #Throw in the chain-rule probability of the root followed by the first bit of punctuation
  if w_giv_min[word[:rootend+1]][word[rootend+1]] == 0.0:
    if w_giv_min_u[word[rootend+1]]*w_giv_min_bak[word[:rootend+1]] == 0.0:
      fwprob += math.log(10**-200)
    else:
      fwprob += math.log(w_giv_min_u[word[rootend+1]]*w_giv_min_bak[word[:rootend+1]])
  else:
    fwprob += math.log(w_giv_min[word[:rootend+1]][word[rootend+1]])
  #Throw in the chain-rule probability of the first bit of punctuation preceded by the root
  if w_giv_plus[word[rootend+1]][word[:rootend+1]] == 0.0:
    if w_giv_plus_u[word[:rootend+1]]*w_giv_plus_bak[word[rootend+1]] == 0.0:
      bwprob += math.log(10**-200)
    else:
      bwprob += math.log(w_giv_plus_u[word[:rootend+1]]*w_giv_plus_bak[word[rootend+1]])
  else:
    bwprob += math.log(w_giv_plus[word[rootend+1]][word[:rootend+1]])
  return ((fwprob,bwprob), word[:rootend+1])

def chainruleit(word):
  #Should we use chain-rule decomposition on this word?
  if len(word) > 1 and word != '...' and word[-1] in endofsentpunc+['\'','"']:
    return(True)
  return(False)

#########################
#
# Load Ngrams File
#
#########################

sys.stderr.write('Loading ngrams\n')

with open(sys.argv[ngramfileix],'r',encoding='latin-1') as ngramFile:
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


#########################
#
# Compute ngrams for corpus 
#
#########################

corpus = []

with open(sys.argv[textfileix],'r',encoding='latin-1') as textFile:
  lines = textFile.readlines()
  prevword = ''
  word = ''
  nextword = ''
  SENTBEGIN = True
  SENTEND = False
  sentpos = 0
  entry = {}
  for wordix in range(1,len(lines)): #skip first line because it's a file break
    #for each word in the corpus
    newword = lines[wordix].strip().split()[0]

    if newword == 'WORD':
      #file break
      continue
    elif wordix == len(lines)-1:
      #end of corpus
      SENTEND = True
    else:
      #test for begin and end of sentence
      if SENTEND:
        #if a sentence just ended, we're starting a new sentence
        SENTBEGIN = True
        SENTEND = False
        sentpos = 0
      if newword != '...' and (newword[-1] in endofsentpunc or (len(newword) > 1 and newword[-1] in ["\'",'\"'] and newword[-2] in endofsentpunc)):
        #if a word isn't an elided quote and ends with punctuation, it must signify the end of a sentence
        SENTEND = True 

    if SENTBEGIN:
      #if word is start of a new sentence
      prevword = '<s>'
      word = newword
      if SENTEND:
        #if word is also end of a new sentence (really just the lone '.')
        nextword = '</s>'
        bnextword = '<s>' #because bwbigrams are calculated by reversing the corpus
      else:
        nextword = lines[wordix+1].split()[0]
        bnextword = nextword
    elif SENTEND:
      #if word is the end of a sentence
      prevword = word
      word = newword
      nextword = '</s>'
      bnextword = '<s>'
    else:
      #word is simply somewhere in the middle of a sentence
      prevword = word
      word = newword
      nextword = lines[wordix+1].split()[0]
      bnextword = nextword


    #using srilm #(logprobs are natural logs)
    if chainruleit(word):
      #need to tokenize the word because ngrams were calculated over a different tokenization

      (entry['fwprob'],entry['bwprob']),wordroot = chainruleprobs(word)
      rootend = len(wordroot)-1

      entry['uprob'] = entry['fwprob']

      #Throw in the unigram probability of the tokenized word
      entry['uprob'] += math.log(max(unigrams[word[:rootend+1]],10**-200))

      #Pre-tokenized forward bigram probability
      if chainruleit(prevword):
        #include chain-rule probs of prevword if necessary
        (crfwprevprob,crbwprevprob),prevwordroot = chainruleprobs(prevword)
        if w_giv_min[prevword[-1]][word[:rootend+1]] == 0.0:
          if w_giv_min_u[word[:rootend+1]]*w_giv_min_bak[prevword[-1]] == 0.0:
            entry['fwprob'] += math.log(10**-200) + crfwprevprob
          else:
            entry['fwprob'] += math.log(w_giv_min_u[word[:rootend+1]]*w_giv_min_bak[prevword[-1]]) + crfwprevprob
        else:
          entry['fwprob'] += math.log(w_giv_min[prevword[-1]][word[:rootend+1]]) + crfwprevprob
      else:
        if w_giv_min[prevword][word[:rootend+1]] == 0.0:
          if w_giv_min_u[word[:rootend+1]]*w_giv_min_bak[prevword] == 0.0:
            entry['fwprob'] += math.log(10**-200)
          else:
            entry['fwprob'] += math.log(w_giv_min_u[word[:rootend+1]]*w_giv_min_bak[prevword])
        else:
          entry['fwprob'] += math.log(w_giv_min[prevword][word[:rootend+1]])

      #Pre-tokenized backward bigram probability
      if chainruleit(bnextword):
        (crfwbnextprob,crbwbnextprob),bnextwordroot = chainruleprobs(bnextword)
        #include chain-rule probs of bnextword if necessary
        if w_giv_plus[bnextwordroot][word[-1]] == 0.0:
          if w_giv_plus_u[word[-1]]*w_giv_plus_bak[bnextwordroot] == 0.0:
            entry['bwprob'] += math.log(10**-200) + crbwbnextprob
          else:
            entry['bwprob'] += math.log(w_giv_plus_u[word[-1]]*w_giv_plus_bak[bnextwordroot]) + crbwbnextprob
        else:
          entry['bwprob'] += math.log(w_giv_plus[bnextwordroot][word[-1]]) + crbwbnextprob
      else:
        if w_giv_plus[bnextword][word[-1]] == 0.0:
          if w_giv_plus_u[word[-1]]*w_giv_plus_bak[bnextword] == 0.0:
            entry['bwprob'] += math.log(10**-200)
          else:
            entry['bwprob'] += math.log(w_giv_plus_u[word[-1]]*w_giv_plus_bak[bnextword])
        else:
          entry['bwprob'] += math.log(w_giv_plus[bnextword][word[-1]])
    else:
      #Unigram probability
      entry['uprob'] = math.log(max(unigrams[word],10**-200))

      #Forward bigram probability
      if chainruleit(prevword):
        (crfwprevprob,crbwprevprob),prevwordroot = chainruleprobs(prevword)
        if w_giv_min[prevword[-1]][word] == 0.0:
          if w_giv_min_u[word]*w_giv_min_bak[prevword[-1]] == 0.0:
            entry['fwprob'] = math.log(10**-200) + crfwprevprob
          else:
            entry['fwprob'] = math.log(w_giv_min_u[word]*w_giv_min_bak[prevword[-1]]) + crfwprevprob
        else:
          entry['fwprob'] = math.log(w_giv_min[prevword[-1]][word]) + crfwprevprob
      else:
        if w_giv_min[prevword][word] == 0.0:
          if w_giv_min_u[word]*w_giv_min_bak[prevword] == 0.0:
            entry['fwprob'] = math.log(10**-200)
          else:
            entry['fwprob'] = math.log(w_giv_min_u[word]*w_giv_min_bak[prevword])
        else:
          entry['fwprob'] = math.log(w_giv_min[prevword][word])

      #Backward bigram probability
      if chainruleit(bnextword):
        (crfwbnextprob,crbwbnextprob),bnextwordroot = chainruleprobs(bnextword)
        if w_giv_plus[bnextwordroot][word] == 0.0:
          if w_giv_plus_u[word]*w_giv_plus_bak[bnextwordroot] == 0.0:
            entry['bwprob'] = math.log(10**-200) + crbwbnextprob
          else:
            entry['bwprob'] = math.log(w_giv_plus_u[word]*w_giv_plus_bak[bnextwordroot]) + crbwbnextprob
        else:
          entry['bwprob'] = math.log(w_giv_plus[bnextwordroot][word]) + crbwbnextprob
      else:
        if w_giv_plus[bnextword][word] == 0.0:
          if w_giv_plus_u[word]*w_giv_plus_bak[bnextword] == 0.0:
            entry['bwprob'] = math.log(10**-200)
          else:
            entry['bwprob'] = math.log(w_giv_plus_u[word]*w_giv_plus_bak[bnextword])
        else:
          entry['bwprob'] = math.log(w_giv_plus[bnextword][word])

    entry.update({'endofsentence':SENTEND,'startofsentence':SENTBEGIN,'word':word,'sentpos':sentpos,'trigram':[prevword,word,nextword]})
    sentpos += 1

    corpus.append(dict(entry))

    SENTBEGIN = False

#########################
#
# Output Compiled Data
#
#########################

header = ['word','uprob','bwprob','fwprob','endofsentence','startofsentence','sentpos']

sys.stderr.write('Writing output\n')

sys.stdout.write(' '.join(header)+'\n')

for w in corpus:
  #output ngrams
  #sys.stdout.write(' '.join(str(w[heading]) for heading in header+['trigram'])+'\n') #outputs stats and trigrams
  sys.stdout.write(' '.join(str(w[heading]) for heading in header)+'\n') #outputs stats
