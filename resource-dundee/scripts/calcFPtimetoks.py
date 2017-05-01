#python calcXtimetoks.py tokfixnFile textFile 

import sys
import re

tokfixnfileix = 1
textfileix = 2

#########################
#
# Load TokFixns File
#
#########################

DEBUG = False

corpus = []

sys.stderr.write('Loading tokfixns\n')

firstLine = True
firstWord = True
header = []

accumulated = {'fdur':0,'startofline':False,'endofline':False,'startoffile':False,'endoffile':False,'startofscreen':False,'endofscreen':False,'unknown':False,'words':[]}
prevwnum = 0
prevprevwnum = -1

with open(sys.argv[tokfixnfileix],'r',encoding='latin-1') as tokfixnFile:
  for line in tokfixnFile.readlines():
    #Store the first line as headers
    if firstLine:
      firstLine = False
      header = line.strip().split()
      continue
    #store remaining lines in corpus
    sline = line.strip().split()
    entry = {}

    for hix in range(len(header)):
      if header[hix] == 'wordid':
        entry[header[hix]] = int(sline[hix])
      elif header[hix] == 'fdur':
        entry[header[hix]] = float(sline[hix])
      else:
        entry[header[hix]] = sline[hix]

    ####################
    #
    # Accumulate stats for fixation metric below
    #
    ####################

    if int(entry['wordid']) > prevwnum and not firstWord:
      #store the current region's stats and start a new region
      corpus.append(dict(accumulated))
      accumulated = dict(entry)
      prevprevwnum = prevwnum
      prevwnum = entry['wordid']
      accumulated['words'] = []
    elif int(entry['wordid']) > prevprevwnum:
      #accumulate the current region's stats
      if firstWord:
        #or start a new region for the first fixation in the corpus
        firstWord = False
        accumulated = dict(entry)
        accumulated['words'] = []
        prevwnum = entry['wordid']
      else:
        accumulated['words'].append(entry['word'])
        accumulated['fdur'] += entry['fdur']
      for h in header:
        #accumulate info from header columns
        if 'start' in h or 'end' in h or h == 'unknown':
          if entry[h] == 'True':
            accumulated[h] = True
        elif (h not in ('words','fdur','word','wdelta') and h[-2:] != 'id'):
          accumulated[h] = entry[h]

    ####################
    #
    # End of stat accumulation
    #
    ####################

#append final region in corpus
corpus.append(dict(accumulated))


#########################
#
# Insert remainder of corpus 
#
#########################

outCorpus = []

with open(sys.argv[textfileix],'r',encoding='latin-1') as textFile:
  lines = textFile.readlines()
  textix = 0
  modix = 1 #keeps track of file breaks so we don't count them as corpus words
  entry = {'fdur':0}
  for region in corpus:
    #for each region we have stats for
    while textix < region['wordid']:
      #add words until we reach end of region
      if lines[textix+modix].strip() == 'WORD':
        #be sure not to count the file breaks as legitimate corpus words
        modix += 1
      entry.update({'word':lines[textix+modix].split()[0],'wordid':textix})
      outCorpus.append(dict(entry))
      textix += 1
    #now we're at the region boundary, so append stats for region
    if lines[textix+modix].strip() == 'WORD':
      #be sure not to count the file breaks as legitimate corpus words
      modix += 1
    outCorpus.append(region)
    textix += 1
  while textix+modix < len(lines):
    #fixations didn't reach the end of the file, so add non-fixations until we complete the corpus
    if lines[textix+modix].strip() == 'WORD':
      #be sure not to count the file breaks as legitimate corpus words
      modix += 1
    entry.update({'word':lines[textix+modix].split()[0],'wordid':textix})
    outCorpus.append(dict(entry))
    textix += 1

#########################
#
# Output Compiled Data
#
#########################

sys.stderr.write('Writing output\n')

SubjectLabel = re.search('\.(s[a-j])\.tokfixns$',sys.argv[tokfixnfileix]).group(1)

sys.stdout.write('subject ')
sys.stdout.write(' '.join(header)+'\n')

for w in outCorpus:
  #output region stats
  sys.stdout.write(SubjectLabel+' ')
  sys.stdout.write(' '.join(str(w.get(heading,'nan')) for heading in header)+'\n')
