#python buildDundeeTokFixns.py origFile eyeFile eventFile

import sys
import re

textfileix = 1
textfixfileix = 2
eventfileix = 3

#########################
#
# Load Text File
#
#########################

#NB: startofsentence and endofsentence measures aren't accurate here, so we shouldn't include them here

corpus = []

sys.stderr.write('Loading original text\n')

fnum = 0 #id of file being processed
wnumoffset = -1 #file offset for word ids
prevwnum = 0

refilebreak = re.compile('^WORD ')

with open(sys.argv[textfileix],'r',encoding='latin-1') as textFile:
  for line in textFile.readlines():
    #beginning processing a new file
    if refilebreak.match(line) != None:
      fnum += 1
      wnumoffset += prevwnum
      continue
    sline = line.split()

    word = sline[0]
    screenid = int(sline[2]) #screen relative to file
    lineid = int(sline[3]) #line relative to file
    lw = int(sline[4]) #wnum relative to current line
    screenwnum = int(sline[5]) #wnum relative to current snum
    wnum = int(sline[12]) #wnum relative to current fnum

    corpus.append( {'word':word,'wordid':wnum+wnumoffset,'fileid':fnum,'screenid':screenid,'lineid':lineid,\
             'startoffile':False,'startofline':False,'startofscreen':False,\
             'endoffile':False,'endofline':False,'endofscreen':False} )
#             'startofsentence':False,'endofsentence':False} )
    if lw == 1:
      #first word in line
      corpus[-1]['startofline'] = True
      if wnum == 1:
        #first word in a file
        corpus[-1]['startoffile'] = True
        corpus[-1]['startofscreen'] = True
#        corpus[-1]['startofsentence'] = True
        if fnum != 1:
          #first word in nonfirst file
          corpus[-2]['endoffile'] = True
          corpus[-2]['endofscreen'] = True
          corpus[-2]['endofline'] = True
#          corpus[-2]['endofsentence'] = True
        #else: #first word in first file
      else:
        corpus[-2]['endofline'] = True
        if screenwnum == 1:
          #start a new screen
          corpus[-1]['startofscreen'] = True
          corpus[-2]['endofscreen'] = True
#          corpus[-1]['startofsentence'] = True
#          corpus[-2]['endofsentence'] = True
    prevwnum = wnum

corpus[-1]['endoffile'] = True
corpus[-1]['endofscreen'] = True
corpus[-1]['endofline'] = True
#corpus[-1]['endofsentence'] = True

#########################
#
# Load Fixation Information
#
#########################

sys.stderr.write('Loading eyetracked text\n')

fnum = 0 #id of file being processed
prevwnum = 0 #id of last word in a file
wnumoffset = -1 #file offset for word ids
maxfix = 0 #id of last fixation in a file
fixoffset = -1 #file offset for fixation ids

with open(sys.argv[textfixfileix],'r',encoding='latin=1') as eyeFile:
  for line in eyeFile.readlines():
    #restart file count
    if refilebreak.match(line) != None:
      fnum += 1
      wnumoffset += prevwnum
      fixoffset += maxfix
      maxfix = 0
      prevwnum = 0
      continue

    sline = line.split()
    word = sline[0]
    snum = int(sline[1]) #screen number
    lnum = int(sline[2]) #line number
    olen = int(sline[3]) #word length (including punctuation) #NB: this is 0 if no fixation occurred!
    wlen = int(sline[4]) #word length (excluding puncutation)
    xpos = sline[5]
    wnum = int(sline[6]) #wnum relative to fnum
    fdur = float(sline[7]) #!!!?Can this be a float?
    oblp = sline[8]
    wdlp = sline[9]
    fxno = int(sline[10])
    txfr = int(sline[11]) #local text frequency

    #this could be expanded to be a tuple composed of whatever above values are desired
    # CAUTION: If a word is not fixated, all of the above values will be 0 except snum, which will be -99 and wnum, which will be real
    #sys.stderr.write(str(wnumoffset)+'+'+str(wnum)+':'+str(fxno)+','+str(fdur)+'\n')
    corpus[wnumoffset+wnum].setdefault('fixations',{})[fixoffset+fxno] = fdur
    corpus[wnumoffset+wnum]['wlen'] = wlen
    corpus[wnumoffset+wnum]['olen'] = olen

    if fxno > maxfix:
      maxfix = fxno
    prevwnum = wnum

#########################
#
# Load Event Data
#
#########################

#NB: Currently, we don't use Eventfile, but it can be included similar to eyeFile
#    Care must be taken to properly sync it up, and words beginning with '*' are Off-screen or Blink
#    It might be interesting in the future to see if these have effects on processing (e.g. fixations after a blink might be longer)
#    It might also be interesting to look at landing positions and launch distances to see if they affect processing

#sys.stderr.write('Loading event data\n')

#fnum = 0 #id of current file being processed
#wnumoffset = -1

#with open(sys.argv[eventfileix],'r',encoding='latin-1') as eventFile:

  #iterate over eyetracking output until completion
#  for line in eventFile.readlines():
    #restart file count
#    if refilebreak.match(line) != None:
#      wnumoffset += offsets[fnum] #need to have an offset dict from eyeFile
#      fnum += 1
#      continue
#    sline = line.split()
#    word = sline[0]
#    snum = int(sline[1])
#    lnum = int(sline[2])
#    olen = int(sline[3]) #length of word (including punc) #NB: this is 0 if no fixation occurred!
#    wlen = int(sline[4])
#    xpos = sline[5]
#    wnum = int(sline[6]) #wnum relative to fnum
#    fdur = sline[7]
#    oblp = sline[8]
#    wdlp = int(sline[9])
#    laun = int(sline[10])
#    txfr = sline[11]

#    corpus[wnum+wnumoffset]['laundist'] = laun
#    corpus[wnum+wnumoffset]['landpos'] = wdlp

#########################
#
# Output Compiled Data
#
#########################

sys.stderr.write('Writing output\n')

fixCorpus = {}

#reindex corpus by fixation ids
for i,w in enumerate(corpus):

  for f in w['fixations'].keys():
    PREVWASFIX = False
    PREVWILLFIX = False
    NEXTWILLFIX = False
    NEXTWASFIX = False

    #for each fixation
    #there are plenty of different ways to calculate previsfix/nextisfix
#    if i != 0 and not w['startofscreen'] and len(corpus[i-1]['fixations'].keys()) > 0: #if previous word is ever fixated
    if i != 0 and not w['startofscreen'] and f-1 in corpus[i-1]['fixations'].keys(): #if previous word is fixated immediately previous
      #if we're not the beginning of a new screen and the previous word is fixated...
      PREVWASFIX = True
    if i != 0 and not w['startofscreen'] and f+1 in corpus[i-1]['fixations'].keys(): #if previous word is fixated immediately after
      PREVWILLFIX = True

#    if i != len(corpus)-1 and not w['endofscreen'] and len(corpus[i+1]['fixations'].keys()) > 0: #if next word is ever fixated
#    if i != len(corpus)-1 and not w['endofscreen'] and f+1 in corpus[i+1]['fixations'].keys(): #if next word is fixated immediately after
    if i != len(corpus)-1 and not w['endofscreen'] and f-1 in corpus[i+1]['fixations'].keys(): #if next word is fixated immediately previous
      #if we're not the end of a screen and the next word is fixated...
      NEXTWASFIX = True
    if i != len(corpus)-1 and not w['endofscreen'] and f+1 in corpus[i+1]['fixations'].keys(): #if next word is fixated immediately after
      NEXTWILLFIX = True

    if not fixCorpus: #if fixCorpus is empty
      wdelta = w['wordid']+1
    else: #if fixCorpus has fixations in it
      #calc wdelta from the preceding next smallest fixation's wordid
      # (this is not guaranteed to be correct in cases of non region-frontiers)
      for k in reversed(sorted(fixCorpus)):
        if k < f and fixCorpus[k]['wordid'] != w['wordid']: #self-fixations don't count
          wdelta = w['wordid']-fixCorpus[k]['wordid']
          break
    fixCorpus[f] = {'fdur':w['fixations'][f],'cumwdelta':wdelta,'prevwasfix':PREVWASFIX,'prevwillfix':PREVWILLFIX,'nextwasfix':NEXTWASFIX,'nextwillfix':NEXTWILLFIX}
    for k in w.keys():
      #for every key that's not fixation info, add it to the entry
      if k != 'fixations':
        fixCorpus[f][k] = w[k]

firstfix = min(k for k in fixCorpus.keys() if k > 0)
outputheadings = ['word','fdur','wordid'] + [k for k in fixCorpus[firstfix].keys() if k not in ('word','fdur','wordid')]

sys.stdout.write(' '.join(outputheadings)+'\n')

for fix in sorted(fixCorpus.keys()):
  #order output by fixation ids
  #we also remove artifacts where readers saw the final word of the next file before going to the beginning of that file
  if fix >= 0:
    if fixCorpus[fix]['fdur'] != 0.0:
      sys.stdout.write(' '.join(str(fixCorpus[fix][heading]) for heading in outputheadings)+'\n')
