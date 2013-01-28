# Eval script for the Rimell et al (2009) longrange dependency parsing task
import sys
import re
import getopt

# python lrdepeval.py (-i) <goldfile> <hypothfile>  (where -i ignores role label, just compares topology)

optlist,args = getopt.gnu_getopt(sys.argv,'r')
gfile = open(args[1],'r')
hfile = open(args[2],'r')

gtot = 0.0
htot = 0.0
ctot = 0.0

def expandrels(rlist,hdict,revdict,gold1=''):
  #given the list of hypothesized relations, forward and backward indices, and the gold head...
  #follow relation links through transparent nodes (conjunctions, prepositions, etc)
  outlist = []
  for rel in rlist:
    #for each relation we're attempting to follow...
    if hdict[rel][1]['0'][0] == 'CC':
      #follow to linked conjuncts
      if '1' in hdict[rel][1].keys():
        outlist += expandrels(hdict[rel][1]['1'],hdict,revdict)
      if '2' in hdict[rel][1].keys():
        outlist += expandrels(hdict[rel][1]['2'],hdict,revdict)
    elif '-rN' in hdict[rel][1]['0'][0]:
      #follow through relativizers
      try:
        outlist += expandrels(hdict[rel][1]['2'],hdict,revdict)
      except: #recovers from failure to link relativizer
        outlist.append(rel)
    elif gold1 != '' and 'OS' == hdict[gold1][1]['0'][0][:2] and '-g' not in hdict[gold1][1]['0'][0][2:]:
      #invert modifier/modificand relation due to theoretical differences with Rimell et al 09/10
      outlist += revdict[rel]['2']
    else:
      #Rimell et al 09/10 give the second dependency for free if the first is correct and coordination is correct
      conj = False
      if '1' in revdict[rel].keys():
        relset = revdict[rel]['1']
        for r in relset:
          if hdict[r][1]['0'][0] == 'CC':
            conj = True
            for k in hdict[r][1].keys():
              if k != '0':
                outlist += hdict[r][1][k]
        if not conj:
          outlist.append(rel)
      else:
        outlist.append(rel)
  return outlist

for gsent in gfile:
    #for each gold sentence...
    hdict = {}
    hsent = hfile.readline()
    #grab the hypothesized parse...
    hl = hsent.split()
    for hw in hl:
      #create an entry for each word in that parse
      if '#MAXPROJ' in hw or '#-' == hw[-2:]: #'#-' handles bug(?) in trees2melconts which ends rels with '-' instead of 'MAXPROJ'
        #if relation isn't identity, ignore for now
        continue
      else:
        #if relation is identity, add info to a new hdict entry
        if hw[0] == '#':
          hwt = hw[1:].split('#')
          hwt[0] = '#'+hwt[0]
        else:
          hwt = hw.split('#')
        hws = hwt[0].split('/')
        hwlex = hwt[1]
        hdict[hws[0]] = (hwlex,{hws[1]:[hws[2]]})

    for hw in hl:
      #update each entry with dependency relations
      if '#MAXPROJ' in hw:
        hws = hw[:-8].split('/')
        if hws[1] in hdict[hws[0]][1].keys():
          #if the given relation already exists in this entry, append to its deps
          hdict[hws[0]][1][hws[1]].append(hws[2])
        else:
          #otherwise, create that type of relation
          hdict[hws[0]][1][hws[1]] = [hws[2]]
      elif '#-' == hw[-2:]: #handles bug(?) in trees2melconts which ends rels with '-' instead of 'MAXPROJ'
        hws = hw[:-2].split('/')
        if hws[1] in hdict[hws[0]][1].keys():
          #if the given relation already exists in this entry, append to its deps
          hdict[hws[0]][1][hws[1]].append(hws[2])
        else:
          #otherwise, create that type of relation
          hdict[hws[0]][1][hws[1]] = [hws[2]]

    gl = gsent.split()

    #create revdict
    revdict = {}
    for k in hdict.keys():
      #create a reverse index from deps via relations to heads
      rels = hdict[k][1]
      for rel in rels.keys():
        for r in rels[rel]:
          if r in revdict.keys():
            if rel in revdict[r].keys():
              revdict[r][rel].append(k)
            else:
              revdict[r][rel] = [k]
          else:
            revdict[r] = {}
            revdict[r][rel] = [k]

    lFound = []
    ctotlocal = 0
    for gw in gl:
      #for each gold word...
      gfound = False
      gws = gw.split('/')
      try:
        #if we hypothesized the correct relation for the gold head, check that it links to the correct dep
        if hdict != {} and gws[1] in hdict[gws[0]][1]:
         hrels = hdict[gws[0]][1][gws[1]]
         recrels = expandrels(hrels,hdict,revdict,gws[0])

         for rela in recrels:
           if gws[2] == rela:
             gfound = True
             break
      except:
        #something broke, so report the relevant elements and abort
        print('gsent: '+gsent)
        print('hsent: '+hsent)
        print('hdict: '+str(hdict))
        print('gw: '+str(gw))
        raise

      cFound = ' '
      if gfound:
        #if we found the correct linkage with the correct nodes, record a success
        ctot += 1.0
        ctotlocal += 1
        cFound = '+'
      lFound += '['+cFound+']'+gw,

    #report recorded successes
    gsent = re.sub('(?!</)#[^ ]*','',gsent)
    hsent = re.sub('(?!</)#[^ ]*','',hsent)
    Pg = re.sub('[^ ]*/0/[^ ]*','',gsent).split()
    Ph = re.sub('[^ ]*/0/[^ ]*','',hsent).split()

    gtotlocal = len(Pg)
    htotlocal = len(Ph)
    gtot += gtotlocal
    htot += htotlocal

    if gtot == 0:
      recall = 0
    else:
      recall = ctot/gtot

    if htot == 0:
      precis = 0
    else:
      precis = ctot/htot
    fscore = 'NaN'
    print(' ('+str(ctotlocal)+'/'+str(gtotlocal)+')', end=' ')
    print(' '.join(lFound))

print('TOT recall:', str(ctot)+'/'+str(gtot), 'precis:', str(ctot)+'/'+str(htot))
print('PCT recall:',recall, 'precis:',precis, 'fscore:',fscore)
