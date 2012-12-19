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
  outlist = []
  for rel in rlist:
    if hdict[rel][1]['0'][0] == 'CC':
      if '1' in hdict[rel][1].keys():
        outlist += expandrels(hdict[rel][1]['1'],hdict,revdict)
      if '2' in hdict[rel][1].keys():
        outlist += expandrels(hdict[rel][1]['2'],hdict,revdict)
    elif '-oR' in hdict[rel][1]['0'][0]:
      try:
        outlist += expandrels(hdict[rel][1]['2'],hdict,revdict)
      except: #recover from failure to link relativizer
        outlist.append(rel)
    elif gold1 != '' and 'NE' == hdict[gold1][1]['0'][0][:2] and '-g' not in hdict[gold1][1]['0'][0][2:]:
          outlist += revdict[rel]['2']
    else:
      #rimell et al 09/10 give the second dependency for free if the first is correct and coordination is correct
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
    hdict = {}
    hsent = hfile.readline()
    hl = hsent.split()
    for hw in hl:
      if '#MAXPROJ' in hw or '#-' == hw[-2:]: #'#-' handles bug(?) in trees2melconts which ends rels with '-' instead of 'MAXPROJ'
        continue
      else:
        if hw[0] == '#':
          hwt = hw[1:].split('#')
          hwt[0] = '#'+hwt[0]
        else:
          hwt = hw.split('#')
        hws = hwt[0].split('/')
        hwlex = hwt[1]
        hdict[hws[0]] = (hwlex,{hws[1]:[hws[2]]})

    for hw in hl:
      if '#MAXPROJ' in hw:
        hws = hw[:-8].split('/')
        if hws[1] in hdict[hws[0]][1].keys():
          hdict[hws[0]][1][hws[1]].append(hws[2])
        else:
          hdict[hws[0]][1][hws[1]] = [hws[2]]
      elif '#-' == hw[-2:]: #handles bug(?) in trees2melconts which ends rels with '-' instead of 'MAXPROJ'
        hws = hw[:-2].split('/')
        if hws[1] in hdict[hws[0]][1].keys():
          hdict[hws[0]][1][hws[1]].append(hws[2])
        else:
          hdict[hws[0]][1][hws[1]] = [hws[2]]

    gl = gsent.split()

    #create revdict
    revdict = {}
    for k in hdict.keys():
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
      gfound = False
      gws = gw.split('/')
      if hdict != {} and gws[1] in hdict[gws[0]][1]:
        hrels = hdict[gws[0]][1][gws[1]]
        recrels = expandrels(hrels,hdict,revdict,gws[0])

        for rela in recrels:
          if gws[2] == rela:
            gfound = True
            break

      cFound = ' '
      if gfound:
        ctot += 1.0
        ctotlocal += 1
        cFound = '+'
      lFound += '['+cFound+']'+gw,

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



