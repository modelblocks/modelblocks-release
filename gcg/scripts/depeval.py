
import sys
import re
import getopt

# python peval.py (-i) <goldfile> <hypothfile>  (where -i ignores role label, just compares topology)

optlist,args = getopt.gnu_getopt(sys.argv,'rl')
gfile = open(args[1],'r')
hfile = open(args[2],'r')
labelIgnored = '-l' in sys.argv

def click(gprop, hypoths):
    if not labelIgnored:
        return gprop in hypoths
    ge,gl,gf = gprop.split('/')
    for hprop in hypoths:
        he,hl,hf = hprop.split('/')
        if he==ge and hf==gf:
            return True
    return False

gtot = 0.0
htot = 0.0
ctot = 0.0

for gsent in gfile:
    hsent = hfile.readline()

    gsent = re.sub('(?!</)#[^ ]*','',gsent)
    hsent = re.sub('(?!</)#[^ ]*','',hsent)
    Pg = re.sub('[^ ]*/0/[^ ]*','',gsent).split()
    Ph = re.sub('[^ ]*/0/[^ ]*','',hsent).split()

    gtotlocal = len(Pg)
    htotlocal = len(Ph)
    ctotlocal = 0
    gtot += gtotlocal
    htot += htotlocal

    lFound = []
    for gprop in Pg:
        cFound = ' '
        if click(gprop, Ph): #gprop in Ph:
            ctot += 1.0
            ctotlocal += 1
            cFound = '+'
        lFound += '['+cFound+']'+gprop,
    
    if gtot>0 and htot>0:
      recall = ctot/gtot
      precis = ctot/htot
      fscore = 2 * recall * precis / (recall+precis)
      print(' ('+str(ctotlocal)+'/'+str(gtotlocal)+')', end=' ')
      if '-r' in sys.argv: print(recall, fscore, end=' ')
      print(' '.join(lFound))

recall = ctot/gtot
precis = ctot/htot
fscore = 2 * recall * precis / (recall+precis)

print('TOT recall:', str(ctot)+'/'+str(gtot), 'precis:', str(ctot)+'/'+str(htot))
print('PCT recall:',recall, 'precis:',precis, 'fscore:',fscore)


