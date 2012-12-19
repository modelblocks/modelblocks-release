import sys
import re
import getopt

# python convertGoldUnbound.py (-r) <goldfile>

optlist,args = getopt.gnu_getopt(sys.argv,'ruf')
gfile = open(args[1],'r')

flowix = 0
for l in gfile:
  if l == "\n":
    if flowix == 2:
      print("")
    flowix = 0
  elif flowix == 2: #dealing with gold relations
    ll = l.split()
    e1 = ll[1].split('_')
    e2 = ll[2].split('_')
    if ll[0] in ('nsubj','nsubjpass'):
      r = '1'
    elif ll[0] in ('advmod','prep','nn'):
      r = '1'
      e3 = e1
      e1 = e2
      e2 = e3
    elif ll[0] in ('dobj','pobj','obj2'):
      r = '2'
    else:
      if '-f' in sys.argv:
        r = 'NaN'
      else:
        print(l[:-1], end=" ")
        continue
    print(e1[0][0].lower()+e1[1]+'/'+r+'/'+e2[0][0].lower()+e2[1],end=' ')
    if '-u' in sys.argv:
      print(e2[0][0].lower()+e2[1]+'/'+r+'/'+e1[0][0].lower()+e1[1],end=' ')
  else:
    flowix += 1
