import sys
import collections

MINCOUNT = 100

Items  = [ ]
XFeatCount = collections.defaultdict(int)
YValCount  = collections.defaultdict(int)
for line in sys.stdin:
  XFeats = line.partition(' : ')[0].split(',')
  for xf in XFeats:
    XFeatCount[xf] += 1
  y = line.partition(' : ')[2][:-1]
  YValCount[y] += 1
  Items.append((XFeats,y))

for XFeats,y in Items:
  commonfeats = [xf for xf in XFeats if XFeatCount[xf]>=MINCOUNT]
  print (','.join(commonfeats) if len(commonfeats)>0 else XFeats[0].partition('&')[0]+'&bot=1') + ' : ' + (y if YValCount[y]>=MINCOUNT else y.partition('&')[0]+'&bot')


