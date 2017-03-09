import sys, os, math
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
from model import Model,CondModel

srif = open(sys.argv[1],mode='r',encoding='latin-1')

opts = str(sys.argv[2][1:]).upper()

if opts != 'U':
  suni = Model('U'+opts)
else:
  suni = Model('U')
sunib = Model('UB'+opts)
sbi = CondModel(opts)

ver = -1

for line in srif:
  if line[0] == '\\':
    ver += 1
    if opts == 'U' and ver > 1:
      break
    continue
  elif ver == 0:
    continue

  sline = line.split()
  if (len(sline) < 3 and opts != 'U') or (opts == 'U' and len(sline) < 2):
    continue
  if ver == 1:
    suni[sline[1]] = 10**float(sline[0])
    if opts != 'U':
      sunib[sline[1]] = 10**float(sline[2])
  elif ver == 2:
    sbi[sline[1]][sline[2]] = 10**float(sline[0])

srif.close()

suni.write()
if opts != 'U':
  sunib.write()
  sbi.write()
