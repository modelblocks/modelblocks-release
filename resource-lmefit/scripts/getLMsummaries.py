import sys, re, os
from numpy import inf

R = re.compile('(\[[0-9]+\] "?)?([^"$]*)"?')

def deRify(s):
    return R.match(s).group(2)

# Thanks to Daniel Sparks on StackOverflow for this one (post available at 
# http://stackoverflow.com/questions/5084743/how-to-print-pretty-string-output-in-python)
def getPrintTable(row_collection, key_list, field_sep=' '):
  return '\n'.join([field_sep.join([str(row[col]).ljust(width)
    for (col, width) in zip(key_list, [max(map(len, column_vector))
      for column_vector in [ [v[k]
        for v in row_collection if k in v]
          for k in key_list ]])])
            for row in row_collection])

assert len(sys.argv) > 1

fit_list = sys.argv[1:]

rows = []

for path in fit_list:
    loglik = inf
    with open(path, 'rb') as f:
        text = f.readlines()
        name = path.split('.')[-3]
        cliargs = path.split('.')[-2]
        for i,l in enumerate(text):
            if deRify(l.strip()).startswith('logLik'):
                loglik = float(deRify(text[i+1].strip()))
    rows.append({'modelname': name, 'logLik': str(loglik), 'cliargs': str(cliargs)})

headers = ['modelname', 'logLik', 'cliargs']
header_row = {}
for h in headers:
    header_row[h] = h

converged = rows[:] 
converged.sort(key = lambda x: x['logLik'])
converged.insert(0, header_row)

if len(converged) > 1:
    print('===================================')
    print('Converged models')
    print('===================================')

    print(getPrintTable(converged, headers))

