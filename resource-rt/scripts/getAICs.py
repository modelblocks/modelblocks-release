import sys, os
from numpy import inf

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

dir = sys.argv[1]

fit_list = [x for x in os.listdir(dir) if x.endswith('.lmefit')]

rows = []

for path in fit_list:
    relgrad = inf
    aic = -inf
    with open(dir + '/' + path, 'rb') as f:
        text = f.readlines()
        name = path.split('.')[-3]
        for i,l in enumerate(text):
            if l.strip() == '[1] "Relgrad:"':
                relgrad_cur = float(text[i+1].split()[1])
                aic_cur = float(text[i+3].split()[1])
                if relgrad_cur < relgrad:
                    relgrad = relgrad_cur
                    aic = aic_cur
    rows.append({'filename': name, 'relgrad': str(relgrad), 'AIC': str(aic)})

headers = ['filename', 'relgrad', 'AIC']
header_row = {}
for h in headers:
    header_row[h] = h

converged = [r for r in rows if float(r['relgrad']) <= 0.002]
converged.sort(reverse=True, key = lambda x: x['AIC'])
converged.insert(0, header_row)

nonconverged = [r for r in rows if float(r['relgrad']) > 0.002]
nonconverged.sort(reverse=True, key = lambda x: x['AIC'])
nonconverged.insert(0, header_row)

if len(converged) > 1:
    print('===================================')
    print('Converged models')
    print('===================================')

    print(getPrintTable(converged, headers))

if len(nonconverged) > 1:
    print('===================================')
    print('Non-converged models')
    print('===================================')

    headers = ['filename', 'relgrad', 'AIC']

    print(getPrintTable(nonconverged, headers))

