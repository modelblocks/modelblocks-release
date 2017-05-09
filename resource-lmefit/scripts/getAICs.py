import sys, os
from numpy import inf

def deRify(s):
    if s.startswith('[1] "'):
        s = s[5:]
    return s

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
    converged = True
    relgrad = inf
    aic = -inf
    bic = -inf
    loglik = inf
    with open(path, 'rb') as f:
        text = f.readlines()
        name = path.split('.')[-3]
        for i,l in enumerate(text):
            if deRify(l.strip()).startswith('AIC        BIC'):
                aic_cur, bic_cur, loglik_cur = text[i+1].strip().split()[:3]
                aic_cur = float(aic_cur)
                bic_cur = float(bic_cur)
                loglik_cur = float(loglik_cur)
            elif deRify(l.strip()).startswith('Relgrad'):
                relgrad_cur = float(text[i+1].strip())
                aic_cur = float(text[i+3].strip())
                if relgrad_cur < relgrad:
                    relgrad = relgrad_cur
                    aic = aic_cur
                    bic = bic_cur
                    loglik = loglik_cur
            elif deRify(l.strip()).startswith('Model failed to converge under both bobyqa and nlminb'):
                converged = False
    rows.append({'filename': name, 'relgrad': str(relgrad), 'AIC': str(aic), 'BIC': str(bic), 'logLik': str(loglik), 'converged': str(converged)})

headers = ['filename', 'logLik', 'AIC', 'BIC', 'converged', 'relgrad']
header_row = {}
for h in headers:
    header_row[h] = h

converged = [r for r in rows if r['converged'] == 'True']
converged.sort(key = lambda x: x['logLik'])
converged.insert(0, header_row)

nonconverged = [r for r in rows if r['converged'] == 'False']
nonconverged.sort(key = lambda x: x['logLik'])
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

    print(getPrintTable(nonconverged, headers))

