import sys, re, argparse

argparser = argparse.ArgumentParser('''
Extracts recall values from a user-supplied list of lrtsignifs and outputs a space-delimited table of results.
''')
argparser.add_argument('lrtsignifs', type=str, nargs='+', help='One or more *.lrtsignif files from which to extract lme comparison results.')
args, unknown = argparser.parse_known_args()

val = re.compile('^.+: *([^ "]+)"?$')
effectpair = re.compile('([^ ]+)-vs-([^ ]+)')

R = re.compile('(\[[0-9]+\] "?)?([^"$]*)"?')

def deRify(s):
    return R.match(s).group(2)

def compute_row(f, diamName=None, vs=None):
    row = {}
    line = f.readline()
    while line and not deRify(line).startswith('Main effect'):
        line = f.readline()
    assert line, 'Input not properly formatted'
    row['effect'] = val.match(line).group(1)
    line = f.readline()
    assert deRify(line).startswith('Corpus'), 'Input not properly formatted'
    row['corpus'] = val.match(line).group(1)
    line = f.readline()
    assert deRify(line).startswith('Effect estimate'), 'Input not properly formatted'
    row['estimate'] = '%.5g'%(float(val.match(line).group(1)))
    line = f.readline()
    assert deRify(line).startswith('t value'), 'Input not properly formatted'
    row['t value'] = '%.5g'%(float(val.match(line).group(1)))
    line = f.readline()
    assert deRify(line).startswith('Significance (Pr(>Chisq))'), 'Input not properly formatted'
    row['signif'] = '%.5g'%(float(val.match(line).group(1)))
    line = f.readline()
    assert deRify(line).startswith('Relative gradient (baseline)'), 'Input not properly formatted'
    row['rel_grad_base'] = '%.5g'%(float(val.match(line).group(1)))
    line = f.readline()
    if (deRify(line).startswith('Converged (baseline)')):
        row['converged_base'] = val.match(line).group(1)
    else:
        row['converged_base'] = str(float(row['rel_grad_base']) < 0.002)
    assert deRify(line).startswith('Relative gradient (main effect)'), 'Input not properly formatted'
    row['rel_grad_main'] = '%.5g'%(float(val.match(line).group(1)))
    if (deRify(line).startswith('Converged (main effect)')):
        row['converged_main'] = val.match(line).group(1)
    else:
        row['converged_main'] = str(float(row['rel_grad_main']) < 0.002)
    if diamName:
        row['diamondname'] = diamName
        left = effectpair.match(diamName).group(1)
        right = effectpair.match(diamName).group(2)
    if vs == 'base':
        row['pair'] = str(row['effect']) + '-vs-baseline'
    elif vs == 'both':
        if str(row['effect']) == left:
            base = right
        else:
            base = left
        row['pair'] = 'both-vs-' + base
    elif vs != None:
        row['pair'] = str(row['effect']) + '-vs-' + str(vs)
    return(row)

def print_row(row):
    out = [row['effect'], row['corpus'], row['estimate'], row['t value'], row['signif'], row['rel_grad_base'], row['rel_grad_main']]
    print(' '.join(out))

# Thanks to Daniel Sparks on StackOverflow for this one (post available at 
# http://stackoverflow.com/questions/5084743/how-to-print-pretty-string-output-in-python)
def getPrintTable(row_collection, key_list, field_sep=' '):
  return '\n'.join([field_sep.join([str(row[col]).ljust(width)
    for (col, width) in zip(key_list, [max(map(len, column_vector))
      for column_vector in [ [v[k]
        for v in row_collection if k in v]
          for k in key_list ]])])
            for row in row_collection])

pair_evals = [x for x in args.lrtsignifs if 'diamond' not in x]
diam_evals = [x for x in args.lrtsignifs if 'diamond' in x]

if len(pair_evals) > 0:
    print('===================================')
    print('Pairwise evaluation of significance')
    print('===================================')

    headers = ['effect', 'corpus', 'estimate', 't value', 'signif', 'converged_base', 'rel_grad_base', 'converged_main', 'rel_grad_main', 'filename']
    
    header_row = {}
    for h in headers:
        header_row[h] = h

    rows = []

    for path in pair_evals:
        with open(path, 'rb') as f:
            filename = path.split('/')[-1]
            row = compute_row(f)
            row['filename'] = filename
            rows.append(row)

    converged = [header_row] + sorted([x for x in rows if (x['converged_base'] == 'True' and x['converged_main'] == 'True')], \
                key = lambda y: float(y['signif']))
    nonconverged = [header_row] + sorted([x for x in rows if not(x['converged_base'] == 'True' and x['converged_main'] == 'True')], \
                   key = lambda y: float(y['signif']))

    print(getPrintTable(converged, headers))

    if len(nonconverged) > 1: #First element is the header row
        print('-----------------------------------')
        print('Convergence failures')
        print('-----------------------------------')
        print(getPrintTable(nonconverged, headers))

    print ''
    print ''
        
if len(diam_evals) > 0:
    print('==================================')
    print('Diamond evaluation of significance')
    print('==================================')

    headers = ['effect', 'corpus', 'diamondname', 'pair', 'estimate', 't value', 'signif', 'converged_base', 'rel_grad_base', 'converged_main', 'rel_grad_main', 'filename']

    header_row = {}
    for h in headers:
        header_row[h] = h

    rows = []

    for path in diam_evals:
        with open(path, 'rb') as f:
            filename = path.split('/')[-1]
            line = f.readline()
            while line and not deRify(line).startswith('Diamond Anova'):
                line = f.readline()
            assert line, 'Input is not properly formatted'
            diamName = val.match(line).group(1)
            while line and not deRify(line).startswith('Effect 1 ('):
                line = f.readline()
            assert line, 'Input not properly formatted'
            row = compute_row(f, diamName, 'baseline')
            row['filename'] = filename
            rows.append(row)
            while line and not deRify(line).startswith('Effect 2 ('):
                line = f.readline()
            assert line, 'Input not properly formatted'
            row = compute_row(f, diamName, 'baseline')
            row['filename'] = filename
            rows.append(row)
            while line and not deRify(line).startswith('Both vs. Effect 1'):
                line = f.readline()
            assert line, 'Input not properly formatted'
            row = compute_row(f, diamName, 'both')
            row['filename'] = filename
            rows.append(row)
            while line and not deRify(line).startswith('Both vs. Effect 2'):
                line = f.readline()
            assert line, 'Input not properly formatted'
            row = compute_row(f, diamName, 'both')
            row['filename'] = filename
            rows.append(row)

    converged = [header_row] + sorted([x for x in rows if (x['converged_base'] == 'True' and x['converged_main'] == 'True')], \
                key = lambda y: float(y['signif']))
    nonconverged = [header_row] + sorted([x for x in rows if not(x['converged_base'] == 'True' and x['converged_main'] == 'True')], \
                   key = lambda y: float(y['signif']))

    print(getPrintTable(converged, headers))

    if len(nonconverged) > 0:
        print('-----------------------------------')
        print('Convergence failures')
        print('-----------------------------------')
        print(getPrintTable(nonconverged, headers))

    print ''
    print ''   
 
