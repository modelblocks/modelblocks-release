import sys, re, argparse

argparser = argparse.ArgumentParser('''
Extracts recall values from a user-supplied list of lrtsignifs and outputs a space-delimited table of results.
''')
argparser.add_argument('lrtsignifs', type=str, nargs='+', help='One or more *.lrtsignif files from which to extract lme comparison results.')
argparser.add_argument('-D', '--diamond', dest='D', action='store_true', help='Inputs are diamond LRT files (if not specified, defaults to pairwise LRT).')
argparser.add_argument('-H', '--humanreadable', dest='H', action='store_true', help='Output column-aligned table (if not specified, defaults to CSV output).')
args, unknown = argparser.parse_known_args()

val = re.compile('^.+: *([^ "]+) *"?\n')
effectpair = re.compile('([^ ]+)-vs-([^ ]+)')

R = re.compile('(\[[0-9]+\] "?)?([^"$]*)"?')
true = ['TRUE', 'true']

def deRify(s):
    return R.match(s).group(2)

def compute_row(f, diamName=None, vs=None):
    row = {}
    line = deRify(f.readline())
    while line and not line.startswith('Main effect'):
        line = deRify(f.readline())
    assert line, 'Input not properly formatted'
    row['effect'] = val.match(line).group(1)
    line = deRify(f.readline())
    assert line.startswith('Corpus'), 'Input not properly formatted'
    row['corpus'] = val.match(line).group(1)
    line = deRify(f.readline())
    assert line.startswith('Effect estimate'), 'Input not properly formatted'
    row['estimate'] = '%.5g'%(float(val.match(line).group(1)))
    line = deRify(f.readline())
    assert line.startswith('t value'), 'Input not properly formatted'
    row['t value'] = '%.5g'%(float(val.match(line).group(1)))
    line = deRify(f.readline())
    assert line.startswith('Significance (Pr(>Chisq))'), 'Input not properly formatted'
    row['signif'] = '%.5g'%(float(val.match(line).group(1)))
    line = deRify(f.readline())
    assert line.startswith('Relative gradient (baseline)'), 'Input not properly formatted'
    row['rel_grad_base'] = '%.5g'%(float(val.match(line).group(1)))
    line = deRify(f.readline())
    if (line.startswith('Converged (baseline)')):
        row['converged_base'] = val.match(line).group(1)
        line = deRify(f.readline())
    else:
        row['converged_base'] = str(float(row['rel_grad_base']) < 0.002)
    assert line.startswith('Relative gradient (main effect)'), 'Input not properly formatted'
    row['rel_grad_main'] = '%.5g'%(float(val.match(line).group(1)))
    line = deRify(f.readline())
    if (line.startswith('Converged (main effect)')):
        row['converged_main'] = val.match(line).group(1)
        line = deRify(f.readline())
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


if args.D:
    diam_evals = [x for x in args.lrtsignifs if (x.endswith('.diamond.lrtsignif') or x.endswith('.dlrt'))]
    if len(diam_evals) > 0:
        if args.H:
            print('==================================')
            print('Diamond evaluation of significance')
            print('==================================')
            print('')

        headers = ['effect', 'corpus', 'diamondname', 'pair', 'estimate', 't value', 'signif', 'converged_base', 'rel_grad_base', 'converged_main', 'rel_grad_main', 'formname', 'lmeargs', 'filename']

        header_row = {}
        for h in headers:
            header_row[h] = h

        rows = []

        for path in diam_evals:
            with open(path, 'rb') as f:
                filename = path.split('/')[-1]
                filechunks = filename.split('.')
                formname = filechunks[-5]
                lmeargs = filechunks[-3]
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
                row['formname'] = formname
                row['lmeargs'] = lmeargs
                rows.append(row)
                while line and not deRify(line).startswith('Effect 2 ('):
                    line = f.readline()
                assert line, 'Input not properly formatted'
                row = compute_row(f, diamName, 'baseline')
                row['filename'] = filename
                row['formname'] = formname
                row['lmeargs'] = lmeargs
                rows.append(row)
                while line and not deRify(line).startswith('Both vs. Effect 1'):
                    line = f.readline()
                assert line, 'Input not properly formatted'
                row = compute_row(f, diamName, 'both')
                row['filename'] = filename
                row['formname'] = formname
                row['lmeargs'] = lmeargs
                rows.append(row)
                while line and not deRify(line).startswith('Both vs. Effect 2'):
                    line = f.readline()
                assert line, 'Input not properly formatted'
                row = compute_row(f, diamName, 'both')
                row['filename'] = filename
                row['formname'] = formname
                row['lmeargs'] = lmeargs
                rows.append(row)

        converged = [header_row] + [x for x in rows if (x['converged_base'] in true and x['converged_main'] in true)]
        nonconverged = [header_row] + [x for x in rows if not(x['converged_base'] in true and x['converged_main'] in true)]

        if args.H:
            print(getPrintTable(converged, headers))
        else:
            for r in converged:
                print(', '.join([r[h] for h in headers]))

        if len(nonconverged) > 0:
            if args.H:
                print('')
                print('')
                print('-----------------------------------')
                print('Convergence failures')
                print('-----------------------------------')
                print(getPrintTable(nonconverged, headers))
                print('')
            else:
                for r in nonconverged[int(len(converged)>0):]:
                    print(', '.join([r[h] for h in headers]))


else:
    pair_evals = [x for x in args.lrtsignifs if not (x.endswith('.diamond.lrtsignif') or x.endswith('.dlrt'))]
    if len(pair_evals) > 0:
        if args.H:
            print('===================================')
            print('Pairwise evaluation of significance')
            print('===================================')
            print('')

        headers = ['effect', 'corpus', 'estimate', 't value', 'signif', 'converged_base', 'rel_grad_base', 'converged_main', 'rel_grad_main', 'formname', 'lmeargs', 'filename']
        
        header_row = {}
        for h in headers:
            header_row[h] = h

        rows = []

        for path in pair_evals:
            with open(path, 'rb') as f:
                filename = path.split('/')[-1]
                filechunks = filename.split('.')
                formname = filechunks[-5]
                lmeargs = filechunks[-3]
                row = compute_row(f)
                row['filename'] = filename
                row['formname'] = formname
                row['lmeargs'] = lmeargs
                rows.append(row)

        converged = [header_row] + sorted([x for x in rows if (x['converged_base'] in true and x['converged_main'] in true)], \
                    key = lambda y: float(y['signif']))
        nonconverged = [header_row] + sorted([x for x in rows if not(x['converged_base'] in true and x['converged_main'] in true)], \
                       key = lambda y: float(y['signif']))

        if args.H:
            print(getPrintTable(converged, headers))
        else:
            for r in converged:
                print(','.join([r[h] for h in headers]))

        if len(nonconverged) > 1: #First element is the header row
            if args.H:
                print('')
                print('')
                print('-----------------------------------')
                print('Convergence failures')
                print('-----------------------------------')
                print('')
                print(getPrintTable(nonconverged, headers))
            else:
                for r in nonconverged[int(len(converged)>0):]:
                    print(','.join([r[h] for h in headers]))

