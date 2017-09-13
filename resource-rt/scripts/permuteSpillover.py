import sys, os, re, itertools, argparse

argparser = argparse.ArgumentParser('''
Takes a mixed effects model as input and computes spillover permutations
''')
argparser.add_argument('inform')
argparser.add_argument('outdir')
argparser.add_argument('-m', '--min', type=int, default=0)
argparser.add_argument('-M', '--max', type=int, default=1)
argparser.add_argument('-D', '--dryrun', action='store_true')
argparser.add_argument('exclude', type=str, nargs='*', default=['0', '1', 'subject'])
args, unknown = argparser.parse_known_args()

exclude = set(args.exclude)
splitter = re.compile(' *[-+|:] *')
pred_name = re.compile('[^(]+\((.+)\)')
src_name = re.compile('.*/([^/]+).lmeform')

def cleanParens(l):
    for i, x in enumerate(l):
        x_new = x
        if x.startswith('('):
            x_new = x[1:]
        if x.endswith(')'):
            x_new = x[:-1]
        l[i] = x_new
    return l

def getPreds(bform):
    preds = set()
    for l in bform[1:]:
        if l.startswith('('):
            l = splitter.split(l.strip())
            l_new = cleanParens(l)
            p_list = l
        else:
            p_list = splitter.split(l.strip())
        for p in p_list:
            name = p
            while pred_name.match(name):
                name = pred_name.match(name).group(1)
            if name not in exclude and name not in preds:
                preds.add(name)
    return preds

def updatePreds(preds, perm):
    preds_new = preds[:]
    for i,x in enumerate(preds):
        if perm[i] > 0:
            preds_new[i] = x + 'S%d' %perm[i]
    return preds_new

def printPermPreds(bform, src_form_name, preds, perms, outdir, dryrun=True):
    for perm in perms:
        form_name = []
        for i in xrange(len(preds)):
            form_name.append(preds[i][:2] + str(perm[i]))
        form_name = ''.join(form_name) + src_form_name + 'SP'
        bform_out = bform[:1]
        preds_new = updatePreds(preds, perm)
        for l in bform[1:]:
            for i in xrange(len(preds)):
                l = re.sub(r'([+^ (])'+preds[i]+'([+$ )])', r'\1'+preds_new[i]+r'\2', l)
            bform_out.append(l)
        path = outdir + '/' + form_name + '.lmeform'
        if not dryrun and not os.path.exists(path):
            with open(path, 'wb') as f:
                f.write(''.join(bform_out))
        sys.stdout.write('%s ' %form_name)

src_path = args.inform
with open(src_path, 'rb') as f:
    bform = f.readlines()

src_form_name = src_name.match(src_path).group(1)

if len(bform) == 0:
    exit()

preds = getPreds(bform)
preds = list(preds)
preds.sort(reverse=True, key= lambda x: len(x))
n_pred = len(preds)

perms = list(itertools.product(range(args.min, args.max+1), repeat=n_pred))

printPermPreds(bform, src_form_name, preds, perms, args.outdir, args.dryrun)

