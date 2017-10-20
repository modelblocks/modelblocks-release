import sys, re

strip = False
if len(sys.argv) > 1:
    strip = bool(int(sys.argv[1]))

splitter = re.compile(' *[-+|:] *')
spill = re.compile('(.+)S[0-9]+$')
pred_name = re.compile('[^(]+\((.+)\)')
#pred_name = re.compile('(.*\.\()?([^ ()]+)\)?')

def cleanParens(l):
    for i, x in enumerate(l):
        x_new = x
        if x.startswith('('):
            x_new = x[1:]
        if x.endswith(')'):
            x_new = x[:-1]
        l[i] = x_new
    return l

def getPreds(bform, exclude=['0', '1']):
    preds = ['word', 'subject', 'sentid', 'docid', 'punc', 'startoffile', 'endoffile', 'startofline', 'endofline', 'startofsentence', 'endofsentence', 'startofscreen', 'endofscreen', 'correct', 'time']
    for l in bform:
        if l.strip() != '':
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
                if strip:
                    if name.startswith('fut'):
                        name = name[3:]
                    if name.startswith('cum'):
                        name = name[3:]
                    if spill.match(name):
                        name = spill.match(name).group(1)
                if name not in exclude and name not in preds:
                    preds.append(name)
    return preds

bform = sys.stdin.readlines()
preds = getPreds(bform)
print(' '.join(preds))
