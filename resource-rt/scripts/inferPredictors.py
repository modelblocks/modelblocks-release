import sys, re

splitter = re.compile(' *[-+|] *')
pred_name = re.compile('(.*\.\()?([^ ()]+)\)?')

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
    preds = set(['subject','sentid', 'punc', 'startoffile', 'endoffile', 'startofline', 'endofline', 'startofsentence', 'endofsentence', 'startofscreen', 'endofscreen'])
    for l in bform:
        if l.startswith('('):
            l = splitter.split(l.strip())
            l_new = cleanParens(l)
            p_list = l
        else:
            p_list = splitter.split(l.strip())
        for p in p_list:
            name = pred_name.match(p).group(2)
            if name not in exclude and name not in preds:
                preds.add(name)
    return preds

bform = sys.stdin.readlines()
preds = getPreds(bform)
print(' '.join(preds))
