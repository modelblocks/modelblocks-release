import sys, re, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
import tree

ops = {}
ops['a'] = ['bwd', 'A']
ops['b'] = ['fwd', 'A']
ops['g'] = ['bwd', 'N']
ops['h'] = ['fwd', 'N']
ops['c'] = ['bwd', 'C']
ops['d'] = ['fwd', 'C']

casts = {}
casts['N'] = ['R', 'A', 'S']
casts['A'] = ['R', 'S']
casts['G'] = ['N', 'S']
casts['I'] = ['R-aN', 'A-aN', 'N', 'S', 'V-iN']
casts['V'] = ['C', 'Q', 'S']
casts['E'] = ['N', 'S']
casts['C'] = ['N']
casts['S'] = ['N']



def argsFromString(string):
    if string[-1] != '-':
        string += '-' #the hyphen triggers a push to the list, so this ensures last arg is pushed
    argMap = {}
    currentOp = 'head'
    currentArg = ''
    braceDepth = 0
    
    for char in string:
        if char == '-' and braceDepth == 0:
            if currentArg != '':
                if currentOp in argMap:
                    argMap[currentOp].append(stripBraces(currentArg))
                else:
                    argMap[currentOp] = [stripBraces(currentArg)]
            currentOp = ''
            currentArg = ''
        else:
            if currentOp == '':
                currentOp = char
            else:
                if char == '{':
                    braceDepth += 1
                if char == '}':
                    braceDepth -= 1
                currentArg += char
    return argMap
    
def checkCasts(src, dest):
    if src in casts:
        for cast in casts[src]:
            if re.match(cast, dest):
                return True
    if re.match('S', dest):
        return True
    return False

def parseType(t, strip):
    toParse = ''
    if strip:
        toParse = stripOps(t.c)
    else:
        toParse = t.c
    if len(t.ch) > 0:   #Avoid "parsing" hyphenate leaves
        return argsFromString(toParse)
    else:
        return {'head': [t.c]}
    
def stripBraces(string):
    if string[0] == '{':
        return string[1:-1]
    else:
        return string
        
def stripOps(string):
    return re.sub('-x|-o|-v|-pP[cs]|-l[^- ()]+', '', string)
        
def printArgMap(t):
    print(parseType(t, False))
    for child in t.ch:
        printArgMap(child)
        
def discharge(pred, op, type):
    discharged = False
    predType = parseType(pred, True)
    
    if op in predType:
        if type in predType[op]:
            tType = parseType(pred.p, True)
            argString = '-' + op
            if len(type) == 1 or type[0] == 'P' or type == 'Ne':
                argString += stripOps(type)
            else:
                argString += '{' + type + '}'
            mergedCat = stripOps(re.sub(argString, '', stripOps(pred.c), count=1)) + stripOps(re.sub('^' + stripOps(type), '', pred.sibling().c))
            
            if mergedCat == stripOps(pred.p.c):
                discharged = True
            # else:
                # print argString
                # print mergedCat
                # print stripOps(pred.p.c)
                # print ''
            
    
            if not discharged and 'g'in tType:
                discharged = queryGap(stripOps(pred.p.c), mergedCat, 'g', 'a')
                if not discharged:
                    discharged = queryGap(stripOps(pred.p.c), mergedCat, 'g', 'b')
                if not discharged:
                    tmpType = deepCopyArgMap(tType)
                    del tmpType['g']
                    if argsFromString(mergedCat) == tmpType:
                        discharged = True
                    
            if not discharged and 'h' in tType:
                discharged = queryGap(stripOps(pred.p.c), mergedCat, 'h', 'a')
                if not discharged:
                    discharged = queryGap(stripOps(pred.p.c), mergedCat, 'h', 'b') 
                    if not discharged:
                        tmpType = deepCopyArgMap(tType)
                        del tmpType['h']
                        if argsFromString(mergedCat) == tmpType:
                            discharged = True
                        
            if not discharged:
                discharged = checkCasts(predType['head'][0], pred.p.c)

            if not discharged and predType['head'][0] == 'V' \
                and re.match('([VC]-rN|N)', pred.p.c) \
                and 'g' in predType:
                discharged = True
                
            if not discharged and 'c' in tType:
                coordCat = mergedCat
                if len(coordCat) > 1:
                    coordCat += '-c{' + coordCat + '}'
                else:
                    coordCat += '-c' + coordCat
                if stripOps(pred.p.c) == coordCat:
                    discharged = True
                if not discharged:
                    print stripOps(pred.p.c)
                    print coordCat
                
    return discharged
        
def queryGap(gapCat, targCat, gapOp, op):
    braceDepth = 0
    gapCat = stripOps(gapCat)
    targCat = stripOps(targCat)
    for i in range(0, len(gapCat)):
        char = gapCat[i]
        if char == '{':
            braceDepth += 1
        elif char == '}':
            braceDepth -= 1
        elif braceDepth == 0 and char == gapOp:
            gapCat = gapCat[:i] + op + gapCat[i+1:]
    return gapCat == targCat
    
def deepCopyArgMap(argMap):
    tmp = {}
    for key in argMap:
        tmp[key] = argMap[key][:]
    return tmp
            
def rmGap(dict, op):
    for i in range(0, len(dict[op])):
        arg = dict[op].pop(i)
        if 'g'in dict:
            dict['g'].append(arg)            
        
def reannotate(t):
    opMap = parseType(t, False)
    cont = False
    #Binary tree check
    if (len(t.ch) < 2):
        cont = True
    if (len(t.ch) > 2):
        for i in range(2, len(t.ch)):
            t.ch[i].c += '***Too many children***'
        cont = True

    #Check for rule applications
    if not cont:
        if t.p == None:
            addRule(t, 'S')
            cont = True
        
    if not cont:
        if t.ch[0].c == 'X-cX-dX':
            target = opMap['c'][0]
            if re.match(target, t.ch[1].c):
                addRule(t.ch[1], 'C')
                cont = True
                
    if not cont:
        if stripOps(t.ch[1].c) == 'C-rN' and re.match('N', t.ch[0].c):
            addRule(t.ch[1], 'N')
            cont = True
    
    if not cont:
        if 'c' in parseType(t.ch[1], False):
            cont = parseOp('c', t.ch[1])
            if not cont:
                addRule(t.ch[0], 'M')
                cont = True
                
    if not cont:
        if re.match('A-aN-x|R-aN-x', t.ch[0].c) or re.match('A-aN-x|R-aN-x', t.ch[1].c):
            addRule(t.ch[0], 'M')
            cont = True
        
    if not cont:
        cont = parseOp('b', t.ch[0])
        
    if not cont:
        cont = parseOp('h', t.ch[0])
    
    if not cont:
        modType = re.match('(A-aN|R-aN)', t.ch[0].c)
        if modType:
            if not re.match(modType.group(1)[0], t.c[0]):
                sibType = parseType(t.ch[1], False)
                if 'a' in sibType:
                    if not re.match(sibType['a'][-1], t.ch[0].c):
                        addRule(t.ch[0], 'M')
                        cont = True
                else:
                    addRule(t.ch[0], 'M')
                    cont = True

    if not cont:
        modType = re.match('(A-aN|R-aN)', t.ch[1].c)
        if modType:
            if not re.match(modType.group(1)[0], t.c[0]):
                sibType = parseType(t.ch[0], False)
                if 'b' in sibType:
                    if not re.match(sibType['b'][-1], t.ch[1].c):
                        addRule(t.ch[1], 'M')
                        cont = True     
                else:
                    addRule(t.ch[1], 'M')
                    cont = True

    if not cont:
        if stripOps(t.c) == stripOps(t.ch[0].c):
            addRule(t.ch[1], 'M')
            cont = True

    if not cont:
        if stripOps(t.c) == stripOps(t.ch[1].c):
            addRule(t.ch[0], 'M')
            cont = True

    if not cont:
        cont = parseOp('a', t.ch[1])
        
    if not cont:
        cont = parseOp('g', t.ch[1])
        
    if not cont:
        if checkCasts(t.ch[0].c[0], t.c[0]):
            addRule(t.ch[1], 'M')
            cont = True
        
    if not cont:
        if checkCasts(t.ch[1].c[0], t.c[0]):
            addRule(t.ch[0], 'M')
            cont = True
        
    if not cont:
        if re.match('V-(r|i)N', t.ch[1].c):
            addRule(t.ch[1], 'N')
            cont = True

    if not cont:
        if re.match('V-(r|i)N', t.ch[0].c):
            addRule(t.ch[0], 'N')
            cont = True
            
    if not cont:
        addRule(t.ch[0], '*')
        addRule(t.ch[1], '*')

    for child in t.ch:
        reannotate(child)
        
def parseOp(op, t):
    ruleApp = False
    opMap = parseType(t, True)
    if op in opMap:
        opArgs = opMap[op]
        if len(opArgs) == 1:
            ruleApp = parseRule(op, opArgs[0], t.p)
        else:
            for i in range(len(opArgs) - 1, -1, -1):
                ruleApp = parseRule(op, opArgs[i], t.p)
                if ruleApp:
                    break
    return ruleApp
                    
def parseRule(op, type, t):
    ruleApp = False
    pred = t.ch[0]
    arg = t.ch[1]
    
    if (ops[op][0] == 'bwd'):
        pred = t.ch[1]
        arg = t.ch[0]
        
    if re.match(type, arg.c):
        ruleApp = True
        if discharge(pred, op, type):
            addRule(arg, ops[op][1])
        else:
            addRule(arg, ops[op][1])
            addRule(t, '*TC')
            
    return ruleApp
        
def isTerminal(t):
    if len(t.ch) == 1:
        if len(t.ch[0].ch) == 0:
            return True
    return False
        
def addRule(t, rule):
    if isTerminal(t):
        newTree = tree.Tree()
        tmp = t.ch
        t.ch = []
        newTree.c = t.c
        newTree.ch = tmp
        t.ch.append(newTree)
        t.c += '-l' + rule
    else:
        t.c += '-l' + rule

for line in sys.stdin:
    if (line.strip() !='') and (line.strip()[0] != '%'):
        inputTree = tree.Tree()
        inputTree.read(line)
        reannotate(inputTree)
        print(inputTree)
    
