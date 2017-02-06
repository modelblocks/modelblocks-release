# Reads a depth-annotated linetree file and prints parser tokens flagged with
# various information about center-embedding. Includes "integration cost"
# as defined in Gibson (2000), which is sensitive to argument co-indexation
# and counts intervening referents, as well as a purely syntactic measure
# of center-embedding, such that a center-embedding is taken to be any
# complex left child of a right child. Columns identifying the relevant 
# predictors are printed as header labels in the output.

import re
import sys
import argparse
sys.path.append('../resource-gcg/scripts/')
import tree

argparser = argparse.ArgumentParser()
argparser.add_argument('-d', '--debug', dest='DEBUG', action='store_true')
args, unknown = argparser.parse_known_args()

depth = re.compile('[^ ()]+-b(L|R)-d([0-9]+)')
embeddings = []
terminals = []
isPreviousPunc = False
nonToks = ["-LRB-", "-RRB-", "\'s", "\'d", "n\'t", "\'ll", "\'m", "\'re", "\'ve", ",", \
        "-", ";", ":", "\'", '\"', "`", "``", ".", "!", "?", "Y\'", "y\'"]
#List of categories that introduce discourse referents per Gibson (2000),
#which assumes new referents for all "verbs" and non-pronominal NPs.
#Should non-finite forms 'B' and 'L' be included here as well?
pronouns = ['I', 'he', 'she', 's/he', 'it', 'we', 'they', 'you', 'me', 'him', 'her', 'us', 'them', \
        'who', 'whom', 'which', 'that', 'myself', 'himself', 'herself', 'itself', 'ourselves', 'themselves', 'oneself']
sentid = 0

def argsFromString(string):
    if string[-1] != '-':
        string += '-' #the hyphen triggers a push to the list, so this ensures last arg is pushed
    argMap = {}
    currentOp = 'head'
    if string[0:5] == '-LRB-' or string[0:5] == '-RRB-':
        argMap[currentOp] = [string[0:5]]
        string = string[5:]
        currentOp = ''
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

def stripBraces(string):
    if string[0] == '{':
        return string[1:-1]
    else:
        return string
 
#Return True if tree is complex, false otherwise      
def isComplex(tree):
    if len(tree.ch) == 2:
        return True
    for subtree in tree.ch:
        return isComplex(subtree)
    return False

def isPunc(t):
    if len(t.ch) == 0:
        if t.c not in nonToks:
            return False
        return True
    punc = True
    for c in t.ch:
        punc = punc & isPunc(c)
    return punc
        
#Return True if tree is a center-embedding, false otherwise
def isCenterEmbedding(tree):
    if isComplex(tree):
        if tree.sibling() != None:
            if not isPunc(tree.sibling()):
                if depth.search(tree.c).group(2) > 1:
                    if int(depth.search(tree.c).group(2)) > int(depth.search(tree.p.ch[1].c).group(2)):
                        if args.DEBUG:
                            print '====='
                            print 'Embedded region found:'
                            print tree
                            print '====='
                        return True
    return False
    
#Return True if test is at the left edge of target, false otherwise
def isAtLeftEdgeOf(test, target):
    if test == target:
        return True
    if test.p != None:
        if test.p.ch[0] == test:
            if test.p == target:
                return True
            else:
                return isAtLeftEdgeOf(test.p, target)
    return False

# Return the last terminal label + leaf of a tree
def lastTerm(t, ignorePunc):
    if len(t.ch) == 1:
        if t.ch[0].ch == []:
            return t
        return lastTerm(t.ch[0], ignorePunc)
    if ignorePunc:
        if t.ch[-1].ch[0].c in nonToks:
            return lastTerm(t.ch[-2], ignorePunc)
    return lastTerm(t.ch[-1], ignorePunc)
    
#Requires tree is terminal label + leaf. Return 1 if at start of center-embedding,
#false otherwise
def startOfEmbd(tree):
    global isPreviousPunc
    startEmbd = 0
    if embeddings != []:
        if isAtLeftEdgeOf(tree, embeddings[-1]) or isPreviousPunc:
            if not tree.ch[0].c in nonToks:
                startEmbd = 1
    return startEmbd

def preFoot(t):
    next = getNextTerm(t)
    if next == lastTerm(embeddings[-1], True) and next.ch[0].c == '*FOOT*':
        return True
    return False 

#Requires tree is terminal label + leaf. Return 1 if at end of center-embedding,
#false otherwise
def endOfEmbd(tree): 
    embdLen = 0
    embdRule = None
    if embeddings != []:
        if tree == lastTerm(embeddings[-1], True) or preFoot(tree):
            embdLen = len([word for word in embeddings[-1].words() if word not in nonToks]) - 1
            embdRule = getRule(embeddings[-1])
    return embdLen, embdRule

def getRule(t):
    label = argsFromString(t.c)
    sibLabel = argsFromString(t.sibling().c)
    if 'l' in label:
        rule = label['l']
        return rule[0] + 'b'
    elif 'l' in sibLabel:
        rule = sibLabel['l']
        return rule[0] + 'a'
    elif 'e' in label:
        return 'FGtrans'
    else:
        return None
    
#Requires tree is terminal label + leaf. Return 1 if at start of
#immediate right sibling of center-embedding, false otherwise
def startOfPostEmbd(tree):
    global isPreviousPunc
    embdLen = 0
    if embeddings != []:
        if isAtLeftEdgeOf(tree, embeddings[-1].sibling()) or isPreviousPunc:
            if not (tree.ch[0].c in nonToks):
                embdLen = len([word for word in embeddings[-1].words() if word not in nonToks]) - 1
                isPreviousPunc = False
                embeddings.pop()
            else:
                isPreviousPunc = True
    return embdLen
    
#Return the first terminal label + leaf node of a tree
def getFirst(tree):
    if len(tree.ch) == 1 and tree.ch[0].ch == []:
        return tree
    return getFirst(tree.ch[0])
 
#Return the last terminal label + leaf node of a tree
def getLast(tree):
    if len(tree.ch) == 1 and tree.ch[0].ch == []:
        return tree
    return getLast(tree.ch[-1])
    
def getNextTerm(tree):
    if not tree.p:
        return None
    if len(tree.p.ch) > 1 and tree == tree.p.ch[0]:
        return getFirst(tree.p.ch[1])
    return getNextTerm(tree.p)
    
#Return true if tree has a left sibling, false otherwise
def hasSibling(tree):
    if tree.p is not None and len(tree.p.ch) > 1 and tree.p.ch[0] == tree.sibling():
        return True
    return False

#Print table of tokens with syntactic flags
def printWithFlags(tree):
    global isPreviousPunc, sentid
    if len(tree.ch) == 1 and tree.ch[0].ch == []:
        embdLenAfter = startOfPostEmbd(tree)
        startPostEmbd = 0
        startPostEmbdp1 = 0
        if embdLenAfter > 0:
            startPostEmbd = 1
            startPostEmbdp1 = embdLenAfter + 1
        startEmbd = startOfEmbd(tree)
        embdLenBefore, embdRule = endOfEmbd(tree)
        if embdLenBefore > 0:
            endEmbd = 1
        else:
            endEmbd = 0
        terminals.append(tree)
        if (tree.ch[0].c != '*FOOT*'):
            print (tree.ch[0].c + " " + str(sentid) + " " + str(startEmbd)
                   + " " + str(endEmbd) + " " + str(embdLenBefore)
                   + " " + str(startPostEmbd) + " " + str(startPostEmbdp1) + " "
                   + str(embdLenAfter) + " " + str(embdRule) + " " + str(len(embeddings)))
    else:
        for subtree in tree.ch:
            #Makes sure center-embeddings get pushed to the stack in the right order,
            #especially in the case when a word begins a center-embedding and
            #immediately follows a preceding one.
            if isCenterEmbedding(subtree):
                if embeddings == [] or embeddings[-1] in subtree.getAncestors():
                    embeddings.append(subtree)
                else:
                    embeddings.insert(-1, subtree)
            printWithFlags(subtree)
            


#Main program
print ("word sentid startembd endembd embdLenBefore startPostEmbd"
       + " startPostEmbdp1 embdLenAfter embdRule embddepth")


for line in sys.stdin:
    if args.DEBUG:
        if (len(embeddings) > 0):
            print "EMBEDDINGS DID NOT CLOSE"
    embeddings = []
    terminals = []
    isPreviousPunc = False
    if (line.strip() !='') and (line.strip()[0] != '%'):
        inputTree = tree.Tree()
        inputTree.read(line)
        printWithFlags(inputTree)
        sentid += 1
