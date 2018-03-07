import re
import sys

# a Tree consists of a category label 'c' and a list of child Trees 'ch'
class Tree:

    def __init__(self,c='',ch=[], p=None, l=0, r=0):
        self.c  = c
        self.ch = ch
        self.p = p
        self.l = l
        self.r = r


    # obtain string from tree
    def __str__(self):
        if self.ch == []:
            if not hasattr(self, 'e'):
                return self.c
            else:
                return self.c + '[' + str(self.e) + ']'
        s = '(' + self.c
        if hasattr(self, 'e'):
            s += '[' + str(self.e) + ']'
        for t in self.ch:
            s += ' ' + t.__str__()
        return s + ')'


    def words(self):
        if self.ch == []:
            return [self.c]
        else:
            if type(self.ch[0]) is str:
                return self.ch
            l = []
            for t in self.ch:
                l += t.words()
            return l

    
    # obtain tree from string
    def read(self,s,fIndex=0):
        self.ch = []
        # parse a string delimited by whitespace as a terminal branch (a leaf)
        m = re.search('^ *([^ ()]+) *(.*)',s)
        if m != None:
            (self.c,s) = m.groups()
            self.l = fIndex
            self.r = fIndex
            return s, fIndex+1
        # parse nested parens as a non-terminal branch
        m = re.search('^ *\( *([^ ()]*) *(.*)',s)
        if m != None:
            (self.c,s) = m.groups()
            self.l = fIndex
            # read children until close paren
            while True:
                m = re.search('^ *\) *(.*)',s)
                if m != None:
                    return m.group(1), fIndex
                t = Tree()
                s, fIndex = t.read(s, fIndex)
                self.ch += [t]
                t.p = self
                self.r = t.r
        return ''

    # return the lowest child, but not leaf, that spans the range
    def findBySpan(self, left, right):
        t = None
        #sys.stderr.write( 'left: ' + str(left) + ' right: ' + str(right) + '\n')
        #sys.stderr.write( 's.left: ' + str(self.l) + ' s.right: ' + str(self.r) + '\n')
#        if self.l == 0 and self.r == 0:
#            for child in self.ch:
#                if type(child) is Tree:
#                    child.p = self
#            if self.p != None:
#                ind = self.p.ch.index(self)
#                if ind == 0:
#                    self.l = self.p.l
#                else:
#                    self.l = self.p.ch[ind-1].r
#            self.r = self.l + len(self.words())
        #sys.stderr.write( 's.left: ' + str(self.l) + ' s.right: ' + str(self.r) + '\n')
        if left == self.l and right == self.r:
            t = self
            #while len(t.ch) == 1 and len(t.ch[0].ch) > 0:
            #    t = t.ch[0]
            return t
        if left < self.l or right > self.r:
            return None
        for child in self.ch:
#            print child.c
            val = child.findBySpan(left, right)
            if val != None:
                return val
        return t
        raise Exception('No node span the range: ' + str(left) + ',' + str(right))
        

    def findByLeftAndHeight(self, left, height):
        t = self.findBySpan(left, left)
        while height > 0:
            t = t.p
            height -= 1
        return t  
    
    
    def sibling(self):
        if not self.p or len(self.p.ch) == 1:
            return None
        elif self.p.ch[0] == self:
            return self.p.ch[1]
        else:
            return self.p.ch[0]  
    
    def setRefs(self,ctr=0):
        if len(self.ch)==1:
            #self.e = self.ch[0].c[0]+str(ctr)
            m = re.search('\#(.)',self.ch[0].c if len(self.ch[0].ch)==0 else self.ch[0].ch[0].c if len(self.ch[0].ch[0].ch)==0 else self.ch[0].ch[0].ch[0].c)
            if m != None:
                #self.e = 'e' + str(ctr)
                self.e = m.group(1).lower() + str(ctr)
            else:
                #self.e = 'e' + str(ctr)
                self.e = (self.ch[0].c if len(self.ch[0].ch)==0 else self.ch[0].ch[0].c if len(self.ch[0].ch[0].ch)==0 else self.ch[0].ch[0].ch[0].c)[0].lower() + str(ctr)
            ctr += 1
        else:
            #if self.e: self.e = None
            for st in self.ch:
                #if re.match('-lI$',st.c):
                st.e = self.e # by default, inherit parent ref
                ctr = st.setRefs(ctr)
                if re.search('-lC',st.c) != None:
                    self.e = st.e
                else:
                    #if self.e == None and
                    if re.search('-lI',st.c) != None:
                        self.e = st.e
        return ctr

    def getAncestors(self):
        ancestors = []
        p = self  #put itself as the first element in its ancestor list
        while p != None:
            ancestors.append(p)
            p = p.p
        return ancestors
    
    def findArgBoundaries(self, predicateIdx, argHeadwordIdx, allHeadwordIdxs):
        maxProj = self.findMaxProj(predicateIdx, argHeadwordIdx, allHeadwordIdxs)
        return maxProj.l, maxProj.r
    
    def leftBoundary(self):
        l = self
        while len(l.ch) != 1:
            l = l.ch[0]
        return l.e[1:]
    
    def rightBoundary(self):
        r = self
        while len(r.ch) != 1:
            r = r.ch[1]
        return r.e[1:]
    
    def coverOtherArg(self, argHeadwordIdx, allHeadwordIdxs):
        for i in allHeadwordIdxs:
            if i != argHeadwordIdx and i >= self.l and i <= self.r:
                return True
        return False
    
    def findMaxProj(self, predicateIdx, argHeadwordIdx, allHeadwordIdxs):
        predTree = self.treeAt(predicateIdx)
        argTree = self.treeAt(argHeadwordIdx)
        predAncestors = predTree.getAncestors()
        while (not argTree.p in predAncestors) and not argTree.p.coverOtherArg(argHeadwordIdx, allHeadwordIdxs):
            argTree = argTree.p 
        return argTree
    
    def treeAt(self, idx):
        if idx < self.l or idx > self.r:
            raise 'idx is out of range'
        if self.l == self.r and self.l == idx:
            return self
        if idx <= self.ch[0].r:
            return self.ch[0].treeAt(idx)
        if idx >= self.ch[1].l:
            return self.ch[1].treeAt(idx)
        
    def toLatex(self, spaces=1, leafColor=None, refColor=None, guoColor=None, lColor=None):
        color = ("\\" + leafColor) if leafColor else ""
        rColor = ("\\" + refColor) if refColor else ""
        if refColor and self.e:
            ref = " ({" + rColor + " " + self.e + "})"
        else:
            ref = ""
        annotatedCat = self.annotateColors(guoColor, lColor)
        if len(self.ch) == 1:
            return "\lingtree{\TR{" + annotatedCat + ref + "}}{ \TR{ " + color + " " + self.ch[0].c + "} }"
        else:
            s = ""
            for i in range(spaces):
                s += "  "
            spaces += 1
            return "\lingtree{\TR{" + annotatedCat + ref + "}}{\n" + s + self.ch[0].toLatex(spaces, leafColor, refColor, guoColor, lColor) \
                + "\n" + s + self.ch[1].toLatex(spaces, leafColor, refColor, guoColor, lColor) + "\n" + s + "}" 
     
    def annotateColors(self, guoColor, lColor):
        annotatedCat = self.c
        if guoColor:
            annotatedCat = re.sub('(\-[guo][A-Z]*)', '{\\\\' + guoColor + r' \1}', annotatedCat)
        if lColor:
            annotatedCat = re.sub('(\-[l][A-Z])', '{\\\\' + lColor + r' \1}', annotatedCat) 
        return annotatedCat
    
    def str(self):
        if self.ch == []:
            return self.c
        s = '(' + self.c
        for t in self.ch:
            s = s + ' ' + t.str()
        return s + ')'

    def leaf(self):
        s = ''
        if self.ch == []:
            s= ' ' + self.c
        for t in self.ch:
                s = s + ' ' + t.leaf()
        return s

    def leafList(self):
        s = []
        if self.ch == []:
            s.append(self.c)
        for t in self.ch:
            s = s + t.leafList()
        return s

    def preterm(this):
        s = []
        if len(this.ch) == 1 and len(this.ch[0].ch) == 0:
            pret = this.c
            for t in this.ch:
                lexi = t.c
            s.append((pret,lexi))
        for t in this.ch:
            s = s + t.preterm()
        return s

    #given word index and a tree, calculate the length of path going to the word
    def depth(self, i, t0):
        depth = 0
        t = self.findBySpan(i,i)
        #print t.c
        tList = t.getAncestors()
        if t0 in tList:
            for tr in tList:
                depth += 1
                if tr == t0:
                    break
        else:
            depth = None
        return depth

    #return the path distance within a tree between two words
    #two words needs to be different
    def pathDist(self,left, right):
        dist = 0
        if left == right:
            return dist
        else:      
            t = self.findBySpan(left, left)
            if t != None:
                tList = t.getAncestors()
                d1 = 0
                d2 = 0
                for tr in tList:
                    d1 += 1
                    if right >= tr.l and right <= tr.r:
                        d2 = self.depth(right, tr)
                        break
#                print "d1", d1
#                print "d2", d2
                dist = d1 + d2 
                return dist
            else:
                return dist

    def cats(self):
        ct = []
        if self.ch != []:
            if self.c not in ct:
                ct.append(self.c)
        for ch in self.ch:
            ct = ct + ch.cats()
        return ct
                    
    
        
#line = "(S (NP-SBJ (NP (NNP Pierre) (NNP Vinken) )(, ,)(ADJP (NP (CD 61) (NNS years) )(JJ old) )(, ,) )(VP (MD will)(VP (VB join)(NP (DT the) (NN board) )(PP-CLR (IN as)(NP (DT a) (JJ nonexecutive) (NN director) ))(NP-TMP (NNP Nov.) (CD 29) )))(. .))"

#if line != '':
#    t=Tree()
#    t.read(line)
#    rawS = re.sub("\*[^ ]*", "", t.leaf())
#    s = re.sub(" +", " ",rawS)
#    print(t.leaf())
#    print(rawS)
#    print(s)
#    for ch in t:
#        print type(ch)
#        print ch.c

#t=Tree()
#t.read(line)
#print t.cats()
#print t.pathDist(7,8)
#print str(t)
#print t.leafList()
#print t.preterm()
#for i in t.ch:
#    print "parent cat", t.c
#    for j in i.ch:
#        print j.c
#    print "child cat", i.c
#    print "left", i.l
#    print "right", i.r
#    print "parent cat", i.p.c
#    print "depth", t.depth(3, t)
#    print "lowest child", t.findBySpan(0,1).c
#    print "span", t.findBySpan(11,14)
    
 #   for j in i.ch:
 #       if j.c == "ADJP":
 #           ancs = j.getAncestors()
 #           if ancs != None:
 #               for n in ancs:
 #                   print n.c

#    print i.leftBoundary()
#    print i.rightBoundary()
    
#print t.treeAt(3).str()
##print t.str()
#for i in t.ch:
##    print i.c
#    if i.c == "NP-SBJ":
#        sbj = i
#        for n in i.ch:
#            if n.c == "NP":
#                acst = n.getAncestors()
#                ccom = []
#                for a in acst:
#                    print "ancestero of NP", a.c
#                    print type(a.sibling())
#                    if a.sibling() != None:
#                        ccom.append(a.sibling())
#                print len(ccom)
#                for c in ccom:
#                    print c.c
#        
#print sbj.str()
#for i in range (0, len(t.ch)):
#    if t.ch[i].c == "VP":
#        t.ch[i] = sbj
#
#print t.str()
##print t.str()
#print t.leaf()
#print type(t.ch)
#clist = []
#for ch in t.ch:
#        clist.append(ch.c)
#
#print clist
#            
