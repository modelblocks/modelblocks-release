import re

########################################
#
#  Model class
#
########################################

# define distribution to map value tuples to probabilities (or frequencies or scores, if not normalized)
class Model(dict):
    # init with model id
    def __init__(self,i=''):
        self.id = i
        self.regex = re.compile('^ *'+self.id+' +: +(.*) += +(.*) *$')
    # define as promiscuous dictionary: populate with default values when queried on missing keys
    def __missing__(self,k):
        self[k]=0.0
        return self[k]
    # define get without promiscuity
    def get(self,k):
        return dict.get(self,k,0.0)
    # normalize to make consistent probability distribution
    def normalize(self):
        tot = 0.0
        for v in self: tot += self[v]
        if tot > 0.0:
            for v in self: self[v] /= tot
    # read model
    def read(self,s):
        m = self.regex.match(s)
        if m is not None:
            v = tuple(re.split(' +',m.group(1)))
            if len(v)==1: v = v[0]
            self[v] = float(m.group(2))
            return 1
        return 0
    # write model
    def write(self):
        for v in sorted(self):
            if self[v]>0.0:
                print(self.id, end=' ')
                print(':', end=' ')
                if type(v) is tuple:
                    for f in v:
                        print(f, end=' ')
                else: print(v, end=' ')
                print('=',self[v])
    # kneser-ney smoothing # l is an empty list (to allow recursive iteration from CondModels) # smooth is a dummy flag for similar reasons
    def kneserNey(self,l,smooth=False):
        n1,n2,n3,n4 = 0,0,0,0
        for v in self:
            if self[v] == 1:
                n1 += 1
            elif self[v] == 2:
                n2 += 1
            elif self[v] == 3:
                n3 += 1
            elif self[v] == 4:
                n4 += 1
#        if n1 == 0:
#            n1 = 1
#        if n2 == 0:
#            n2 = 1
#        if n3 == 0:
#            n3 = 1
#        if n4 == 0:
#            n4 = 1
        Y = n1 / (n1 + 2 * n2)
        if n1 == 0:
          D1 = 1 - 2 * Y * n2 / 1
        else:
          D1 = 1 - 2 * Y * n2 / n1
        if n2 == 0:
          D2 = 2 - 3 * Y * n3 / 1
        else:
          D2 = 2 - 3 * Y * n3 / n2
        if n3 == 0:
          D3 = 3 - 4 * Y * n4 / 1
        else:
          D3 = 3 - 4 * Y * n4 / n3

        tot = 0.0
#        N1 = 0
#        N2 = 0
#        N3 = 0
        for v in self:
            tot += self[v]
#            if self[v] == 1:
#                N1 += 1
#            elif self[v] == 2:
#                N2 += 1
#            elif self[v] >= 3:
#                N3 += 1

        #gamma = (D1 * N1 + D2 * N2 + D3 * N3) / tot

        for v in self:
            if self[v] == 0:
                D = 0
            elif self[v] == 1:
                D = D1
            elif self[v] == 2:
                D = D2
            else:
                D = D3

            self[v] = (self[v] - D) / tot #+ gamma * self[v]/tot #gamma factor only affects context


########################################
#
#  CondModel class
#
########################################

# define model to map condition tuples to distributions
class CondModel(dict):
    # init with model id
    def __init__(self,i):
        self.id = i
        self.regex = re.compile('^ *'+self.id+' +(.*) +: +(.*) += +(.*) *$')
    # define as promiscuous dictionary: populate with default values when queried on missing keys
    def __missing__(self,k):
        self[k]=Model()
        return self[k]
    # define get without promiscuity
    def get(self,k):
        return dict.get(self,k,Model())
    # normalize to make consistent probability distribution
    def normalize(self):
        for c in self:
            tot = 0.0
            for v in self[c]: tot += self[c][v]
            if tot > 0.0:
                for v in self[c]: self[c][v] /= tot
    # read model
    def read(self,s):
        m = self.regex.match(s)
        if m is not None:
            c = tuple(re.split(' +',m.group(1)))
            if len(c)==1: c = c[0]
            v = tuple(re.split(' +',m.group(2)))
            if len(v)==1: v = v[0]
            self[c][v] = float(m.group(3))
            return 1
        return 0
    # write model
    def write(self):
        for c in sorted(self):
            for v in sorted(self[c]):
                if self[c][v]>0.0:
                    print(self.id, end=' ')
                    if type(c) is tuple:
                        for f in c:
                            print(f, end=' ')
                    else: print(c, end=' ')
                    print(':', end=' ')
                    if type(v) is tuple:
                        for f in v:
                            print(f, end=' ')
                    else: print(v, end=' ')
                    print('=',self[c][v])
    # obtain size
    def size(self):
        k = 0
        for c in self:
            k += len(self[c])
        return k
    # remove a constituent... weird but necessary
    def removeAll(self,s):
        for k in list(self.keys()):
            if k.find(s) != -1:
                del self[k]
    # kneser-ney smoothing # l is an ordered list of models with decreasing numbers of conditioned vars from most to least
    #   if current CondModel is a trigram model, l will consist of a bigram model followed by a unigram model
    def kneserNey(self,l,smooth=False):
        #recursively smooth lower-order ngrams if needed
        if smooth:
          l[0].kneserNey(l[1:],smooth)

        n1,n2,n3,n4 = 0,0,0,0

#        tot = 0.0
        for c in self:
            for v in self[c]:
#                tot += self[c][v]
                if self[c][v] == 1:
                    n1 += 1
                elif self[c][v] == 2:
                    n2 += 1
                elif self[c][v] == 3:
                    n3 += 1
                elif self[c][v] == 4:
                    n4 += 1
        if n1 == 0:
            n1 = 1
        if n2 == 0:
            n2 = 1
        if n3 == 0:
            n3 = 1
        Y = n1 / (n1 + 2 * n2)
        D1 = 1 - 2 * Y * n2 / n1
        D2 = 2 - 3 * Y * n3 / n2
        D3 = 3 - 4 * Y * n4 / n3

        for c in self:
            tot = 0.0
            N1 = 0
            N2 = 0
            N3 = 0
            for v in self[c]:
                tot += self[c][v]
                if self[c][v] == 1:
                    N1 += 1
                elif self[c][v] == 2:
                    N2 += 1
                elif self[c][v] >= 3:
                    N3 += 1

            gamma = (D1 * N1 + D2 * N2 + D3 * N3) / tot

            for v in self[c]:
                if self[c][v] == 0:
                    D = 0
                elif self[c][v] == 1:
                    D = D1
                elif self[c][v] == 2:
                    D = D2
                else:
                    D = D3

                if len(l) > 1:
                  self[c][v] = (self[c][v] - D) / tot + gamma * l[0][c[1:]][v]
                else:
                  self[c][v] = (self[c][v] - D) / tot + gamma * l[0][v]


########################################
#
#  ListModel class
#
########################################

# define list model
class ListModel(dict):
    # init with model id
    def __init__(self,i):
        self.id = i
        self.regex = re.compile('^ *'+self.id+' +(.*) +: +(.*) += +(.*) *$')
    # define as promiscuous dictionary: populate with default values when queried on missing keys
    def __missing__(self,k):
        self[k] = []
        return self[k]
    # define get without promiscuity
    def get(self,k):
        return dict.get(self,k,[])
#     # normalize to make consistent probability distribution
#     def normalize(self):
#         for c in self:
#             tot = 0.0
#             for prv in self[c]: tot += prv[0]
#             if tot > 0.0:
#                 for prv in self[c]: prv[0] /= tot
    # sort
    def sort(self):
        for c in self:
            self[c].sort ( reverse=True )
    # read model
    def read(self,s):
        m = self.regex.match(s)
        if m is not None:
            c = tuple(re.split(' +',m.group(1)))
            if len(c)==1: c = c[0]
            v = tuple(re.split(' +',m.group(2)))
            if len(v)==1: v = v[0]
            self[c].append ( (float(m.group(3)),v) )
            return 1
        return 0
    # write model
    def write(self):
        for c in sorted(self):
            for pr,v in self[c]:
                if pr>0.0:
                    print(self.id, end=' ')
                    if type(c) is tuple:
                        for f in c:
                            print(f, end=' ')
                    else: print(c, end=' ')
                    print(':', end=' ')
                    if type(v) is tuple:
                        for f in v:
                            print(f, end=' ')
                    else: print(v, end=' ')
                    print('=',pr)


