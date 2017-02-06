##############################################################################
##                                                                           ##
## This file is part of ModelBlocks. Copyright 2009, ModelBlocks developers. ##
##                                                                           ##
##    ModelBlocks is free software: you can redistribute it and/or modify    ##
##    it under the terms of the GNU General Public License as published by   ##
##    the Free Software Foundation, either version 3 of the License, or      ##
##    (at your option) any later version.                                    ##
##                                                                           ##
##    ModelBlocks is distributed in the hope that it will be useful,         ##
##    but WITHOUT ANY WARRANTY; without even the implied warranty of         ##
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          ##
##    GNU General Public License for more details.                           ##
##                                                                           ##
##    You should have received a copy of the GNU General Public License      ##
##    along with ModelBlocks.  If not, see <http://www.gnu.org/licenses/>.   ##
##                                                                           ##
###############################################################################

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

########################################
#
#  CondModel class
#
########################################

# define model to map condition tuples to distributions
class CondModel(dict):
    # init with model id
    def __init__ ( self, i = '' ):
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


