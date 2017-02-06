import sys
import collections
#import copy

class CueGraph(dict):

  def add(self,src,lbl,dst):
    self[src,lbl]=dst

  def readline(self,s='',linenum=0):
    ## for each dep...
    for dep in s.split():
      ## report ill-formed deps...
      if ',' not in dep:
        sys.stderr.write('WARNING: Improperly formed dependency: '+dep+'\n')
        continue

      # ## root is not meaningful in semantic graph...
      # if dep.startswith('00,1,'):
      #   continue

      ## split dependency into source, label, and destination...
      src,lbl,dst = dep.split(',')
      ## add line number to token number for unique ids withing article...
      if src!='-' and len(src)<=3 and linenum>0: src = str(linenum+1)+src
      if dst!='-' and len(dst)<=3 and linenum>0: dst = str(linenum+1)+dst
      ## add uniquified dependency...
      self.add(src,lbl,dst)


class CueGraphContexts(dict):

  def calcContexts ( self, G, Gfwd, Grev, v ):
    ## memoization...
    if v in self: return self[v]
    ## calculate type context...
    self[v] = [ G[v,'0']+'_' ] if (v,'0') in G else [ ]
    ## calculate contexts due to incoming predicate dependencies...
    self[v] += [ G[u,'0']+'_'+l for u,l in Grev[v] if l>='1' and l<='9' and (u,'0') in G ]
    ## calculate contexts derived from predicate context of incoming '~' association...
    self[v] += [ k+l for u,l in Grev[v] if l=='~' for k in self.calcContexts(G,Gfwd,Grev,u) if k[-2]=='_' ]
    ## calculate contexts derived from predicate context of outgoing predicate dependencies...
    self[v] += [ k+'-'+l for l,u in Gfwd[v] if l>='1' and l<='9' for k in self.calcContexts(G,Gfwd,Grev,u) if k[-2]!='-' and k!=G.get((v,'0'),'')+'_'+l ]
    ## conjunction inheritance...
    if (v,'c') in G: self[v] += self.calcContexts( G, Gfwd, Grev, G[v,'c'] )
    ## restriction inheritance due to predicative noun phrase...
    if (v,'r') in G: self[v] += self.calcContexts( G, Gfwd, Grev, G[v,'r'] )
    return self[v]


  def __init__(self,G):
    Gfwd = collections.defaultdict(list)
    Grev = collections.defaultdict(list)
    ## add to inheritance chain V every 1-9 dependency dest...
    for src,lbl in G:
      ## add forward dependency src -> [ (lbl,dst), (lbl',dst') ] ...
      if G[src,lbl]!='-': Gfwd[src].append( (lbl,G[src,lbl]) )
      ## add reverse dependency dst -> [ (src,lbl), (src',lbl') ] ...
      if src!='00' and lbl!='0' and G[src,lbl]!='-': Grev[G[src,lbl]].append( (src,lbl) )
    ## calc contexts using forward and reverse graphs...
    for src,lbl in G:
      self.calcContexts(G,Gfwd,Grev,src)
      if lbl!='0' and G[src,lbl]!='-': self.calcContexts(G,Gfwd,Grev,G[src,lbl])
        
  '''
  def calcInheritances ( self, G, v ):
    ## memoization...
    if v in self: return self[v]
    ## calculate local predicate context...
    self[v] =  [G[v,'0']+'_0'] if (v,'0') in G else []                                                            # from predicate itself.
    self[v] += [G.get((u,'0'),'???')+'_'+l for u,l in G if u!='00' and G[u,l]==v and l>='0' and l<='9']           # from impinging predicates.
    self[v] += [G.get((u,'0'),'???')+'_'+l+'~' for u,l in G if G.get((G[u,l],'~'),'?')==v and l>='0' and l<='9']  # from outgoing tilde-assoc deps.
    ## conjunction inheritance...
    if (v,'c') in G:    self[v] += self.calcInheritances( G, G[v,'c'] )
    ## restriction inheritance due to predicative noun phrase...
    if (v,'r') in G:    self[v] += self.calcInheritances( G, G[v,'r'] )
    return self[v]

  def __init__(self,G):
    self.V = { }
    ## add to inheritance chain V every 1-9 dependency dest...
    for src,lbl in G:
      #if src!='00' and lbl>='0' and lbl<='9' and G[src,lbl]!='-':
      if G[src,lbl]!='-':
        self.V [ G[src,lbl] ] = 1
        self.V [ src ] = 1
    ## calc contexts of each inheritance chain V...
    for v in sorted(self.V):
      self.calcInheritances(G,v)
    ## make copy of self to prevent infinite recursion...
    copyofself = copy.deepcopy(self)
    ## add second-order contexts
    for src,lbl in G:
      if src!='00' and lbl>='0' and lbl<='9' and G[src,lbl]!='-':
        ## for each context of destination...
        for cx in copyofself[G[src,lbl]]:
          if cx != "{0}_{1}".format(G.get((src,'0'),''), lbl):
            ## add to source...
            self[src] += [cx+'-'+lbl]
    for src,lbl in G:
      if (G.get((src,lbl),''), '0') in G:
        self[src] += self[G[src,lbl]]
      #if lbl=='~':
      #  ## for each context of source...
      #  for cx in copyofself[src]:
      #    ## add to destination...
      #    self[G[src,lbl]] += [cx+lbl]
    ## subtract from inheritance chain V everything with an incoming rin...
#    for src,lbl in G:
#      if lbl=='r' and G[src,lbl] in self.V:
#        del self.V[G[src,lbl]]
  '''
