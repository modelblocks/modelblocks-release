import fileinput
import re
import tree
import operator
#import model
#import quants
#import argparse
import sys
import argparse


parser = argparse.ArgumentParser(description='Print eventuality (semantic) dependencies for Chinese GCG tree.')
parser.add_argument('-d', '--debug', default=False, required=False, action='store_true', help='print debug info')
parser.add_argument('-t', '--tree', default=False, required=False, action='store_true', help='enforce tree restriction (for scoring)')
opts = parser.parse_args()



#traverse the tree top-down to read out head for each span of the tree
#output into a dic={(0,3):3, (1,5):3...}
#use span as key and head idx as value
def getHead(t, dic, leafList, heads):
#  print t.l
#  print t.l
  heads.append((t.l,t.r))

  #reach the terminals 
  if len(t.ch) == 0:
    if heads != []:
      for h in heads:
        dic[h] = (t.l, leafList[t.l])
    dic[(t.l,t.r)] = (t.l, leafList[t.l])
    
  #unary branch
  elif len(t.ch) == 1:
    getHead(t.ch[0], dic, leafList, heads)
    
  #binary branch
  elif len(t.ch) == 2:
    leftC = t.ch[0].c
    rightC = t.ch[1].c
    leftList = re.split("-", t.ch[0].c)
    rightList = re.split("-", t.ch[1].c)
    leftCats = decompCat(t.ch[0].c)
    rightCats = decompCat(t.ch[1].c)
#    print ("left", leftCats)
#    print ("right", rightCats)

    #right branching for parsing mistakes
    if t.ch[0].c == t.ch[1].c:
      getHead(t.ch[0], dic , leafList, heads=[])
      getHead(t.ch[1], dic, leafList, heads)
    #modification
    if leftC == "A-bN" or leftC == "R-bV" or rightC == "A-bN" or rightC == "R-bV":
      if leftC == "A-bN":
        getHead(t.ch[0], dic, leafList, heads=[])
        getHead(t.ch[1], dic, leafList, heads)
      elif rightC == "A-bN":
        getHead(t.ch[0], dic, leafList, heads)
        getHead(t.ch[1], dic, leafList, heads=[])
      elif leftC == "R-bV":
        getHead(t.ch[0], dic, leafList, heads=[])
        getHead(t.ch[1], dic, leafList, heads)
      elif rightC == "R-bV":
        getHead(t.ch[0], dic, leafList, heads)
        getHead(t.ch[1], dic, leafList, heads=[])
        
    #make Q-bN as modifier to match chinese semeval format
    elif leftC == "Q-bN" and rightC == "N":
      getHead(t.ch[0], dic, leafList, heads=[])
      getHead(t.ch[1], dic, leafList, heads)

    elif leftC == "N-gN" and rightC == "N":
      getHead(t.ch[0], dic, leafList, heads=[])
      getHead(t.ch[1], dic, leafList, heads)

    elif leftC == "N-bQ" and rightC == "Q-gN":
      getHead(t.ch[0], dic, leafList, heads=[])
      getHead(t.ch[1], dic, leafList, heads)

    elif "D-g" in leftC and rightC == "N":
      getHead(t.ch[0], dic, leafList, heads=[])
      getHead(t.ch[1], dic, leafList, heads)

    elif leftCats[-1][1]== 'b' and leftCats[-1][2] == rightCats[0][2]:
      getHead(t.ch[1], dic, leafList, heads=[])
      getHead(t.ch[0], dic, leafList, heads)

    elif rightCats[-1][1]== 'a' and rightCats[-1][2] == leftCats[0][2]:
      getHead(t.ch[1], dic, leafList, heads)
      getHead(t.ch[0], dic, leafList, heads=[])
    #coordination
    elif leftCats[-1][1]== 'd' and leftCats[-1][2] == t.ch[1].c:
#      print("found coor")
      getHead(t.ch[1], dic, leafList, heads)
      getHead(t.ch[0], dic, leafList, heads=[])

    #argumentation    
    else:
      if len(leftList) > len(rightList):
        getHead(t.ch[0], dic, leafList, heads)
        getHead(t.ch[1], dic, leafList, heads=[])
      elif len(leftList) < len(rightList):
        getHead(t.ch[0], dic, leafList, heads=[])
        getHead(t.ch[1], dic, leafList, heads)
      else:
        #punctuation
        if leftC == "PU":
          getHead(t.ch[0],dic,leafList, heads=[])
          getHead(t.ch[1], dic, leafList, heads)
        elif rightC == "PU":
          getHead(t.ch[1], dic, leafList, heads=[])
          getHead(t.ch[0], dic, leafList, heads)
          
      

def complexCat(agList):
#  print agList
#  print len(agList)
  catList = []
  openB = 0
  i = 0
  if len(agList) == 1:
    catList.append(agList[0])
    return catList
  else:
    while i < len(agList):
      if "{" not in agList[i]:
        if "}" not in agList[i]:
          catList.append(agList[i])
          i += 1
        else:
          i += 1
 #     print catList
      else :
        if agList[i].count("{") == agList[i].count("}"):
          catList.append(agList[i])
          i += 1
        else:
          openB += 1
          complexC = agList[i]
          if i+1 < len(agList):
            for j in range(i+1, len(agList)+1):
              if openB != 0:
                if "{" in agList[j]:
                  openB += 1
                  complexC = complexC + "-" + agList[j]
#            print complexC
                elif "}" in agList[j]:
                  complexC = complexC + "-" + agList[j]
                  openB -= agList[j].count("}")
                else:
                  complexC = complexC + "-" + agList[j]
              else:
                catList.append(complexC)
                i = j
                break
    return catList

#strip curly brakets
def rmBkts(cat):
  if cat[0] == "{" and cat[-1] == "}":
    return cat[1:-1]
  else:
    return cat


#decompose gcg cat into a list of tuples like [(0,0,V), (1, a, N), (2, b, N)] 
def decompCat(cat):
  cats = []      
  agList = re.split("-", cat)
#  print agList
  openB = 0
  catList = complexCat(agList)
#  print (catList)
#  print (len(catList))
  for i in range(0, len(catList)):
#    print (catList[i])
    if len(catList[i]) == 1 or catList[i].isupper():
      cats.append((i, "head", rmBkts(catList[i])))      
    else:
      cats.append((i, catList[i][0], rmBkts(catList[i][1:])))
  return cats

#
def getDeps(t, headDic, deps, fillers, sharedP):
  modifier = ["A-bN", "R-bV"]
  punc = ["PU"]
  #unary branch does not add dep, unless gap introduced
  if len(t.ch) == 1:
    pareCats = decompCat(t.c)
    childCats = []
    if len(t.ch[0].ch) != 0:
      childCats = decompCat(t.ch[0].c)
    if t.c == "N-gN":
      if len(fillers)> 0:
        deps.append(("of-asso", headDic[(t.ch[0].l, t.ch[0].r)], fillers[-1]))
        fillers.pop()
      
    if 'g' in [x[1] for x in pareCats]:
      if 'g' not in [x[1] for x in childCats]:
        if len(pareCats) == len(childCats):
          # Rule Ga: one of arguments becomes gap
          rel = list(set(pareCats)-set(childCats))[0][0]
          if len(fillers) > 0:
            deps.append((rel, headDic[(t.ch[0].l, t.ch[0].r)], fillers[-1]))
            fillers.pop()
        elif len(pareCats) - len(childCats) == 1:
          # Rule Gc: topicalization
          if t.c == "N-gN" and t.ch[0].c == "N":
            if len(fillers)> 0:
              deps.append(("of-asso", headDic[(t.ch[0].l, t.ch[0].r)], fillers[-1]))
              fillers.pop()
          else:
            # Rule Gb: hypothesize a modifier
            if len(fillers) > 0:
              deps.append(("mod", headDic[(t.ch[0].l, t.ch[0].r)], fillers[-1]))
              fillers.pop()
                   
    getDeps(t.ch[0], headDic, deps, fillers, sharedP)
  
  #binary branch
  elif len(t.ch) == 2:
    curCats = decompCat(t.c)
    leftC = t.ch[0].c
    rightC = t.ch[1].c
    leftCats = decompCat(t.ch[0].c)
    rightCats = decompCat(t.ch[1].c)

    #Rule Ma
    if leftC in punc and rightC == t.c:
      deps.append(('punc', headDic[(t.ch[1].l, t.ch[1].r)], headDic[t.ch[0].l, t.ch[0].r]))

    #Rule Mb
    elif rightC in punc and leftC == t.c:
      deps.append(('punc', headDic[(t.ch[0].l, t.ch[0].r)], headDic[t.ch[1].l, t.ch[1].r]))

      
    #Rule Ma
    if leftC in modifier and rightC == t.c:
      deps.append(('mod', headDic[(t.ch[1].l, t.ch[1].r)], headDic[t.ch[0].l, t.ch[0].r]))

    #Rule Mb
    elif rightC in modifier and leftC == t.c:
      deps.append(('mod', headDic[(t.ch[0].l, t.ch[0].r)], headDic[t.ch[1].l, t.ch[1].r]))

    #Rule Ab
    elif leftCats[-1][1] == 'b' and leftCats[-1][2] == t.ch[1].c:
      deps.append((leftCats[-1][0], headDic[(t.ch[0].l, t.ch[0].r)], headDic[t.ch[1].l, t.ch[1].r]))

    #Rule Aa
    elif rightCats[-1][1] == 'a' and rightCats[-1][2] == t.ch[0].c:
      deps.append((rightCats[-1][0], headDic[(t.ch[1].l, t.ch[1].r)], headDic[t.ch[0].l, t.ch[0].r]))

    #Rule Ad
    elif leftCats[-1][1] == 'g' and leftCats[0][2] == rightCats[-1][2] and rightCats[-1][1] == "a":
      deps.append((rightCats[-1][0], headDic[(t.ch[1].l, t.ch[1].r)], headDic[t.ch[0].l, t.ch[0].r]))

    elif rightCats[-1][1] == 'g' and rightCats[0][2] == leftCats[-1][2] and leftCats[-1][1] == "b":
      deps.append((leftCats[-1][0], headDic[(t.ch[0].l, t.ch[0].r)], headDic[t.ch[1].l, t.ch[1].r]))

    elif leftCats[-1][1] == 'g' and leftCats[-2][1] == 'b' and leftCats[-2][2] == t.ch[1].c:
      deps.append((leftCats[-2][0], headDic[(t.ch[0].l, t.ch[0].r)], headDic[t.ch[1].l, t.ch[1].r]))

    #Rule Ac
    elif rightCats[-1][1] == 'g' and rightCats[-2][1] == 'a' and rightCats[-1][2] == t.ch[1].c:
      deps.append((rightCats[-2][0], headDic[(t.ch[1].l, t.ch[1].r)], headDic[t.ch[0].l, t.ch[0].r]))

    #gap filling
    # relative clause
    elif "D-g" in t.ch[0].c and t.ch[1].c == "N":
      fillers.append(headDic[(t.ch[1].l, t.ch[1].r)])

    #left child is filler
    elif rightCats[-1][1] == 'g' and rightCats[-1][2] == t.ch[0].c and curCats == rightCats[0:-1]:
      fillers.append(headDic[(t.ch[0].l, t.ch[0].r)])

    # right child is filler
    elif leftCats[-1][1] == 'g' and leftCats[-1][2] == t.ch[1].c and curCats == leftCats[0:-1]:
      fillers.append(headDic[(t.ch[1].l, t.ch[1].r)])

    #coordination
    #establish shared properties set for coordination sharedP[(2,3)...] means property 3 applies to all conjuncts of 2
    elif rightCats[-1][1] == 'c' and rightCats[-1][2] == t.ch[0].c:
        acst = t.getAncestors()
        ccom = []
        if acst != None:
            for a in acst:
                if a.sibling() != None:
                    ccom.append(a.sibling())
            if len(ccom) != 0:
                for c in ccom:
                    sharedP.append((headDic[(t.l,t.r)], headDic[(c.l, c.r)]))
                        
      #map shared deps to conjuncts
        for sp in sharedP:
            if headDic[(t.l, t.r)] in sp:
                for d in deps:
                    if sp[0] in d and sp[1] in d:
                        #print "parent deps", d
                        newList1 = list(d)
                        newList2 = list(d)
                        newList1[d.index(headDic[(t.l, t.r)])] = headDic[(t.ch[1].l, t.ch[1].r)]
                        newList2[d.index(headDic[(t.l, t.r)])] = headDic[(t.ch[0].l, t.ch[0].r)]
                        if tuple(newList1) not in deps:
                            deps.append(tuple(newList1))
                        if tuple(newList2) not in deps:
                            deps.append(tuple(newList2))
        deps.append(('cor', headDic[(t.ch[1].l, t.ch[1].r)], headDic[(t.ch[0].l, t.ch[0].r)]))
    getDeps(t.ch[0], headDic, deps, fillers, sharedP)
    getDeps(t.ch[1], headDic, deps, fillers, sharedP)
    
#  return deps

#add dependencies for object control, ba, bei, etc
def addDeps(t, headDic, deps):
  # Object control
  if t.c == "V-aN-b{V-aN}-bN":
    sbj = None
    vp = None
    for d in deps:
      if d[1] == headDic[(t.l, t.r)]:
        if d[0] == 3:
          sbj = d[2]
        if d[0] == 2:
          vp = d[2]
        if sbj != None and vp != None:
 #         print ("add object control", (1, vp, sbj))
          deps.append((1, vp, sbj))

  #passive voice
  if t.c == "V-aN-b{E-aN-gN}-bN":
    sbj = None
    obj = None
    vp = None
    for d in deps:
      if d[1] == headDic[(t.l, t.r)]:
        if d[0] == 3:
          sbj = d[2]
        if d[0] == 2:
          vp = d[2]
        if d[0] == 1:
          obj = d[2]
    if vp != None:
      for d in deps:
        if d[1] == headDic[(t.l, t.r)]:
          if d[0] == 3:
            sbj = d[2]
          if d[0] == 1:
            obj = d[2]
          if sbj != None:
            deps.append((1, vp, sbj))
 #           print ("add passive", (1, vp, sbj))
          if obj != None:
            deps.append((2, vp, obj))
 #           print ("add passive", (2, vp, obj))

  #Ba construction
  if t.c == "V-aN-b{B-aN}":
    sbj = None
    vp = None
    for d in deps:
      if d[1] == headDic[(t.l, t.r)]:
        if d[0] == 1:
          sbj = d[2]
        if d[0] == 2:
          vp = d[2]
        if sbj != None and vp != None:
          deps.append((1, vp, sbj))
 #         print ("add ba", (1, vp, sbj))
      
  for ch in t.ch:
    addDeps(ch, headDic, deps)
  return deps



#inputFile = open("test.txt", 'r')
ln = 0
#for line in inputFile:
for line in sys.stdin:
    deps = []
    newdeps = []
    deps1 = []
    moredeps = []
    ln += 1
    if opts.debug:
      sys.stderr.write ( str(ln) + '\n' )
    t = tree.Tree()
    try:
      t.read(line)
    except:
      sys.stderr.write ( 'Error reading tree in line ' + str(ln) + ': ' + line+ '\n' )
#    print(re.sub(" +", " ", t.leaf()))
      
    strList = t.leafList()
    spanDic = {}
    heads = []
    getHead(t, spanDic, strList, heads)

#    print ("length of headDic", len(spanDic))
#    print(spanDic)
    deps = []
    f = []
    s = []
    root = spanDic[(0, len(strList)-1)]
#    print(root)
    deps.append((0, (0, "ROOT"), root))
    getDeps(t, spanDic, deps, f,s)
#    print("length of deps", len(deps))
    moredeps = addDeps(t, spanDic, deps)
#    print ("length of more moredeps", len(moredeps))
    #remove repeated deps
    newdeps = list(set(moredeps))
#    print ("length of newdeps", len(newdeps))
#    print (sorted(newdeps, key=lambda x: x[2]))
    for d in sorted(newdeps, key=lambda x: x[2]):
      print (d)
    print()

#change the head of coordination!!!


#cat = "A-bN-b{V-aN}-g{R-bN-b{A-bN}}-b{V-aN}"
#cat = "V-c{V}"
#print cat
#print decompCat(cat)
