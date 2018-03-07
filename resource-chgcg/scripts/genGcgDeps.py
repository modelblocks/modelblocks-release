import pickle
import fileinput
import re
import tree
import operator
import codecs
import sys
#import argparse


#parser = argparse.ArgumentParser(description='Print eventuality (semantic) dependencies for Chinese GCG tree.')
#parser.add_argument('-d', '--debug', default=False, required=False, action='store_true', help='print debug info')
#parser.add_argument('-t', '--tree', default=False, required=False, action='store_true', help='enforce tree restriction (for scoring)')
#opts = parser.parse_args()



#traverse the tree top-down to read out head for each span of the tree
#output into a dic={(0,3):3, (1,5):3...}
#use span as key and head idx as value
def getHead(t, dic, leafList, heads):
#  print t.l
#  print t.l
  heads.append((t.l,t.r))
  modalV = ["是", "可能", "要", "会", "能否", "能", "能够", "必须", "可", "可以", "愿意", "愿", "应", "应该", "须", "必须"]
  #reach the terminals 
  if len(t.ch) == 0:
#    print("reach terminal", t.c)
    if heads != []:
#      print(heads)
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
    curCats = decompCat(t.c)
    leftCats = decompCat(t.ch[0].c)
    rightCats = decompCat(t.ch[1].c)
    #print ("left", leftCats)
    #print ("right", rightCats)
    # some parsing mistakes
    if leftC == "PU":
      getHead(t.ch[0],dic,leafList, heads=[])
      getHead(t.ch[1], dic, leafList, heads)
    elif rightC == "PU":
      getHead(t.ch[1], dic, leafList, heads=[])
      getHead(t.ch[0], dic, leafList, heads)

    elif leftC == "N-gN" and rightC in ["N-c{N}", "N-cN"]:
      getHead(t.ch[0], dic, leafList, heads=[])
      getHead(t.ch[1], dic, leafList, heads)

    elif leftC == "D-gN" and rightC =="V-aN":
      getHead(t.ch[0], dic, leafList, heads=[])
      getHead(t.ch[1], dic, leafList, heads)

    elif leftC == "N-bQ" and rightC =="Q":
 #     print("fires?", leftC, rightC)
      getHead(t.ch[0], dic, leafList, heads=[])
      getHead(t.ch[1], dic, leafList, heads)

    elif leftC == "N" and rightC =="N" and t.c == "N":
      print("fires?", leftC, rightC)
      getHead(t.ch[0], dic, leafList, heads)
      getHead(t.ch[1], dic, leafList, heads=[])


    #modification
    elif leftC == "A-bN" and rightC == t.c:
#      print("found Nmod", leftC, t.ch[1].ch[0].c)
      getHead(t.ch[0], dic, leafList, heads=[])
      getHead(t.ch[1], dic, leafList, heads)
    elif rightC == "A-bN" and leftC == t.c:
      getHead(t.ch[0], dic, leafList, heads)
      getHead(t.ch[1], dic, leafList, heads=[])
    elif leftC == "R-bV" and rightC == t.c:
      getHead(t.ch[0], dic, leafList, heads=[])
      getHead(t.ch[1], dic, leafList, heads)
    elif rightC == "R-bV" and leftC == t.c:
      getHead(t.ch[0], dic, leafList, heads)
      getHead(t.ch[1], dic, leafList, heads=[])
        
    #make Q-bN as modifier to match chinese semeval format
    elif leftC == "Q-bN" and rightC == "N":
      getHead(t.ch[0], dic, leafList, heads=[])
      getHead(t.ch[1], dic, leafList, heads)

    elif leftC == "N-gN" and rightC == "N":
      getHead(t.ch[0], dic, leafList, heads=[])
      getHead(t.ch[1], dic, leafList, heads)

    elif leftC == "N-gN" and rightC == "V-aN":
      getHead(t.ch[0], dic, leafList, heads=[])
      getHead(t.ch[1], dic, leafList, heads)

#    elif leftC == "N-gN" and rightC == "N-c{N}":
#      getHead(t.ch[0], dic, leafList, heads=[])
#      getHead(t.ch[1], dic, leafList, heads)

    elif leftC == "N-bQ" and rightC == "Q-gN":
      getHead(t.ch[0], dic, leafList, heads=[])
      getHead(t.ch[1], dic, leafList, heads)

 #   elif leftC == "N-bQ" and rightC == "Q":
 #     getHead(t.ch[0], dic, leafList, heads=[])
 #     getHead(t.ch[1], dic, leafList, heads)

    elif "D-g" in leftC and rightC == "N":
      getHead(t.ch[0], dic, leafList, heads=[])
      getHead(t.ch[1], dic, leafList, heads)

      #semeval 16 modal verbs are not heads
    elif leftC == 'V-aN-b{V-aN}' and rightC == "V-aN" and t.ch[0].ch[0].c in modalV:
#      print("found modal verbs")
      getHead(t.ch[0], dic, leafList, heads=[])
      getHead(t.ch[1], dic, leafList, heads)

    elif leftC == 'V-aN-b{V-aN}' and rightC == "V-aN-gN" and t.ch[0].ch[0].c in modalV:
#      print("found modal verbs")
#      print(leftC, rightC, t.ch[0].ch[0].c)
#      print(dic)
      getHead(t.ch[0], dic, leafList, heads=[])
      getHead(t.ch[1], dic, leafList, heads)

    elif rightC == 'V-aN-bN-a{V-aN-bN}' and leftC == "V-aN-bN":
      getHead(t.ch[0], dic, leafList, heads)
      getHead(t.ch[1], dic, leafList, heads=[])

    elif rightC == 'V-aN-bN-a{V-aN}' and leftC == "V-aN":
#      print("found modal verbs")
      getHead(t.ch[0], dic, leafList, heads)
      getHead(t.ch[1], dic, leafList, heads=[])

    #N N-aN -> N is head
    elif leftC == 'N' and rightC == "N-aN":
#      print("found N-aN")
      getHead(t.ch[0], dic, leafList, heads)
      getHead(t.ch[1], dic, leafList, heads=[])

    #semeval 16 prepositions are not heads
    elif leftCats[0][2] == "R" and leftCats[1][2] == "V" and  leftCats [-1][1] == 'b' and  leftCats[-1][2] == rightCats[0][2]:
      getHead(t.ch[0], dic, leafList, heads=[])
      getHead(t.ch[1], dic, leafList, heads)

    elif leftCats[0][2] == "A" and leftCats[1][2] == "N" and  leftCats [-1][1] == 'b' and  leftCats[-1][2] == rightCats[0][2]:
      getHead(t.ch[0], dic, leafList, heads=[])
      getHead(t.ch[1], dic, leafList, heads)

    elif leftCats[0][2] == "R" and leftCats[1][2] == "V" and  leftCats [-1][1] == 'b' and  leftCats[-1][2] == rightC:
      getHead(t.ch[0], dic, leafList, heads=[])
      getHead(t.ch[1], dic, leafList, heads)

    elif leftCats[0][2] == "A" and leftCats[1][2] == "N" and  leftCats [-1][1] == 'b' and  leftCats[-1][2] == rightC:
      getHead(t.ch[0], dic, leafList, heads=[])
      getHead(t.ch[1], dic, leafList, heads)

    elif rightCats[0][2] == "R" and rightCats[1][2] == "V" and  rightCats [-1][1] == 'a' and  rightCats[-1][2] == leftCats[0][2]:
      getHead(t.ch[0], dic, leafList, heads)
      getHead(t.ch[1], dic, leafList, heads=[])

    elif rightCats[0][2] == "A" and rightCats[1][2] == "N" and  rightCats [-1][1] == 'a' and  rightCats[-1][2] == leftCats[0][2]:
      getHead(t.ch[0], dic, leafList, heads)
      getHead(t.ch[1], dic, leafList, heads=[])

    elif rightCats[0][2] == "R" and rightCats[1][2] == "V" and  rightCats [-1][1] == 'a' and  rightCats[-1][2] == leftC:
      getHead(t.ch[0], dic, leafList, heads)
      getHead(t.ch[1], dic, leafList, heads=[])

    elif rightCats[0][2] == "A" and rightCats[1][2] == "N" and  rightCats [-1][1] == 'a' and  leftCats[-1][2] == leftC:
      getHead(t.ch[0], dic, leafList, heads)
      getHead(t.ch[1], dic, leafList, heads=[])

    #DER
    elif leftC == "V-aN-bN" and rightC=="V-aN-b{V-aN}-a{V-aN-bN}":
      getHead(t.ch[0], dic, leafList, heads)
      getHead(t.ch[1], dic, leafList, heads=[])

    elif leftC == "V-aN-bN" and rightC=="V-aN-b{V-aN}-bN-a{V-aN-bN}":
      getHead(t.ch[0], dic, leafList, heads)
      getHead(t.ch[1], dic, leafList, heads=[])


    elif leftCats[-1][1]== 'b' and leftCats[-1][2] == rightCats[0][2]:
      getHead(t.ch[1], dic, leafList, heads=[])
      getHead(t.ch[0], dic, leafList, heads)

    elif rightCats[-1][1]== 'a' and rightCats[-1][2] == leftCats[0][2]:
#      print("fires?", leftCats, rightCats)
      getHead(t.ch[1], dic, leafList, heads)
      getHead(t.ch[0], dic, leafList, heads=[])
    #coordination
    elif leftCats[-1][1]== 'd' and leftCats[-1][2] == t.ch[1].c:
#      print("found coor")
      getHead(t.ch[1], dic, leafList, heads)
      getHead(t.ch[0], dic, leafList, heads=[])

    #argumentation    
    else:
      #if len(leftList) > len(rightList):
      if len(leftCats) > len(rightCats):
        getHead(t.ch[0], dic, leafList, heads)
        getHead(t.ch[1], dic, leafList, heads=[])
#      elif len(leftList) < len(rightList):
      elif len(leftCats) < len(rightCats):
        #print("activate")
        getHead(t.ch[0], dic, leafList, heads=[])
        getHead(t.ch[1], dic, leafList, heads)
      else:
        print("same length")
        if curCats[0][2] == leftCats[0][2]:
          print("found same length cat", curCats, leftCats, rightCats)
          getHead(t.ch[0], dic, leafList, heads)
          getHead(t.ch[1], dic, leafList, heads=[])
        elif curCats[0][2] == rightCats[0][2]:
          getHead(t.ch[0], dic, leafList, heads=[])
          getHead(t.ch[1], dic, leafList, heads)
        else:
          print("same left and right ")
          getHead(t.ch[0], dic, leafList, heads=[])
          getHead(t.ch[1], dic, leafList, heads)
          
          
        #punctuation
        if leftC == "PU":
          getHead(t.ch[0],dic,leafList, heads=[])
          getHead(t.ch[1], dic, leafList, heads)
        elif rightC == "PU":
          getHead(t.ch[1], dic, leafList, heads=[])
          getHead(t.ch[0], dic, leafList, heads)
  else:
    print(t.c, t.ch[0].c, t.ch[1].c, t.ch[2])
          
      

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


#decompose gcg cat into a list of tuples like [(0,0,V), (1, a, N), (2, b, N)] for V-aN-bN 
def decompCat(cat):
#  print(cat)
  cats = []      
  agList = re.split("-", cat)
#  print(agList)
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
def getDeps(t, headDic, deps, fillers, sharedP, catcher):
  modifier = ["A-bN", "R-bV"]
  punc = ["PU"]
  rightPunc = ["“", "（"]
  #unary branch does not add dep, unless gap introduced
  if len(t.ch) == 1:
    #print("parent, child categories", t.c, t.ch[0].c)
    pareCats = decompCat(t.c)
    childCats = []
    if len(t.ch[0].ch) != 0:
      childCats = decompCat(t.ch[0].c)
    if t.c == "N-gN":
      if len(fillers)> 0:
        deps.append(("of-asso", headDic[(t.ch[0].l, t.ch[0].r)], fillers[-1]))
        fillers.pop()
    #print([x[1] for x in pareCats], [x[1] for x in childCats])
    #print("oareCats, childCats", pareCats, childCats)
    if 'g' in [x[1] for x in pareCats]:
      if 'g' not in [x[1] for x in childCats]:
        if len(pareCats) == len(childCats):
          # Rule Ga: one of arguments becomes gap
          rel = list(set(pareCats)-set(childCats))[0][0]
#          print("found gappy", pareCats, childCats, set(pareCats)-set(childCats), rel)
#          print("fillers", fillers)
          
          if len(fillers) > 0:
#            deps.append(("gap"+str(rel), headDic[(t.ch[0].l, t.ch[0].r)], fillers[-1]))
            deps.append((rel, headDic[(t.ch[0].l, t.ch[0].r)], fillers[-1]))
            #print("add ", (rel, headDic[(t.ch[0].l, t.ch[0].r)], fillers[-1]))
            catcher.append(fillers[-1])
            fillers.pop()
          elif len(catcher) != 0:
            deps.append((rel, headDic[(t.ch[0].l, t.ch[0].r)], catcher[-1]))
            #print("add from catcher", (rel, headDic[(t.ch[0].l, t.ch[0].r)], catcher[-1]))
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
                   
    getDeps(t.ch[0], headDic, deps, fillers, sharedP, catcher)
  
  #binary branch
  elif len(t.ch) == 2:
    curCats = decompCat(t.c)
    leftC = t.ch[0].c
    rightC = t.ch[1].c
    leftCats = decompCat(t.ch[0].c)
    rightCats = decompCat(t.ch[1].c)
#    if t.c == "V-aN" and t.ch[0].c == "N":
#      print("current", curCats)
#      print("left", leftCats)
#      print("right", rightCats)
#      print(rightCats[-1][1], rightCats[-1][2],  t.ch[0].c,  curCats, rightCats[0:-1])
    #topicalization
    if t.c == "N-gN" and t.ch[1].c == "N":
      if len(fillers)> 0:
        deps.append(("of-asso", headDic[(t.ch[1].l, t.ch[1].r)], fillers[-1]))
        fillers.pop()

    #punctuation
    if leftC == "PU":
      if t.ch[0].ch[0].c in rightPunc:
        deps.append(('punc', headDic[(t.ch[1].l, t.ch[1].r)], headDic[t.ch[0].l, t.ch[0].r]))
      else:
        if t.p != None:
          deps.append(('punc',headDic[(t.p.ch[0].l, t.p.ch[0].r)], headDic[t.ch[0].l, t.ch[0].r]))
#          print("redirected punctuation")

    #Rule Mb
    elif rightC in punc and leftC == t.c:
      deps.append(('punc', headDic[(t.ch[0].l, t.ch[0].r)], headDic[t.ch[1].l, t.ch[1].r]))

      
    #Rule Ma
    elif leftC == "A-bN" and rightC == t.c:
      deps.append(('Nmod', headDic[(t.ch[1].l, t.ch[1].r)], headDic[t.ch[0].l, t.ch[0].r]))

    elif leftC == "R-bV" and rightC == t.c:
      deps.append(('Vmod', headDic[(t.ch[1].l, t.ch[1].r)], headDic[t.ch[0].l, t.ch[0].r]))

    #Rule Mb
    elif rightC == "A-bN" and leftC == t.c:
      deps.append(('Nmod', headDic[(t.ch[0].l, t.ch[0].r)], headDic[t.ch[1].l, t.ch[1].r]))

    elif rightC == "R-bV" and leftC == t.c:
      deps.append(('Vmod', headDic[(t.ch[0].l, t.ch[0].r)], headDic[t.ch[1].l, t.ch[1].r]))


    #semeval 16 prepositions are not heads
    elif leftCats[0][2] == "R" and leftCats[1][2] == "V" and  leftCats [-1][1] == 'b' and  leftCats[-1][2] == rightCats[0][2]:
      deps.append((leftCats[-1][0], headDic[(t.ch[1].l, t.ch[1].r)], headDic[(t.ch[0].l, t.ch[0].r)]))
      
    elif leftCats[0][2] == "A" and leftCats[1][2] == "N" and  leftCats [-1][1] == 'b' and  leftCats[-1][2] == rightCats[0][2]:
      deps.append((leftCats[-1][0], headDic[(t.ch[1].l, t.ch[1].r)], headDic[(t.ch[0].l, t.ch[0].r)]))

    elif leftCats[0][2] == "R" and leftCats[1][2] == "V" and  leftCats [-1][1] == 'b' and  leftCats[-1][2] == rightC:
      deps.append((leftCats[-1][0], headDic[(t.ch[1].l, t.ch[1].r)], headDic[(t.ch[0].l, t.ch[0].r)]))

    elif leftCats[0][2] == "A" and leftCats[1][2] == "N" and  leftCats [-1][1] == 'b' and  leftCats[-1][2] == rightC:
      deps.append((leftCats[-1][0], headDic[(t.ch[1].l, t.ch[1].r)], headDic[(t.ch[0].l, t.ch[0].r)]))

    elif rightCats[0][2] == "R" and rightCats[1][2] == "V" and  rightCats [-1][1] == 'a' and  rightCats[-1][2] == leftCats[0][2]:
      deps.append((rightCats[-1][0], headDic[(t.ch[0].l, t.ch[0].r)], headDic[(t.ch[1].l, t.ch[1].r)]))

    elif rightCats[0][2] == "A" and rightCats[1][2] == "N" and  rightCats [-1][1] == 'a' and  rightCats[-1][2] == leftCats[0][2]:
      deps.append((rightCats[-1][0], headDic[(t.ch[0].l, t.ch[0].r)], headDic[(t.ch[1].l, t.ch[1].r)]))

    elif rightCats[0][2] == "R" and rightCats[1][2] == "V" and  rightCats [-1][1] == 'a' and  rightCats[-1][2] == leftC:
      deps.append((rightCats[-1][0], headDic[(t.ch[0].l, t.ch[0].r)], headDic[(t.ch[1].l, t.ch[1].r)]))

    elif rightCats[0][2] == "A" and rightCats[1][2] == "N" and  rightCats [-1][1] == 'a' and  leftCats[-1][2] == leftC:
      deps.append((rightCats[-1][0], headDic[(t.ch[0].l, t.ch[0].r)], headDic[(t.ch[1].l, t.ch[1].r)]))

    #DER
 #   elif leftC == "V-aN-bN" and rightC=="V-aN-b{V-aN}-a{V-aN-bN}":
 #     getHead(t.ch[0], dic, leafList, heads)
 #     getHead(t.ch[1], dic, leafList, heads=[])

    #Rule Ab
    elif leftCats[-1][1] == 'b' and leftCats[-1][2] == rightC:
      deps.append((leftCats[-1][0], headDic[(t.ch[0].l, t.ch[0].r)], headDic[t.ch[1].l, t.ch[1].r]))

    #Rule Aa
    elif rightCats[-1][1] == 'a' and rightCats[-1][2] == leftC:
      deps.append((rightCats[-1][0], headDic[(t.ch[1].l, t.ch[1].r)], headDic[t.ch[0].l, t.ch[0].r]))

    #Rule Ad
    elif t.c == "V-aN-gN"  and  t.ch[0].c == "V-aN-b{V-aN}" and t.ch[1].c == "V-aN-gN":
      #print("found modal verb with gappy complement")
      deps.append((2, headDic[(t.ch[0].l, t.ch[0].r)], headDic[t.ch[1].l, t.ch[1].r]))

    elif leftCats[-1][1] == 'g' and leftCats[0][2] == rightCats[-1][2] and rightCats[-1][1] == "a":
      deps.append((rightCats[-1][0], headDic[(t.ch[1].l, t.ch[1].r)], headDic[t.ch[0].l, t.ch[0].r]))

    elif rightCats[-1][1] == 'g' and rightCats[0][2] == leftCats[-1][2] and leftCats[-1][1] == "b":
      deps.append((leftCats[-1][0], headDic[(t.ch[0].l, t.ch[0].r)], headDic[t.ch[1].l, t.ch[1].r]))

    elif leftCats[-1][1] == 'g' and leftCats[-2][1] == 'b' and leftCats[-2][2] == t.ch[1].c:
      deps.append((leftCats[-2][0], headDic[(t.ch[0].l, t.ch[0].r)], headDic[t.ch[1].l, t.ch[1].r]))

    #Rule Ac
    elif rightCats[-1][1] == 'g' and rightCats[-2][1] == 'a' and rightCats[-2][2] == t.ch[0].c and curCats == rightCats[0:-2] + [rightCats[-1]]:
      deps.append((rightCats[-2][0], headDic[(t.ch[1].l, t.ch[1].r)], headDic[t.ch[0].l, t.ch[0].r]))


    #gap filling
    # relative clause
    elif "D-g" in t.ch[0].c and t.ch[1].c == "N":
      fillers.append(headDic[(t.ch[1].l, t.ch[1].r)])

    #left child is filler
    elif rightCats[-1][1] == 'g' and rightCats[-1][2] == t.ch[0].c and curCats == rightCats[0:-1]:
#      print("see whether this fires", leftCats, rightCats, curCats)
      fillers.append(headDic[(t.ch[0].l, t.ch[0].r)])
     # print("filler", filler)

#    elif rightCats[-1][1] == 'g' and rightCats[-1][2] == t.ch[0].c and curCats == rightCats[0:-1]:
#      fillers.append(headDic[(t.ch[0].l, t.ch[0].r)])

    # right child is filler
    elif leftCats[-1][1] == 'g' and leftCats[-1][2] == t.ch[1].c and curCats == leftCats[0:-1]:
      fillers.append(headDic[(t.ch[1].l, t.ch[1].r)])
      
    # coordination of relative clause, add one more filler
    elif t.ch[0].c == "D-gN" and t.ch[1].c == "D-gN-c{D-gN}":
      if len(fillers) != 0:
        filler = fillers[-1]
        fillers.append(filler)

#    elif t.ch[0].c == "V-gN" and t.ch[1].c == "V-gN-c{V-gN}":
#      filler = fillers[-1]
#      fillers.append(filler)

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
                          if tuple(newList1)[0] != "punc":
                            deps.append(tuple(newList1))
                        if tuple(newList2) not in deps:
                          if tuple(newList2)[0] != "punc":
                            deps.append(tuple(newList2))
        deps.append(('cor', headDic[(t.ch[1].l, t.ch[1].r)], headDic[(t.ch[0].l, t.ch[0].r)]))
    getDeps(t.ch[0], headDic, deps, fillers, sharedP, catcher)
    getDeps(t.ch[1], headDic, deps, fillers, sharedP, catcher)
    
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
    mod = []
    cor = []
    pc = []
    for d in deps:
      if d[1] == headDic[(t.l, t.r)]:
        if d[0] == 3:
          sbj = d[2]
        if d[0] == 2:
          vp = d[2]
        if d[0] == 1:
          obj = d[2]
        if d[0] == "Vmod":
          mod.append(d[2])
#          deps.remove(d)
        if d[0] == "cor":
          cor.append(d[2])
#          deps.remove(d)
        if d[0] == "punc":
          pc.append(d[2])
    for d in deps:
      if d[2] == headDic[(t.l, t.r)]:
        if d[0] == "cor":
          cor.append(d[1])
#          deps.remove(d)
    if vp != None:
      if sbj != None:
#        deps.append(("pass1", vp, sbj))
        deps.append((1, vp, sbj))
#        print ("add passive", (1, vp, sbj))
#      if obj != None:
#        deps.append(("pass_obj", vp, obj))
#        deps.append((2, vp, obj))
      if mod != []:
#        print("modList", mod)
        for m in mod:
          deps.append(("Vmod", vp, m))
#          print("found passive modifiers")
      if cor != []:
        for c in cor:
          deps.append(("cor", vp, c))
#          print("found passive conjunction")
      if pc!=[]:
        for p in pc:
          deps.append(("punc", vp, p))
#          print("found punc passive")
 #           print ("add passive", (2, vp, obj))

  io = ["E-aN-bN-gN", "E-aN-b{V-aN}-gN"]
  if t.c == "V-aN" and t.ch[0].c == "V-aN-b{E-aN-gN}":
    bei = headDic[t.ch[0].ch[0].l, t.ch[0].ch[0].r]
 #   print("bei", bei)
    sbj = None
    vp = None
    obj = None
    for d in deps:
      if d[1] == bei:
        if d[0] == 3:
          sbj = d[2]
        if d[0] == 2:
          vp = d[2]
        if d[0] == 1:
          obj = d[2]
    if vp != None:
      if obj != None:
        if t.ch[1].ch[0].c in io or t.ch[1].ch[0].c in io:
#          print("add io")
          deps.append((3, vp, obj))
        elif t.ch[1].ch[0].c == "R-bV":
          if t.ch[1].ch[1].ch[0].c in io or t.ch[1].ch[1].ch[0].ch[0].c in io:
            deps.append((3, vp, obj))
        else:
 #         print("add do")
          deps.append((2, vp, obj))
   
  #Ba construction
  if t.c == "V-aN-b{B-aN}":
    sbj = None
    vp = None
    mod = []
    cor = []
    pc = []
    for d in deps:
      if d[1] == headDic[(t.l, t.r)]:
        if d[0] == 1:
          sbj = d[2]
        if d[0] == 2:
          vp = d[2]
        if d[0] == "Vmod":
          mod.append(d[2])
#          deps.remove(d)
        if d[0] == "cor":
          cor.append(d[2])
        if d[0] == "punc":
          pc.append(d[2])
#          deps.remove(d)
    for d in deps:
      if d[2] == headDic[(t.l, t.r)]:
        if d[0] == "cor":
          cor.append(d[2])
          deps.remove(d)
    if sbj != None and vp != None:
      deps.append((1, vp, sbj))
    if mod != []:
#      print("modList",mod)
      for m in mod:
        deps.append(("Vmod", vp, m))
    if cor != []:
      for c in cor:
        deps.append(("cor", vp, c))
    if pc != []:
      for p in pc:
        deps.append(("punc", vp, p))
          
#        print("found passive modifiers")
            
 #         print ("add ba", (1, vp, sbj))
  #DER
  if t.c == "V-aN-b{V-aN}-bN-a{V-aN-bN}":
    sbj = None
    sbj2 = None
    vp = None
    vp2 = None
    for d in deps:
      if d[1] == headDic[(t.l, t.r)]:
        if d[0] == 1:
          sbj = d[2]
        if d[0] == 2:
          vp2 = d[2]
        if d[0] == 3:
          sbj2 = d[2]
        if d[0] == 4:
          vp = d[2]
    if vp != None:
      deps.append(("Dcom",vp, headDic[(t.l, t.r)]))
    if sbj2 == None:
      if sbj != None:
        if vp != None:
          deps.append((1, vp, sbj))
        if vp2 != None:
          deps.append((1, vp2, sbj))
        if vp != None and vp2 != None:
          deps.append(("Dcor", vp, vp2))
    else:
      if sbj != None:
        if vp != None:
          deps.append((1, vp, sbj))
        if vp2 != None:
          deps.append((1, vp2, sbj2))
  #DEV
  if t.c == "R-bV-a{V-aN}":
    #print("Found DEV",headDic[(t.l, t.r)])
    vp = None
    vmod = None
    for d in deps:
      if d[1] == headDic[(t.l, t.r)]:
        if d[0] == 2:
          vmod = d[2]
          #print("Vmod", vmod)
      if d[2] == headDic[(t.l, t.r)]:
        #print("D", d)
        if d[0] == "Vmod":
          vp = d[1]
          #print("VP", vp)
    if vp != None and vmod != None:
      deps.append(("Man", vp, vmod))

  #DEV
  if t.c == "A-bN-aN":
    np = None
    nmod = None
    for d in deps:
      if d[1] == headDic[(t.l, t.r)]:
        if d[0] == 2:
          nmod = d[2]
      if d[2] == headDic[(t.l, t.r)]:
        if d[0] == "Nmod":
          np = d[1]
    if np != None and nmod != None:
      deps.append(("attr", np, nmod))

  for ch in t.ch:
    addDeps(ch, headDic, deps)
  return deps

def expandNMod(ds):
  modD = {}
  for d in ds:
    if d[0] == "Nmod":
      mHead = d[1]
      modD[mHead] = []
      for d in ds:
        if d[0] == "Nmod" and d[1]== mHead:
          if d[2] not in modD[mHead]:
            modD[mHead].append(d[2])
  for m in modD:
    if len(modD[m]) > 1:
      for i in range(0, len(modD[m])):
        for j in range(0, len(modD[m])):
          if i < j:
            deps.append(("coNmod",modD[m][i],modD[m][j]))
  return deps


#def addGapCor(ds):
#  newdeps = []
#  for d in ds:
#    newdeps.append(d)
#  for d in ds:
#    if d[0] == "cor":
#      c1 = d[1]
#      c2 = d[2]
#      for i in ds:
#        if i[0] in ["gap1", "gap2", "gap3", "of-asso"]:
#          if c1 == i[1]:
#            newdeps.append((i[0],c2,i[2]))
#          elif c2 == i[1]:
#            newdeps.append((i[0], c1,i[2]))
# #         elif c1==i[2]:
# #           newdeps.append((i[0],i[1],c2))
# #         elif c2==i[2]:
# #           newdeps.append((i[0],i[1],c1))
#  return newdeps

def addGapCor(ds):
  newdeps = []
  for d in ds:
    newdeps.append(d)
  for d in ds:
    head = d[1]
    child = d[2]
    for d1 in ds:
      if d1[0] == "cor":
        c1 = d1[1]
        c2 = d1[2]
        if child == c1:
          newdeps.append((d[0], head, c2))
        elif child == c2:
          newdeps.append((d[0], head, c1))
        if head == c1:
          newdeps.append((d[0], c2, child))
        elif child == c2:
          newdeps.append((d[0], c1, child))
  return newdeps

#treeFile = open(sys.argv[1], 'r')

for line in sys.stdin:
  deps = []
  newdeps = []
  moredeps = []
  t = tree.Tree()
  try:
    t.read(line)
  except:
    sys.stderr.write ( 'Error reading tree in line ' + str(i) + ': ' + gtree+ '\n' )
#    print(re.sub(" +", " ", t.leaf()))
#   
  strList = t.leafList()
#  print(strList)
  spanDic = {}
  heads = []
  getHead(t, spanDic, strList, heads)
#  for h in sorted(spanDic):
#    print(h, spanDic[h])
  f = []
  s = []
  c = []
  root = spanDic[(0, len(strList)-1)]
#    print(root)
  deps.append((0, (0, "ROOT"), root))
  getDeps(t, spanDic, deps, f, s, c)
#    print("length of deps", len(deps))
  moredeps = addDeps(t, spanDic, deps)
#  moredeps1 = addGapCor(moredeps)
#    moredeps = expandNMod(deps)
#    print ("length of more moredeps", len(moredeps))
    #remove repeated deps
#  newdeps = list(set(moredeps1))
  newdeps = list(set(moredeps))

  for d in sorted(newdeps, key =lambda x:x[2][0]):
    #output deps in the format of "label(head-idx, child-idx)"
#    print(str(d[0])+"("+d[1][1]+"-"+str(d[1][0]+1)+", "+d[2][1]+"-"+str(d[2][0]+1)+")")
    print(str(d[0])+"("+d[1][1]+"-"+str(d[1][0])+", "+d[2][1]+"-"+str(d[2][0])+")")
  print()
#  break

    
