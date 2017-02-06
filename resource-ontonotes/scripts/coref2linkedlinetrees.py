import sys
import re
import getopt
sys.path.append('../resource-gcg/scripts')
import tree


def smallestContainingCat ( tr, targbeg, targend, catbeg=0 ):
  catmid,lfind = smallestContainingCat ( tr.ch[0], targbeg, targend, catbeg ) if len(tr.ch)>0 and len(tr.ch[0].ch)!=0 else (catbeg+1,[])
  catend,rfind = smallestContainingCat ( tr.ch[1], targbeg, targend, catmid ) if len(tr.ch)>1 else (catmid,[])
  return catend,lfind+rfind+([tr] if catbeg<=targbeg and (len(tr.ch)<2 or targbeg<catmid and catmid<targend) and targend<=catend else [])

def getHead ( tr ):
  if len(tr.ch)==1 and len(tr.ch[0].ch)==0: return tr
  #if tr.l==tr.r: return tr
  if len(tr.ch)==2 and tr.ch[0].c=='N-b{N-aD}' and '-lA' in tr.ch[1].c : return getHead ( tr.ch[1] )  # special case for complement of determiner as actual head
  if len(tr.ch)==2 and '-lA' in tr.ch[0].c     and tr.ch[1].c=='D-aN'  : return getHead ( tr.ch[0] )  # special case for complement of 's as actual head
  for subtr in tr.ch:
    if '-l' not in subtr.c:
      return getHead ( subtr )
  return tr

def addCoref ( tr, beg, end ):
  ls = smallestContainingCat ( tr, beg-1, end )[1]
  return getHead ( ls[0] ) if len(ls) > 0 else tree.Tree()


## command-line options...
optlist,args = getopt.gnu_getopt(sys.argv,'d')

treefile = open ( args[1] ) #sys.argv[1] )

PrevMention = { }

linenum = 0
for line in sys.stdin:

  if line.startswith('<TEXT') or line.startswith('</TEXT') or line.startswith('</DOC'):
    continue

  linenum += 1
  treestring = treefile.readline()

  if line.startswith('<DOC'):
    if '!ARTICLE' not in treestring: sys.stderr.write ( 'ERROR: instead of "...!ARTICLE..." I got: ' + treestring )
    print '(U !ARTICLE)'
    linenum = 0
    PrevMention = { }
    continue

  if treestring == '\n':
    print ''
    continue

  t = tree.Tree()
  t.read(treestring)

  RefStack = []
  toknum = 0
  for token in line.split():
    toknum += 1
    if ('-d','') in optlist:
      print '|' + token + '|' + str(toknum) + '|'
    for mentionbegin in re.findall('<COREF>',token):
      RefStack.append(toknum)
      if ('-d','') in optlist:
        print '  push '+str(toknum)
    for mentionid in re.findall('(?<=</COREF)[0-9]+(?=>)',token):
      mentionbegin = RefStack.pop()
      head = addCoref ( t, mentionbegin, toknum )
      if ('-d','') in optlist:
        print '  i think ' + str(mentionbegin) + ' ' + str(toknum) + ' is ' + str(head)
      if mentionid in PrevMention: head.c += '-n' + PrevMention[mentionid]
      PrevMention[mentionid] = str(linenum) + ('0' if head.r<9 else '') + str(head.r+1)
      if ('-d','') in optlist:
        print '  pop ' + mentionid + ' -> ' + PrevMention[mentionid]
  print str(t)


