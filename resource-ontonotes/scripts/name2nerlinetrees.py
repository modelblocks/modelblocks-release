import sys, os, re, getopt
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
import tree
import pdb


def smallestContainingCat ( tr, targbeg, targend, catbeg=0 ):
  catmid,lfind = smallestContainingCat ( tr.ch[0], targbeg, targend, catbeg ) if len(tr.ch)>0 and len(tr.ch[0].ch)!=0 else (catbeg+1,[])
  catend,rfind = smallestContainingCat ( tr.ch[1], targbeg, targend, catmid ) if len(tr.ch)>1 else (catmid,[])
  return catend,lfind+rfind+([tr] if catbeg<=targbeg and (len(tr.ch)<2 or targbeg<catmid and catmid<targend) and targend<=catend else [])

def getHead ( tr ):
  if len(tr.ch)==1 and len(tr.ch[0].ch)==0: return tr
  #if tr.l==tr.r: return tr
  if len(tr.ch)==2 and tr.ch[0].c.startswith('N-b{N-aD}') and (('-lA' in tr.ch[1].c) or ('-lU' in tr.ch[1].c)) : return getHead ( tr.ch[1] )  # special case for complement of determiner as actual head
  if len(tr.ch)==2 and ('-lA' in tr.ch[0].c or '-lU' in tr.ch[0].c) and tr.ch[1].c.startswith('D-aN')  : return getHead ( tr.ch[0] )  # special case for complement of 's as actual head
  for subtr in tr.ch:
    if '-l' not in subtr.c:
      return getHead ( subtr )
  return tr

def addName ( tr, beg, end ):
  ls = smallestContainingCat ( tr, beg-1, end )[1]
  return getHead ( ls[0] ) if len(ls) > 0 else tree.Tree()


## command-line options...
optlist,args = getopt.gnu_getopt(sys.argv,'d')
#optlist = [("-d","")]

treefile = open ( args[1] ) #sys.argv[1] )

#linenum = 0
#pdb.set_trace() #
namefile = open (args[2] )

#for line in sys.stdin: #name file streamed from stdin
for line in namefile: #name file passed as argument

  if line.startswith('</DOC'):
    continue

  #linenum += 1
  treestring = treefile.readline()

  if line.startswith('<DOC'):
    if '!ARTICLE' not in treestring: sys.stderr.write ( 'ERROR: instead of "...!ARTICLE..." I got: ' + treestring + '\n')
    print '(U !ARTICLE)'
    #linenum = 0
    continue

  if treestring == '\n':
    print ''
    continue

  t = tree.Tree()
  t.read(treestring)

  RefStack = []
  NameStack = []
  toknum = 0
  for token in line.split():
    toknum += 1
    # print (toknum)
    if ('-d','') in optlist:
      print '|' + token + '|' + str(toknum) + '|'
    # currently only uses PERSON and ORGANIZATION
    for necat in re.findall('<ENAMEXTYPE="(PERSON)">', token):
    #for necat in re.findall('<ENAMEXTYPE="([A-Z_]+)">', token):
      RefStack.append(toknum)
      # print (RefStack)
      NameStack.append(necat)
      # print (NameStack)
      if ('-d','') in optlist:
        print '  push '+str(toknum)
    for neend in re.findall('</ENAMEX>',token):
      # nebeg = RefStack.pop()
      # head = addName(t, nebeg, toknum)
      # delete this after finished
      if not RefStack == []:
        nebeg = RefStack.pop()
        head = addName ( t, nebeg, toknum )
      if ('-d','') in optlist:
        print '  i think ' + str(nebig) + ' ' + str(toknum) + ' is ' + str(head)
      # head.c += '-e' + NameStack.pop()
      # lowercase the named entity
      # try 'word' to 'categories'
      # delete this after finished
      if not NameStack == []:
        head.c += '-x%' + head.ch[0].c.lower() + '|%' + NameStack.pop().lower()
      # if mentionid in PrevMention: head.c += '-n' + PrevMention[mentionid]
      # PrevMention[mentionid] = str(linenum) + ('0' if head.r<9 else '') + str(head.r+1)
      # if ('-d','') in optlist:
      #   print '  pop ' + mentionid + ' -> ' + PrevMention[mentionid]
  print str(t)