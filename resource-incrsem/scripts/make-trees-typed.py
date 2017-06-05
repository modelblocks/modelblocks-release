import sys, os, collections, re
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
import tree

## recursive function to add referent type to 
def type ( tree, Types, artnum, linenum, toknum=1 ):
  if len(tree.ch)>=1 and len(tree.ch[0].ch)==0:
    ####if tree.c.startswith('N-aD'):
    if (artnum,linenum,toknum) in Types:
      ## obtain stem
      s = '(' + tree.c + ' ' + tree.ch[0].c + ')'
      while re.search('-x[A-Z]',s) is not None:
        s = re.sub( '\((.)([^ ]*?)-x(.)([^% ]*)%([^:|%() ]*):([^:|%() ]*)%([^:|%() ]*)\|(.)([^:|%() ]*)%([^:|%() ]*):([^:|%() ]*)%([^- ]*)([^ ]*) \\6([^() ]*)\\7\)', '(\\8\\2-o\\3\\4%\\5:\\6%\\7|\\8\\9%\\10:\\11%\\12\\13 \\11\\14\\12)', s )
        s = re.sub( '\((.)([^ ]*?)-x(.)([^% ]*)%([^| ]*)\|(.)([^% ]*)%([^- ]*)([^ ]*) ([^() ]*)\\5\)', '(\\6\\2-o\\3\\4%\\5|\\6\\7%\\8\\9 \\10\\8)', s )
        ##print( 'doing', s )
      s = re.sub( '\(.* (.*)\)', '\\1', s )
      ##while( s/\((.)([^ ]*?)-x(.)([^% ]*)%([^:|%() ]*):([^:|%() ]*)%([^:|%() ]*)\|(.)([^:|%() ]*)%([^:|%() ]*):([^:|%() ]*)%([^- ]*)([^ ]*) \6([^() ]*)\7\)/\($8$2-o$3$4%$5:$6%$7\|$8$9%$10:$11%$12$13 $11$14$12\)/g ||
      ##       s/\((.)([^ ]*?)-x(.)([^% ]*)%([^\| ]*)\|(.)([^% ]*)%([^- ]*)([^ ]*) ([^() ]*)\5\)/\($6$2-o$3$4%$5\|$6$7%$8$9 $10$8\)/g ) { }
      ####print( 'looking up', artnum, linenum, toknum )
      tree.c += '-xN%' + s + '|N%type' + Types[artnum,linenum,toknum][0] 
    return toknum+1
  for subtree in tree.ch:
    toknum = type( subtree, Types, artnum, linenum, toknum )
  return toknum

## read in Y model and store most probable type for each referent...
Types = collections.defaultdict(lambda: ('ERR',0.0))
for line in open(sys.argv[1]):
  m = re.match("^Y (.*)-(.*)(..)s : (.*) = (.*)$", line)
  if m is not None:
    artnum,linenum,toknum,y,prob = m.groups()
    ####print( 'storing', int(artnum), int(linenum), int(toknum) )
    if float(prob) > Types[int(artnum),int(linenum),int(toknum)][1]: Types[int(artnum),int(linenum),int(toknum)] = y,float(prob)

## for each article until eof...
artnum = 0
eof = False
while ( not eof ):

  ## for each sentence in article...
  for linenum,sent in enumerate(sys.stdin):
    if '!article' in sent: break

    ## read in and annotate trees with types...
    t = tree.Tree()
    t.read( sent )
    type( t, Types, artnum, linenum+1 )
    print t

  ## if all lines read, end article loop...
  else: eof = True

  artnum += 1

