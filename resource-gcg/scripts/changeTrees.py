import sys
import re
import tree

def change( t ):

  ## Fix adjectives combining with nouns to get noun phrases...
  if len(t.ch)==2 and re.search( '^.*-lM(-[lmnstuwxy].*)? N-aD(-[lmnstuwxy].*)? N(-[lmnstuwxy].*)?$', t.ch[0].c + ' ' + t.ch[1].c + ' ' + t.c ) != None:
    sys.stderr.write( 'Fixing ' + t.ch[0].c + ' ' + t.ch[1].c + ' => ' + t.c + '\n' )
    tmp = [ t.ch[0], t.ch[1] ]
    t.ch = [ tree.Tree() ]
    t.ch[0].c = 'N-aD'
    t.ch[0].ch = tmp

  ## Fix post-modifiers combining with nouns to get noun phrases...
  if len(t.ch)==2 and re.search( '^N-aD(-[lmnstuwxy].*)? .*-lM(-[lmnstuwxy].*)? N(-[lmnstuwxy].*)?$', t.ch[0].c + ' ' + t.ch[1].c + ' ' + t.c ) != None:
    sys.stderr.write( 'Fixing ' + t.ch[0].c + ' ' + t.ch[1].c + ' => ' + t.c + '\n' )
    tmp = t.ch[0]
    t.ch[0] = tree.Tree()
    t.ch[0].c = 'N'
    t.ch[0].ch = [ tmp ]

  ## Fix numbers being adjectives instead of quantifiers...
  if len(t.ch)==2 and 'A-aN-x-lM' in t.ch[0].c and len(t.ch[0].ch)==1 and len(t.ch[0].ch[0].ch)==1 and len(t.ch[0].ch[0].ch[0].ch)==0 and re.search('^[0-9]+$',t.ch[0].ch[0].ch[0].c) != None:
    sys.stderr.write( 'Fixing ' + t.ch[0].c + ' ' + t.ch[1].c + ' => ' + t.c + '\n' )
    t.ch[0].c = 'N-aD-b{N-aD}'
    t.ch[0].ch[0] = t.ch[0].ch[0].ch[0]
    tmp = t.ch[1]
    t.ch[1] = tree.Tree()
    t.ch[1].ch = [ tmp ]
    t.ch[1].c = 'N-aD-lU'
#    print( 'Fixed:', t )
  for st in t.ch:
    change( st )

## For each tree in input...
for line in sys.stdin:
#  print(line)
  t = tree.Tree()
  t.read( line )
  change( t )

  print( t if line!='!ARTICLE\n' else '!ARTICLE' )

