import sys, os, re, collections
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
import tree

for a in sys.argv:
  if a=='-d':
    gcgtree.VERBOSE = True
    semcuegraph.VERBOSE = True

################################################################################

def compare( tG, tH ):
  if tG.c == tH.c and len( tG.words() ) == len( tH.words() ) and len( tG.ch ) == len( tH.ch ) and ( len(tH.ch) < 2 or len( tG.ch[0].words() ) == len( tH.ch[0].words() ) ):
    sys.stdout.write( ' (' if len(tG.ch)>0 else ' ' + tG.c )
    for i in range( len(tH.ch) ):
      compare( tG.ch[i] if i<len(tG.ch) else tree.Tree(), tH.ch[i] )
    if len(tG.ch)>0: sys.stdout.write( ')' )
  else:
    sys.stdout.write( ' <***GOLD***> ' + str(tG) + ' <**HYPOTH**> ' + str(tH) + ' <**********>' )

################################################################################

fG = open( sys.argv[1] )
fH = open( sys.argv[2] )

for i,lineH in enumerate(fH):

  tH = tree.Tree( )
  tH.read( lineH )

  lineG = fG.readline( )
  tG = tree.Tree( )
  tG.read( lineG )

  sys.stdout.write( 'TREE ' + str(i+1) + ':\n' )
  compare( tH, tG )
  sys.stdout.write( '\n' )

