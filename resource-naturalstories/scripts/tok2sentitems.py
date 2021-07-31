import sys
import pandas
import re

X = pandas.read_csv(sys.stdin,sep='\t')

for disc in X['item'].unique():
  print( '!ARTICLE' )
  ## Concatenate all words in the same discourse...
  txt = ' '.join( X['word'][ X['item']==disc ].tolist() )
  ## Segment sentences using rules with task-dependent exceptions for cases like "'Why?' I asked"...
  for sent in re.split( '(?<!Mr.|Dr.)(?<!Mrs.)(?<=.[.!?]|[.!?][\'"]) +(?=[^a-z])(?!I asked)(?!I cried)(?!Abby excl)(?!Abby asked)', txt ):
    print( sent )
