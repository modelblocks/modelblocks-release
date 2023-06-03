import sys
import pandas
import csv
import re

X = pandas.read_csv( sys.stdin, sep=' ', index_col=False, quoting=0, converters={'word': str} ) #, na_values=default_missing ) #quotechar='|' ) # quoting=csv.QUOTE_ALL )  #, quotechar=None )  #, quoting=csv.QUOTE_NONE )
#X.to_csv( sys.stdout, sep=' ', quoting=0 )

## Find breaks...
X['prevword'] = X['word'].shift(1)
X['prevwordlastchar'] = [ str(x).strip()[-1] for x in X['prevword'] ]
X['prevwordpenultchar'] = [ str(x).strip()[-2] if len(str(x))>1 else ' ' for x in X['prevword'] ]
X['artbreak'] = 0
X.loc[ (X['iteminpage']==1) & (X['prevwordlastchar']=='.'), 'artbreak' ] = 1
X.loc[ (X['iteminpage']==1) & (X['prevwordlastchar']=='!'), 'artbreak' ] = 1
X.loc[ (X['iteminpage']==1) & (X['prevwordlastchar']=='?'), 'artbreak' ] = 1
X.loc[ (X['iteminpage']==1) & (X['prevwordpenultchar']=='.'), 'artbreak' ] = 1
X.loc[ (X['iteminpage']==1) & (X['prevwordpenultchar']=='!'), 'artbreak' ] = 1
X.loc[ (X['iteminpage']==1) & (X['prevwordpenultchar']=='?'), 'artbreak' ] = 1
## Exceptions...
X.loc[ (X['doc']==1) & (X['page']==28) & (X['artbreak']==1), 'artbreak' ] = 0
X.loc[ (X['doc']==4) & (X['page']==26) & (X['artbreak']==1), 'artbreak' ] = 0
X.loc[ (X['doc']==5) & (X['page']==31) & (X['artbreak']==1), 'artbreak' ] = 0
X.loc[ (X['doc']==6) & (X['page']==5) & (X['artbreak']==1), 'artbreak' ] = 0
X.loc[ (X['doc']==7) & (X['page']==20) & (X['artbreak']==1), 'artbreak' ] = 0
X.loc[ (X['doc']==7) & (X['page']==22) & (X['artbreak']==1), 'artbreak' ] = 0
X.loc[ (X['doc']==7) & (X['page']==40) & (X['artbreak']==1), 'artbreak' ] = 0
X.loc[ (X['doc']==8) & (X['page']==5) & (X['artbreak']==1), 'artbreak' ] = 0
X.loc[ (X['doc']==8) & (X['page']==39) & (X['artbreak']==1), 'artbreak' ] = 0
X.loc[ (X['doc']==9) & (X['page']==15) & (X['artbreak']==1), 'artbreak' ] = 0
X.loc[ (X['doc']==10) & (X['page']==16) & (X['artbreak']==1), 'artbreak' ] = 0
X.loc[ (X['doc']==11) & (X['page']==24) & (X['artbreak']==1), 'artbreak' ] = 0
X.loc[ (X['doc']==12) & (X['page']==17) & (X['artbreak']==1), 'artbreak' ] = 0
X.loc[ (X['doc']==12) & (X['page']==35) & (X['artbreak']==1), 'artbreak' ] = 0
X.loc[ (X['doc']==13) & (X['page']==21) & (X['artbreak']==1), 'artbreak' ] = 0
X.loc[ (X['doc']==13) & (X['page']==25) & (X['artbreak']==1), 'artbreak' ] = 0
X.loc[ (X['doc']==13) & (X['page']==29) & (X['artbreak']==1), 'artbreak' ] = 0
X.loc[ (X['doc']==14) & (X['page']==11) & (X['artbreak']==1), 'artbreak' ] = 0
X.loc[ (X['doc']==15) & (X['page']==21) & (X['artbreak']==1), 'artbreak' ] = 0
X.loc[ (X['doc']==16) & (X['page']==3) & (X['artbreak']==1), 'artbreak' ] = 0
X.loc[ (X['doc']==16) & (X['page']==33) & (X['artbreak']==1), 'artbreak' ] = 0
X.loc[ (X['doc']==16) & (X['page']==40) & (X['artbreak']==1), 'artbreak' ] = 0
X.loc[ (X['doc']==17) & (X['page']==24) & (X['artbreak']==1), 'artbreak' ] = 0
X.loc[ (X['doc']==17) & (X['page']==40) & (X['artbreak']==1), 'artbreak' ] = 0
X.loc[ (X['doc']==18) & (X['page']==22) & (X['artbreak']==1), 'artbreak' ] = 0
X.loc[ (X['doc']==19) & (X['page']==30) & (X['artbreak']==1), 'artbreak' ] = 0
## Number items...
X['article'] = X['artbreak'].cumsum()

#X.to_csv( sys.stdout, sep=' ', quoting=0 )
#exit(0)

## Print senttoks...
for disc in X['article'].unique():
  print( '!ARTICLE' )
  ## Concatenate all words in the same discourse...
  txt = ' '.join( X['word'][ X['article']==disc ].tolist() )
  ## Segment sentences using rules with task-dependent exceptions for cases like "'Why?' I asked"...
  for sent in re.split( '(?<!Mr.|Dr.)(?<!Mrs.)(?<=.[.!?]|[.!?][\'"]) +(?=[^a-z])(?!I asked)(?!I cried)(?!Abby excl)(?!Abby asked)', txt ):
    print( sent )

