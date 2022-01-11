import sys
import pandas

offset = 230

X1 = pandas.read_csv( sys.argv[1], index_col=False )
X2 = pandas.read_csv( sys.argv[2], index_col=False )
W  = pandas.read_csv( sys.argv[3], sep='\t', index_col=False )
X = X1.append(X2)

## Remove the 231st item, which was an empty token in the orig expt that resulted in the misalignment
X = X[ ~((X['item']==3) & (X['zone']==offset+1)) ]
## Shift the "zone" (index) down by two for all words
X[ 'zone' ] -= 2
## Shift the "zone" (index) down by an additional one for all words in story 3 following the empty token
X.loc[ (X['item']==3) & (X['zone']>=offset), 'zone' ] -= 1

## inner merge the word and RT arrays, using all shared columns as keys
X = X.merge(W,how='inner')  #,on=['item','zone'])

## Remove outlier RTs and subjs with too few correct responses. THIS IS THE PART WE SHOULDN'T DO.
#X = X[ (X['RT']>100) & (X['RT']<3000) & (X['correct']>4) ]

X.to_csv( sys.stdout, sep ='\t', index=False )


