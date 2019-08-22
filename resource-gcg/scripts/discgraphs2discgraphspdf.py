import sys


## For each discourse graph...
for line in sys.stdin:

  line = line.rstrip()

  print( 'digraph G {' )

  ## For each assoc...
  for assoc in sorted( line.split(' ') ):
    src,lbl,dst = assoc.split(',')
 
    print( '"' + src + '" -> "' + dst + '" [label="' + lbl + '"];' )

  print( '}' )

