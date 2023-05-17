import sys

OPENS = '(['
CLOSES = ')]'

Indents = [0]
followstate = None
## For each line...
for nLine,line in enumerate( sys.stdin ):
  i = 0
  ## For each character...
  for c in line:
    ## If open delimiter, record tab...
    if c in OPENS:
      sys.stdout.write( c )
      i += 1
      if followstate == '(':  Indents[-1] = i-1
      Indents.append(i)
      followstate = '('
    ## If close delimiter, set follow state...
    elif c in CLOSES:
      if len(Indents) == 0:  sys.stdout.write( 'ERROR: Too many \'' + c + '\' in line ' + str(nLine) + ': ' + line + '\n' )
      sys.stdout.write( c )
      i += 1
      Indents.pop()
      followstate = ')'
    ## If whitespace following close delimiter...
    elif c in ' \t\n' and followstate == ')':
      sys.stdout.write( '\n' + ' '*Indents[-1] )
      i = Indents[-1]
      followstate = None
    else:
      sys.stdout.write( c )
      i += 1

