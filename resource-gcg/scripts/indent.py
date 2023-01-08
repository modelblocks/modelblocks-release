import sys

OPENS = '(['
CLOSES = ')]'

Indents = [0]
ready = None
for nLine,line in enumerate( sys.stdin ):
  i = 0
  for c in line:
    if c in OPENS:
      sys.stdout.write( c )
      i += 1
      Indents.append(i)
      ready = '('
    elif c in CLOSES:
      if len(Indents) == 0:  sys.stdout.write( 'ERROR: Too many \'' + c + '\' in line ' + str(nLine) + ': ' + line + '\n' )
      sys.stdout.write( c )
      i += 1
      Indents.pop()
      ready = ')'
    elif c == ' ' and ready == '(':
      sys.stdout.write( c )
      i += 1
      Indents[-1] = i
      ready = None
    elif c in ' \t\n' and ready == ')':
      sys.stdout.write( '\n' + ' '*Indents[-1] )
      i = Indents[-1]
      ready = None
    else:
      sys.stdout.write( c )
      i += 1

