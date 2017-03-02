import sys

def correct_parens(s):
  s = s.strip()
  l = 0
  r = 0
  for c in s:
    if c == '(':
      l += 1
    elif c == ')':
      r += 1
  paren_diff = l-r
  if paren_diff < 0:
#    print('Too many right parens!')
#    print(l)
#    print(r)
#    print(s + '|')
    s = s[:paren_diff] 
#    print(s + '|')
#    print('')
  elif paren_diff > 0:
#    print('Too many left parens!')
#    print(l)
#    print(r)
#    print(s + '|')
    s += (')'*paren_diff)
#    print(s + '|')
#    print('')
  return s

linetree = ''
for line in sys.stdin:
  if line.strip().startswith('(ROOT'):
    linetree = correct_parens(linetree)
    if linetree.strip() != '':
      print(linetree)
    linetree = line.strip()
  else:
    linetree += ' ' + line.strip()

