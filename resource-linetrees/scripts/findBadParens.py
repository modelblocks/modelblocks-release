import sys

depth = 0
curStr = ''
for line in sys.stdin:
    for c in line:
        curStr += c
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                print curStr
                curStr = ''
            elif depth < 0:
                print "ERROR: Extra closing paren:"
                print curStr
                curStr = ''
                depth = 0
if depth > 0:
    print "ERROR: Missing closing paren:"
    print curStr
    