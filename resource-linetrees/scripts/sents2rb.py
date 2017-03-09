import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
import tree

t = tree.Tree()
left = '(1 '
right = ') '
for line in sys.stdin:
    line = line.strip()
    line = line.lower().split()
    string = ''
    for word in line[::-1]:
        w = left+word+right
        if string:
            string = left+w+string+right
        else:
            string = w
    t.read(string)
    print(t)
