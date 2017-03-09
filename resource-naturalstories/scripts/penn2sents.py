import sys, os, re
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resource-gcg', 'scripts'))
import tree

trace = re.compile('\*')

t = tree.Tree()
for line in sys.stdin:
    if (line.strip() !='') and (line.strip()[0] != '%'):
        t.read(line)
        all_words = t.words()
        out = ''
        for w in all_words:
            if not trace.match(w):
                if out != '':
                    out += ' '
                out += w
        print out
