import sys

print('word sentid sentpos')

sentid = 0
for line in sys.stdin:
    line = line.strip()
    if line:
        sentpos = 1
        for w in line.split():
            print('%s %d %d' % (w, sentid, sentpos))
            sentpos += 1
        sentid += 1
