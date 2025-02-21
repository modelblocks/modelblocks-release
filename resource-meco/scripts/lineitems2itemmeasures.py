import sys

print('word sentid sentpos discid discpos')

# 'id's start from 0, 'pos'es start from 1
discid = 0
sentid = 0
discpos = 1
for line in sys.stdin:
    line = line.strip()
    if line == "!ARTICLE":
        discid += 1
        discpos = 1
        continue
    if line:
        sentpos = 1
        for w in line.split():
            print('%s %d %d %d %d' % (w.replace('"',''), sentid, sentpos, discid, discpos))
            sentpos += 1
            discpos += 1
        sentid += 1
