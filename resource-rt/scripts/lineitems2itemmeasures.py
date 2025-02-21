import sys

print('word sentid sentpos discid discpos corpus')

# 'id's start from 0, 'pos'es start from 1
discid = -1
sentid = 0
discpos = 1
corpus = sys.argv[1] if len(sys.argv) > 1 else 0
for line in sys.stdin:
    line = line.strip()
    if line == "!ARTICLE":
        discid += 1
        discpos = 1
        continue
    if line:
        sentpos = 1
        for w in line.split():
            print('%s %d %d %d %d %s' % (w, sentid, sentpos, discid, discpos, corpus))
            sentpos += 1
            discpos += 1
        sentid += 1
