import sys

skip=True

out = []
last_sentid = 0

for line in sys.stdin:
    if line.strip() != '' and not skip:
        row = line.strip().split()
        if int(row[1]) > last_sentid:
            print(' '.join(out))
            out = [row[0]]
            last_sentid = int(row[1])
        else:
            out.append(line.strip().split()[0])
    else:
        skip = False

if len(out) > 0:
    print(' '.join(out))
