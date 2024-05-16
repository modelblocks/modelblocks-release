import sys

sentid=0

print('word sentid sentpos')

for line in sys.stdin:
    sentpos=1
    line = line.split()
    for word in line:
        print(word + ' ' + str(sentid) + ' ' + str(sentpos))
        sentpos += 1
    sentid += 1
