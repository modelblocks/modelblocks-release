import sys

docid = sys.argv[1]

header = sys.stdin.readline().strip()

print(header + ' docid')

for l in sys.stdin:
    print(l.strip() + ' ' + docid)
