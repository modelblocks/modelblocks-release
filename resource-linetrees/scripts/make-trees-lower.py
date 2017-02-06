import sys

for line in sys.stdin.readlines():
    sline = line.split()
    for i in xrange(len(sline)):
        if sline[i][0].isalnum():
            sline[i] = sline[i].lower()
    sys.stdout.write(' '.join(sline)+'\n')
            
