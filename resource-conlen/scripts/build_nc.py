import sys

linetoks_path = sys.argv[1]

with open(linetoks_path, 'r') as linetoks:
    h = sys.stdin.readline().strip().split()
    wix = h.index('word')

    l = sys.stdin.readline().strip()
    s = linetoks.readline().strip()
    i = 0

    out_cur = ''

    while s and l:
        w = l.split()[wix]
        t = s.split()[i]

        if w.lower() == t.lower():
            out_cur += l + '\n'
            i += 1
            if i >= len(s.split()):
                sys.stdout.write(out_cur)
                i = 0
                out_cur = ''
                s = linetoks.readline().strip()
        else:
            i = 0
            out_cur = ''

        l = sys.stdin.readline().strip()

