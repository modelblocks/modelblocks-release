import sys, re

addPunc = False
if len(sys.argv) > 1:
    addPunc = bool(sys.argv[1])

wrds = []

max = re.compile('            xmax = ([^ ]*)')
word = re.compile('            text = \"([^ "]*)')

with sys.stdin as tg:
    line = tg.readline()
    # Skip the first interval
    while line and not line.startswith('        intervals [2]:'):
        line = tg.readline()
    while line:
        while line and not line.startswith('            xmax ='):
            line = tg.readline()

        t = max.match(line).group(1)
        while line and not line.startswith('            text ='):
            line = tg.readline()
        w = word.match(line).group(1)
        wrds.append((w,t))
        line = tg.readline()

# For complicated reasons, sentid is zero-indexed and sentpos is 1-indexed
sentid = 0
sentpos = 1

wrds_out = []

# Handmade list of pause indices that are actual sentence breaks
break_ids = [
              1, 4, 5, 11, 16, 18, 31, 41, 45, 47, 49,
              50, 53, 55, 56, 60, 62, 63, 64, 66, 67, 69,
              70, 71, 75, 78, 83, 84, 86, 88, 90, 93,
              95, 98, 99, 100, 101, 103, 104, 105, 106,
              107, 108, 109, 110, 111, 114, 116, 121, 122,
              123, 129, 134, 136, 138, 140, 141, 145, 147,
              149, 156, 160, 165, 168, 170, 174, 177, 183,
              187, 188, 189, 192, 196, 198, 203, 204, 215,
              218, 227, 228, 229, 230, 233, 235, 236, 240,
              242, 246, 249, 254, 259, 262, 264, 266, 269,
              274, 275, 280, 287, 293, 295
            ]

break_punc = {}

if addPunc:
    for i in break_ids:
        break_punc[i] = '.'

    del break_punc[107]

    break_punc[5] = '?'
    break_punc[67] = '?'
    break_punc[100] = '?'
    break_punc[116] = '?'
    break_punc[121] = '?'
    break_punc[129] = '?'
    break_punc[242] = '?'

sp_id = 1

for i in xrange(len(wrds)):
    wrd = wrds[i]
    if wrd[0] in ['sp','']:
        if addPunc and sp_id in break_punc:
            wrds_out.append((break_punc[sp_id],str(sentid),str(sentpos),wrds[i-1][1]))
        if sp_id in break_ids or sp_id > break_ids[-1]:
            sentid += 1
            sentpos = 1
        sp_id += 1
    else:
        wrds_out.append((wrd[0],str(sentid),str(sentpos),wrd[1]))
        sentpos += 1

print('word sentid sentpos timestamp')
for wrd in wrds_out:
    print(' '.join(wrd))
