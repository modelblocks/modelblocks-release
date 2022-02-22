import sys, json

itemmeasures = open(sys.argv[1])
# word time
time_info = open(sys.argv[2])

words = list()
start_times = list()
# header
time_info.readline()
for l in time_info:
    word, time = l.strip().split()
    words.append(word)
    start_times.append(time)

header_cols = itemmeasures.readline().strip().split()
# word sentid sentpos discid discpos docid
assert len(header_cols) == 6
header_cols.append('time')
print(' '.join(header_cols))

for i, l in enumerate(itemmeasures):
    w, sentid, sentpos, discid, discpos, docid = l.strip().split()
    assert w == words[i], 'itemmeasures word: {} time_info word: {}'.format(w, words[i])
    print(' '.join([w, sentid, sentpos, discid, discpos, docid, start_times[i]]))


