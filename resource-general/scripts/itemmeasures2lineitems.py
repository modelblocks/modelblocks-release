import sys

sentid_prev = 0
first_line = True
first_word = True

for line in sys.stdin:
    row = line.strip().split()
    if first_line:
        word_ix = row.index('word')
        sentid_ix = row.index('sentid')
        first_line = False
    else:
        word = row[word_ix]
        sentid = row[sentid_ix]
        if first_word:
            delim = ''
            first_word = False
        elif sentid == sentid_prev:
            delim = ' '
        else:
            delim = '\n'

        sentid_prev = sentid
        
        sys.stdout.write(delim + word)
sys.stdout.write('\n')
