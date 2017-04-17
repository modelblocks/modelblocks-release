import sys

punc = ["-LRB-", "-RRB-", ",", "-", ";", ":", "\'", "\'\'", '\"', "`", "``", ".", "!", "?", "*FOOT*", "-RRB-", ",", "-", ";", ":", ".", "!", "?"]

for line in sys.stdin:
    if line.strip() != '':
        wrds_in = line.strip().split()
        wrds_out = []
        for w in wrds_in:
            if w in punc:
                wrds_out[-1] += w
            else:
                wrds_out.append(w)

        print(' '.join(wrds_out))
