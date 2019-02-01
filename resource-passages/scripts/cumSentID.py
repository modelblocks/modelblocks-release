import sys

base_id = 0
last_id = 0
header = True

for line in sys.stdin:
    if line.strip() != '':
        row = line.strip().split()
        if header==True:
            sentid_col = row.index('sentid')
            header = False
        else:
            try:
                cur_id = int(row[sentid_col])
            except:
                continue
            if cur_id < last_id:
                assert cur_id == 0, 'Sentid dropped but not to 0 (dropped to %d).' %cur_id
                base_id += last_id + 1
                last_id = cur_id
            elif cur_id > last_id:
                last_id = cur_id
            row[sentid_col] = str(base_id + cur_id)
        print(' '.join(row))
            
