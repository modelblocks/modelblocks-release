import sys, re, string


lineitems = sys.argv[1]
textgrid_dir = sys.argv[2]

onset = re.compile('            xmin = ([^ ]*)')
offset = re.compile('            xmax = ([^ ]*)')

timestamp = onset

word = re.compile('            text = \" *([^ "]*)')

doc_list = ['Boar', 'Aqua', 'MatchstickSeller', 'KingOfBirds', 'Elvis', 'MrSticky', 'HighSchool', 'Roswell', 'Tourettes']

def get_onset_times(textgrid):
    wrds = []
    with open(textgrid, 'r') as f:
        line = f.readline()
        while line:
            if line.startswith('    item [2]:'):
                break
            while line and not line.startswith('            xmin ='):
                line = f.readline()
    
            t = timestamp.match(line).group(1)
            while line and not line.startswith('            text ='):
                line = f.readline()
            w = word.match(line).group(1)
            w = w.lower()
            if w in ['', '<s>', '</s>', '<s']:
                exclude = True
            #elif w == 'the' and t == '291.5030594213008':
            elif w == 'the' and t == '291.4':
                exclude = True
            #elif w == 'worry' and t == '140.04999':
            elif w == 'worry' and t == '140.06966742907173':
                exclude = True
            else:
                exclude = False
            w = w.translate(None, string.punctuation)
            if w == 'shrilll':
                w = 'shrill'
            if w == 'noo':
                w = 'no'
            if w == 'yess':
                w = 'yes'
            if not exclude:
                wrds.append((w,t))
            line =f.readline()
    return wrds

with open(lineitems, 'r') as li:
    headers = li.readline().strip().split() + ['time']
    print(' '.join(headers))
    col_map = {}
    for i in range(len(headers)):
        col_map[headers[i]] = i
    li_line = li.readline()
    tg = []
    for i in range(10):
        f = textgrid_dir + '/' + str(i+1) + '.TextGrid'
        tg.append(get_onset_times(f))
    tg_ix = 0
    tg_pos = 0
    while li_line:
        if li_line:
            vals = li_line.strip().split()
            word = vals[col_map['word']]
            word = word.replace('-LRB-', '')
            word = word.replace('-RRB-', '')
            word = word.lower().translate(None, string.punctuation)
            tg_word = tg[tg_ix][tg_pos][0]
            while len(word) > len(tg_word):
                tg_pos += 1
                tg_word += tg[tg_ix][tg_pos][0]
            assert word == tg_word, 'Mismach in document %d: %s vs. %s' %(tg_ix, word, tg_word)
            print(li_line.strip() + ' ' + tg[tg_ix][tg_pos][1])
            tg_pos += 1
            if tg_pos >= len(tg[tg_ix]):
                tg_ix += 1
                tg_pos = 0
        li_line = li.readline()
    assert tg_ix == len(tg) and tg_pos == 0
        
        
