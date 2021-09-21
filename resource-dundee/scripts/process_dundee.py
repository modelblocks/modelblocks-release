import sys
import os
from numpy import nan
import pandas as pd
import argparse

#sys.stdin.reconfigure(encoding='latin-1',errors='replace') #'utf-8',errors='replace') #'ignore')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Extract eye-tracking time series from Dundee eye-tracking corpus source.
    ''')
    argparser.add_argument('dundee_dir', help='Path to directory containing Dundee files.')
    argparser.add_argument('lineitems_path', help='Path to file with space-tokenized stimulus sentences in order, one per line.')
    argparser.add_argument('-v', '--verbose', action='store_true', help='Report verbose log')
    argparser.add_argument('-w', '--warn', action='store_true', help='Report warnings to stderr')
    args = argparser.parse_args()

    textdata = []

    if args.verbose:
        sys.stderr.write('Processing stimulus data...\n')
        sys.stderr.flush()

    i = 0
    k = 0
    l = -1
    m = 1
    with open(args.lineitems_path, 'r') as f:
        for line in f:
            if line.strip() == "!ARTICLE":
                l += 1
                m = 1
            else:
                for j, w in enumerate(line.strip().split()):
                    textdata.append({
                        'word': w,
                        'sentid': i,
                        'sentpos': j + 1,
                        'discid': l,
                        'discpos': m,
                        'startofsentence': int(j == 0)
                    })
                    k += 1
                    m += 1
                i += 1

    k = 0
    start_ix = []
    for p in sorted([x for x in os.listdir(args.dundee_dir) if x.endswith('wrdp.dat')]):
        start_ix.append(k)
        with open(args.dundee_dir + '/' + p, 'r', encoding='latin-1') as f:
            for i, line in enumerate(f):
                line = line.replace('(', '-LRB-').replace(')', '-RRB-')
                fields = line.strip().split()
                w = fields[0]
                doc_id = int(fields[1]) - 1
                screen_id = int(fields[2]) - 1
                line_id = int(fields[3]) - 1
                word_pos_in_line = int(fields[4]) - 1
                word_pos_in_screen = int(fields[5]) - 1
                word_pos_in_text = int(fields[12]) - 1

                if word_pos_in_text == 0:
                    start_of_file = True
                    start_of_screen = True
                    start_of_line = True
                elif word_pos_in_screen == 0:
                    start_of_file = False
                    start_of_screen = True
                    start_of_line = True
                elif word_pos_in_line == 0:
                    start_of_file = False
                    start_of_screen = False
                    start_of_line = True
                else:
                    start_of_file = False
                    start_of_screen = False
                    start_of_line = False

                if args.warn and textdata[k]['word'] != w:
                    sys.stderr.write('WARNING: Saw mismatched words "%s" and "%s" at position %d.\n' % (textdata[k]['word'], w, k))
                    sys.stderr.flush()

                textdata[k]['startoffile'] = int(start_of_file)
                textdata[k]['startofscreen'] = int(start_of_screen)
                textdata[k]['startofline'] = int(start_of_line)

                k += 1

    for kp1 in range(1, len(textdata) + 1):
        if kp1 == len(textdata):
            end_of_file = 1
            end_of_screen = 1
            end_of_line = 1
            end_of_sentence = 1
        else:
            end_of_file = textdata[kp1]['startoffile']
            end_of_screen = textdata[kp1]['startofscreen']
            end_of_line = textdata[kp1]['startofline']
            end_of_sentence = textdata[kp1]['startofsentence']

        textdata[kp1-1]['endoffile'] = end_of_file
        textdata[kp1-1]['endofscreen'] = end_of_screen
        textdata[kp1-1]['endofline'] = end_of_line
        textdata[kp1-1]['endofsentence'] = end_of_sentence


    if args.verbose:
        sys.stderr.write('Processing fixation data...\n')
        sys.stderr.flush()

    out = []

    # Loop through fixations in order
    for i, p in enumerate(sorted([x for x in os.listdir(args.dundee_dir) if x.endswith('ma1p.dat')])):
        out_file = []
        with open(args.dundee_dir + '/' + p, 'r', encoding='latin-1') as f:
            subject = p[:2]
            doc_id = int(p[2:4]) - 1
            word_id_prev = -1
            max_word_id = -1
            time = 0

            fdurSP = 0
            fdurSPsummed = 0
            fdurFP = 0
            fdurGP = 0
            fdurTT = 0

            fp_cur = None
            fp_blink_cur = None
            gp_cur = None
            gp_blink_cur = None
            sp_cur = None
            tt_cur = None

            prev_was_blink = False
            prev_was_offscreen = False
            blinkFP = False
            blinkGP = False

            s = start_ix[doc_id]
            npass = {}
            wordid2firstfix = {}

            nfix = 0

            for j, line in enumerate(f):
                line = line.replace('(', '-LRB-').replace(')', '-RRB-').replace('"', "'")
                if j > 0:
                    fields = line.strip().split()
                    word_cur = fields[0]
                    word_id_cur = int(fields[6]) - 1
                    fdur_cur = float(fields[7])
                    isfix = False
                    isblink = False
                    isoffscreen = False
                    if word_cur.startswith('*'):
                        if word_cur == '*Blink':
                            isblink = True
                        elif word_cur == '*Off-screen':
                            isoffscreen = True
                        else:
                            raise ValueError('Unrecognized star (*) token: %s' % word_cur)
                    else:
                        if word_id_cur >= 0:
                            isfix = True

                    if isfix:
                        k = s + word_id_cur
                        if k in npass:
                            npass[k] += 1
                        else:
                            npass[k] = 1
                        if word_id_cur not in wordid2firstfix:
                            wordid2firstfix[word_id_cur] = nfix
                        if args.warn and textdata[k]['word'] != word_cur:
                            sys.stderr.write('WARNING: Saw mismatched words "%s" and "%s" at global position %d, file %s, line %d.\n' % (
                            textdata[k]['word'], word_cur, k, p, j))
                            sys.stderr.flush()
                        out_cur = {
                            'subject': subject,
                            'docid': doc_id,
                            'fdurSP': fdur_cur,
                            'blinkbeforefix': int(prev_was_blink),
                            'blinkafterfix': 0,
                            'offscreenbeforefix': int(prev_was_offscreen),
                            'offscreenafterfix': 0,
                            'wdelta': word_id_cur - word_id_prev,
                            'npass': npass[k],
                            'inregression': int(word_id_cur < max_word_id),
                            'time': time
                        }
                        out_file.append(out_cur)

                        tt_cur = out_file[wordid2firstfix[word_id_cur]]
                        if word_id_cur != word_id_prev:
                            sp_cur = out_cur
                            sp_blink_cur = out_cur
                        if word_id_cur > max_word_id:
                            fp_cur = out_cur
                            gp_cur = out_cur
                            fp_blink_cur = out_cur
                            gp_blink_cur = out_cur
                        elif word_id_cur < max_word_id:
                            fp_cur = None
                            fp_blink_cur = None

                        out_cur.update(textdata[k])
                        word_id_prev = word_id_cur
                        prev_was_blink = False
                        prev_was_offscreen = False
                        max_word_id = max(max_word_id, word_id_cur)

                        nfix += 1

                    else:
                        prev_was_blink = prev_was_blink or isblink
                        prev_was_offscreen = prev_was_offscreen or isoffscreen
                        if word_id_cur > 0 and isblink:
                            out_file[-1]['blinkafterfix'] = 1
                        if word_id_cur > 0 and isoffscreen:
                            out_file[-1]['offscreenafterfix'] = 1
                            sp_cur = None
                            sp_blink_cur = None
                            fp_cur = None
                            fp_blink_cur = None
                            gp_cur = None
                            gp_blink_cur = None

                    if sp_cur is not None:
                        if 'fdurSPsummed' in sp_cur:
                            sp_cur['fdurSPsummed'] += fdur_cur
                        else:
                            sp_cur['fdurSPsummed'] = fdur_cur

                    if sp_blink_cur is not None:
                        if 'blinkdurSPsummed' not in sp_blink_cur:
                            sp_blink_cur['blinkdurSPsummed'] = 0
                            sp_blink_cur['blinkduringSPsummed'] = 0
                        if isblink:
                            sp_blink_cur['blinkdurSPsummed'] += fdur_cur
                            sp_blink_cur['blinkduringSPsummed'] = 1

                    if fp_cur is not None:
                        if 'fdurFP' in fp_cur:
                            fp_cur['fdurFP'] += fdur_cur
                        else:
                            fp_cur['fdurFP'] = fdur_cur

                    if fp_blink_cur is not None:
                        if 'blinkdurFP' not in fp_blink_cur:
                            fp_blink_cur['blinkdurFP'] = 0
                            fp_blink_cur['blinkduringFP'] = 0
                        if isblink:
                            fp_blink_cur['blinkdurFP'] += fdur_cur
                            fp_blink_cur['blinkduringFP'] = 1

                    if gp_cur is not None:
                        if 'fdurGP' in gp_cur:
                            gp_cur['fdurGP'] += fdur_cur
                        else:
                            gp_cur['fdurGP'] = fdur_cur

                    if gp_blink_cur is not None:
                        if 'blinkdurGP' not in gp_blink_cur:
                            gp_blink_cur['blinkdurGP'] = 0
                            gp_blink_cur['blinkduringGP'] = 0
                        if isblink:
                            gp_blink_cur['blinkdurGP'] += fdur_cur
                            gp_blink_cur['blinkduringGP'] = 1

                    if tt_cur is not None:
                        if 'fdurTT' in tt_cur:
                            tt_cur['fdurTT'] += fdur_cur
                        else:
                            tt_cur['fdurTT'] = fdur_cur

                    time += fdur_cur / 1000

        out += out_file

    if args.verbose:
        sys.stderr.write('Computing tabular output...\n')
        sys.stderr.flush()
    out = pd.DataFrame(out)
    out.docid += 1

    out['prevwasfix'] = (out['wdelta'] == 1).astype('int')
    out['nextwasfix'] = (out['wdelta'] == -1).astype('int')
    out['resid'] = (out['time']*1000).astype('int')

    if args.verbose:
        sys.stderr.write('Writing output...\n')
        sys.stderr.flush()

    toprint = [
        'word',
        'subject',
        'docid',
        'discpos',
        'discid',
        'sentpos',
        'sentid',
        'resid',
        'time',
        'wdelta',
        'prevwasfix',
        'nextwasfix',
        'startoffile',
        'endoffile',
        'startofscreen',
        'endofscreen',
        'startofline',
        'endofline',
        'startofsentence',
        'endofsentence',
        'blinkbeforefix',
        'blinkafterfix',
        'offscreenbeforefix',
        'offscreenafterfix',
        'inregression',
        'fdurSP',
        'fdurSPsummed',
        'blinkdurSPsummed',
        'blinkduringSPsummed',
        'fdurFP',
        'blinkdurFP',
        'blinkduringFP',
        'fdurGP',
        'blinkdurGP',
        'blinkduringGP',
        'fdurTT'
    ]

    out[toprint].to_csv(sys.stdout, sep=' ', index=False, na_rep='NaN')

