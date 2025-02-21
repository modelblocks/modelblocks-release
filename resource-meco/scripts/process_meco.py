import sys
import os
from numpy import nan
import pandas as pd
import argparse
import csv

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Extract eye-tracking time series from MECO eye-tracking corpus source.
    ''')
    argparser.add_argument('sentitems_path', help='Path to file with space-tokenized stimulus sentences in order, one per line.')
    argparser.add_argument('-v', '--verbose', action='store_true', help='Report verbose log')
    argparser.add_argument('-w', '--warn', action='store_true', help='Report warnings to stderr')
    args = argparser.parse_args()

    df = pd.read_csv(sys.stdin, sep=" ", quoting=csv.QUOTE_NONE)

    textdata = []
    indexdata = {}

    if args.verbose:
        sys.stderr.write('Processing stimulus data...\n')
        sys.stderr.flush()

    h = 0
    i = 0
    k = 0
    l = -1
    m = 1
    with open(args.sentitems_path, 'r') as f:
        for line in f:
            if line.strip() == "!ARTICLE":
                l += 1
                m = 1
                h = 0
            else:
                for j, w in enumerate(line.strip().split()):
                    textdata.append({
                        'word': w,
                        'sentid': i,
                        'sentpos': j + 1,
                        'discid': l,
                        'discpos': m,
                        'discsentid': h,
                        'startofsentence': int(j == 0),
                        'startoffile': int(h == 0 and j == 0),
                        'startofline': 0
                    })
                    indexdata[(l, h, j+1)] = len(textdata)-1
                    k += 1
                    m += 1
                i += 1
                h += 1

    for _, row in df.iterrows():
        w = row["word"]
        l = int(row["trialid"]-1)
        h = int(row["sentnum"]-1)
        j = int(row["sent.word"])
        indexdata_idx = (l, h, j)
        # print(l,h,j,w)
        index = indexdata[indexdata_idx]

        if row["line.word"] == 1:
            textdata[index]["startofline"] = 1
        # else:
        #     textdata[index]["startofline"] = 0

    # for i in textdata:
    #     print(i)

    for i in range(1, len(textdata)+1):
        if i == len(textdata):
            end_of_file = 1
            end_of_line = 1
            end_of_sentence = 1
        else:
            end_of_file = textdata[i]["startoffile"]
            end_of_line = textdata[i]["startofline"]
            end_of_sentence = textdata[i]["startofsentence"]

        textdata[i-1]["endoffile"] = end_of_file
        textdata[i-1]["endofline"] = end_of_line
        textdata[i-1]["endofsentence"] = end_of_sentence

    if args.verbose:
        sys.stderr.write('Processing fixation data...\n')
        sys.stderr.flush()

    out = []

    for subject in list(df["uniform_id"].unique()):
        subj_df = df[df["uniform_id"] == subject]
        # print(list(subj_df["trialid"].unique()))
        for doc_id in list(subj_df["trialid"].unique()):
            out_file = []
            subj_article_df = subj_df[subj_df["trialid"] == doc_id]
            # print(subj_article_df)
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

            npass = {}
            wordid2firstfix = {}

            nfix = 0

            for row_num, row in subj_article_df.iterrows():
                if row_num < 2:
                    continue

                word_cur = row["word"]
                word_id_cur = row["wordnum"]
                fdur_cur = row["dur"]
                isblink = (row["blink"] == 1)
                isoffscreen = (row["type"] == "out")

                l = int(row["trialid"]-1)
                h = int(row["sentnum"]-1)
                j = int(row["sent.word"])
                indexdata_idx = (l, h, j)
                index = indexdata[indexdata_idx]
                textdata_cur = textdata[index]

                fix_id = row["fixid"]
                correct = row["correct"]

                if not isoffscreen:
                    if word_id_cur in npass:
                        npass[word_id_cur] += 1
                    else:
                        npass[word_id_cur] = 1
                    if word_id_cur not in wordid2firstfix:
                        wordid2firstfix[word_id_cur] = nfix
                    if args.warn and textdata[index]['word'] != word_cur:
                        sys.stderr.write('WARNING: Saw mismatched words "%s" and "%s" at global position %d, file %s, line %d.\n' % (
                        textdata[index]['word'], word_cur, l, h, j))
                        sys.stderr.flush()
                    out_cur = {
                        'subject': subject,
                        'docid': doc_id,
                        'correct': correct,
                        'fixid': fix_id,
                        'fdurSP': fdur_cur,
                        'blinkbeforefix': int(prev_was_blink),
                        'blinkafterfix': 0,
                        'offscreenbeforefix': int(prev_was_offscreen),
                        'offscreenafterfix': 0,
                        'wdelta': word_id_cur - word_id_prev,
                        'npass': npass[word_id_cur],
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

                    out_cur.update(textdata[index])
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

    out['prevwasfix'] = (out['wdelta'] == 1).astype('int')
    out['nextwasfix'] = (out['wdelta'] == -1).astype('int')
    out['resid'] = (out['time']*1000).astype('int')
    out['word'] = out['word'].str.replace('"', '')

    if args.verbose:
        sys.stderr.write('Writing output...\n')
        sys.stderr.flush()

    out.to_csv(sys.stdout, sep=' ', index=False, na_rep='NaN')
