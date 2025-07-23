import sys
import os
from numpy import nan
import math
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
                for j, w in enumerate(line.strip().split(" ")):
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
                    indexdata[(l, h, m)] = len(textdata)-1
                    k += 1
                    m += 1
                i += 1
                h += 1
    # for i in indexdata:
    #     print(i, indexdata[i])
    # exit()
    for _, row in df.iterrows():
        w = row["word"]
        l = int(row["trialid"]-1)
        h = int(row["sentnum"]-1)
        m = int(row["wordnum"])
        indexdata_idx = (l, h, m)
        # print(l,h,j,w)
        index = indexdata[indexdata_idx]

        # if row["line.word"] == 1:
        #     textdata[index]["startofline"] = 1
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
                word_cur = row["word"]
                word_id_cur = row["wordnum"]
                fdur_cur = row["dur"]
                fdur_fp = row["firstrun.dur"]
                fdur_gp = row["firstrun.gopast"]
                isblink = (row["blink"] == 1)

                l = int(row["trialid"]-1)
                h = int(row["sentnum"]-1)
                m = int(row["wordnum"])
                indexdata_idx = (l, h, m)
                index = indexdata[indexdata_idx]
                textdata_cur = textdata[index]

                correct = row["correct"]

                if textdata[index]['word'] != word_cur:
                    sys.stderr.write('WARNING: Saw mismatched words "%s" and "%s" at global position %d, file %s, line %d.\n' % (
                    textdata[index]['word'], word_cur, l, h, j))
                    sys.stderr.flush()
                
                out_cur = {
                    'subject': subject,
                    'docid': doc_id,
                    'correct': correct,
                    'fdurFP': fdur_fp,
                    'fdurGP': fdur_gp,
                    'fdurTT': fdur_cur,
                    'time': time
                }
                out_file.append(out_cur)
                out_cur.update(textdata[index])
                if not math.isnan(fdur_cur):
                    time += fdur_cur / 1000
                
            out += out_file

    if args.verbose:
        sys.stderr.write('Computing tabular output...\n')
        sys.stderr.flush()
    out = pd.DataFrame(out)

    out['resid'] = (out['time']*1000).astype('int')
    out['word'] = out['word'].str.replace('"', '')

    if args.verbose:
        sys.stderr.write('Writing output...\n')
        sys.stderr.flush()

    out.to_csv(sys.stdout, sep=' ', index=False, na_rep='NaN')
