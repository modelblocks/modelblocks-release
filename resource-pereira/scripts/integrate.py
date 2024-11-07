"""
@author: Hongao ZHU

@description:
    read from the .itemmeasures and the .csv files, integrate the information within,
    and output a comprehensive .csv file

@example usage:
    python3 integrate.py -C <input_csv_path> -S <input_itemmeasures_path> -O <output_csv_path>

"""


import argparse
import pandas as pd

def get_sentences_surprisal(csv_filename, measures_filename):
    # read sentences from .csv
    df = pd.read_csv(csv_filename)
    sentences = df['sentences']

    # read surprisals from .itemmeasures
    surprisal_values = []
    sent_ids = []
    discids = []
    with open(measures_filename, 'r', encoding='utf-8') as measuresfile:
        next(measuresfile)  # skip headers
        for line in measuresfile:
            word, surprisal, rolled, word2, sentid, sentpos, discid, discpos, wlen = line.strip().split()
            surprisal_values.append((word,float(surprisal)))
            sent_ids.append(sentid)
            discids.append(discid)

    # create surprisal lists for each sentence
    sentences_surprisal = []
    sentences_sentid = []
    sentences_discid = []
    count = 0
    for sentence in sentences:
        words = sentence.split()
        surprisal_list = []
        sentences_sentid.append(sent_ids[count%len(sent_ids)])
        sentences_discid.append(discids[count%len(discids)])
        for word in words:
            try:
                form, surprisal = surprisal_values[count%len(surprisal_values)]
                count += 1
                if word == form:
                    surprisal_list.append(surprisal)
            except IndexError:
                print(f"surprisal value mismatch: {word} != {form}; Linecount:{count}")
                surprisal_list.append(None)

        sentences_surprisal.append(surprisal_list)

    # append surprisal values, token numbers to the new .csv file
    df['totsurp'] = [sum(i) for i in sentences_surprisal]
    df['avgsurp'] = [sum(i)/len(i) for i in sentences_surprisal]
    df['lastwordsurp'] = [i[-1] for i in sentences_surprisal]
    df['sentlen'] = [len(sentence.split()) for sentence in sentences]
    df['sentid'] = sentences_sentid
    df['discid'] = sentences_discid

    return df


def main():
    parser = argparse.ArgumentParser(description='read from the .itemmeasures and the .csv files, integrate the information within, and output a comprehensive .csv file')
    parser.add_argument('--csv', '-C', type=str, default="test.csv", help='input .csv file path')
    parser.add_argument('--sent', '-S', type=str, default="test.unigram.itemmeasures", help='input .sentitems file path')
    parser.add_argument('--output', '-O', type=str, default="", help='output .csv file path, none for rewritting the input .csv file')

    args = parser.parse_args()
    df = get_sentences_surprisal(args.csv, args.sent)

    if args.output == "":
        df.to_csv(args.csv, index=False)
        print(f"result saved to {args.csv}")
    else:
        df.to_csv(args.output, index=False)
        print(f"result saved to {args.output}")

if __name__ == '__main__':
    main()
