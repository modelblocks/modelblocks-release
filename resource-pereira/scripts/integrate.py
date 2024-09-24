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
    with open(measures_filename, 'r', encoding='utf-8') as measuresfile:
        next(measuresfile)  # skip headers
        for line in measuresfile:
            word, surprisal = line.strip().split()[:2]
            surprisal_values.append((word,float(surprisal)))

    # create surprisal lists for each sentence
    sentences_surprisal = []
    for sentence in sentences:
        words = sentence.split()
        surprisal_list = []
        for word in words:
            try:
                form, surprisal = surprisal_values.pop(0)
                if word == form:
                    surprisal_list.append(surprisal)
            except IndexError:
                print(f"surprisal value mismatch: {word} != {form}")
                surprisal_list.append(None)

        sentences_surprisal.append(surprisal_list)

    # append surprisal values, token numbers to the new .csv file
    df['ave_surprisal'] = [sum(i) for i in sentences_surprisal]
    df['total_surprisal'] = [sum(i)/len(i) for i in sentences_surprisal]
    df['num_tokens'] = [len(sentence.split()) for sentence in sentences]

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
