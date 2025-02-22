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
    # read surprisals from .itemmeasures
    df_surp = pd.read_csv(measures_filename,sep=' ')
    grouped_df = df.groupby('corpus')
    grouped_df_surp = df_surp.groupby('corpus')
    assert grouped_df.groups.keys() == grouped_df_surp.groups.keys(), "The corpus names in .all-itemmeasures mismatched with those in .evmeasures"
    # create surprisal lists for each sentence
    sentences_surprisal = []
    sentences_sentid = []
    sentences_discid = []
    # group by the corpus to get surprisal iteratively
    for (name_df, group_df), (name_df_surp, group_df_surp) in zip(grouped_df, grouped_df_surp):
        print(f"Processing corpus {name_df}")
        # get sentences from the .evmeasures, get sentid and discid from the .itemmeasures
        sentences = group_df['sentences']
        sent_ids = group_df_surp['sentid'].tolist()
        discids = group_df_surp['discid'].tolist()
        for n in group_df_surp.columns:
            if 'surp' in n:
                surprisal_values=group_df_surp[n].tolist()
                break
        # sum up the surprisal by sentence
        count = 0
        for sentence in sentences:
            words = sentence.split()
            surprisal_list = []
            sentences_sentid.append(sent_ids[count%len(sent_ids)])
            sentences_discid.append(discids[count%len(discids)])
            for word in words:
                try:
                    surprisal = surprisal_values[count%len(surprisal_values)]
                    count += 1
                    surprisal_list.append(surprisal)
                except IndexError:
                    surprisal_list.append(None)

            sentences_surprisal.append(surprisal_list)
        print("***len(sentences):",len(sentences))
        print("***len(sentences_surprisal):",len(sentences_surprisal))
        print("***len(sentences_sentid):",len(sentences_sentid))
        print("***len(sentences_discid):",len(sentences_discid))
    # append surprisal values, token numbers to the new .csv file
    df['totsurp'] = [sum(i) for i in sentences_surprisal]
    df['avgsurp'] = [sum(i)/len(i) for i in sentences_surprisal]
    df['lastwordsurp'] = [i[-1] for i in sentences_surprisal]
    df['sentlen'] = [len(sentence.split()) for sentence in df['sentences']]
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
