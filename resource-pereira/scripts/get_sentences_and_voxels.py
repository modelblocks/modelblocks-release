"""
@author: Hongao ZHU

@description:
    extract the human response data from the .mat files and rearrange them in .csv format,
    meanwhile producing a .sentitems file

    this program averages the voxel responses for the 12 rois relavant to language

@example usage:
    python3 transform_mat.py -I <input_mat_path> -C <output_csv_path> -S <output_sentitems_path>

"""


import scipy.io
import numpy as np
import pandas as pd
import argparse

def transform_mat_data(mat_file_dir="P01/data_384sentences.mat", save_csv_dir="save.csv", save_sent_dir="M02_data_384sentences.sentitems"):
    # load the .mat file
    try:
        data = scipy.io.loadmat(mat_file_dir)
        print(f"Successfully loaded {mat_file_dir}...")
        #for key in data.keys():
        #    print(key)
    except Exception as e:
        print(f"Error handling {mat_file_dir}: {e}")

    # load voxels (each column is a voxel)
    voxels = np.stack(data['examples_passagesentences'],axis=0)

    # load sentences, calculate sentence order in passages
    sentences = [s[0][0] for s in data['keySentences']]
    labelsPassageForEachSentence = [s for s in data['labelsPassageForEachSentence'][0]] if len(data['labelsPassageForEachSentence'])==1 else [s[0] for s in data['labelsPassageForEachSentence']]

    sentence_order_in_passage = []
    passage_sentence_count = {}
    for label in labelsPassageForEachSentence:
        if label not in passage_sentence_count:
            passage_sentence_count[label] = 0
        passage_sentence_count[label] += 1
        sentence_order_in_passage.append(passage_sentence_count[label])

    # load rois
    index_languageLH = -1
    index_languageRH = -1
    for index, region in enumerate(data['meta']['atlases'][0][0][0]):
        if region == 'languageLH':
            index_languageLH = index
        if region == 'languageRH':
            index_languageRH = index
    assert index_languageLH != -1 and index_languageRH != -1, "languageLH or languageRH not found!!!"

    roi_languageLH = [roi[0][0] for roi in data['meta']['rois'][0][0][0][index_languageLH]]
    roi_languageRH = [roi[0][0] for roi in data['meta']['rois'][0][0][0][index_languageRH]]

    assert roi_languageLH and roi_languageRH, "ROIs not found within languageLH or languageRH!!!"

    # find corresponding voxels (by index) for each roi
    voxel_index = dict()
    for brain_region in [roi_languageLH, roi_languageRH]:
        for i, roi in enumerate(brain_region):
            indexes = data['meta']['roiColumns'][0][0][0][index_languageLH][i][0]
            voxel_index[roi] = indexes
    print(f"Done getting voxel indexes...")

    # concat the voxels (the first column being the sentences per se)
    ave_columns = [np.array(sentences), np.array(sentence_order_in_passage)]
    for roi in voxel_index.keys():
        ave = voxels[:, voxel_index[roi]].mean(axis=1)
        ave = np.array(ave) if type(ave) is list else ave
        ave_columns.append(ave)
    matrix = np.column_stack(ave_columns)

    # save the sentences and voxels
    df = pd.DataFrame(matrix,columns=['sentences', 'sentence_order_in_passage'] + list(voxel_index.keys()))
    df.to_csv(save_csv_dir, index=False)
    print(f"processed data saved to: {save_csv_dir}")

    with open(save_sent_dir, 'w') as file:
        for sentence, order in zip(sentences, sentence_order_in_passage):
            if order == 1: # if the first sentence in a paragraph, insert "!ARTICLE"
                file.write('!ARTICLE\n')
            file.write(sentence + '\n')
    print(f"processed sentences saved to: {save_sent_dir}")

def main():
    parser = argparse.ArgumentParser(description='Transform the .mat data into .csv format')
    parser.add_argument('--input', '-I', type=str, default="/fs/project/schuler.77/corpora/original/english/pereira2018/M02/data_384sentences.mat", help='input .mat file path')
    parser.add_argument('--csv', '-C', type=str, default="M02_data_384sentences.csv", help='output .csv file path')
    parser.add_argument('--sent', '-S', type=str, default="M02_data_384sentences.sentitems", help='output .sentitems file path')

    args = parser.parse_args()
    transform_mat_data(args.input, args.csv, args.sent)

if __name__ == '__main__':
    main()