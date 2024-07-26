"""
@author: Hongao ZHU

@description:
    extract the human response data from the .mat files and rearrange them in .csv format
    this program averages the voxel responses for the 12 rois relavant to language

@example usage:
    cd ~/zhu.3986
    python3 scripts/transform_mat.py -I /fs/project/schuler.77/corpora/original/english/pereira2018/M02/data_384sentences.mat -O data

"""


import scipy.io
import numpy as np
import pandas as pd
import argparse

def transform_mat_data(mat_file_dir="P01/data_384sentences.mat", save_csv_dir="P01"):
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

    # load sentences
    sentences = [s[0][0] for s in data['keySentences']]

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
    print(f"Done getting voxel indexes.")

    # concat the voxels (the first column being the sentences per se)
    ave_columns = [np.array(sentences)]
    for roi in voxel_index.keys():
        ave = voxels[:, voxel_index[roi]].mean(axis=1)
        ave = np.array(ave) if type(ave) is list else ave
        ave_columns.append(ave)
    matrix = np.column_stack(ave_columns)

    # save the sentences and voxels
    df = pd.DataFrame(matrix,columns=['sentences'] + list(voxel_index.keys()))
    output_path = f"{save_csv_dir}/{'_'.join(mat_file_dir.split('/')[-2:-1])}.csv"
    df.to_csv(output_path, index=False)
    print(f"processed data saved to: {output_path}")

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Transform the .mat data into .csv format')
    parser.add_argument('--input', '-I', type=str, default="/fs/project/schuler.77/corpora/original/english/pereira2018/M02/data_384sentences.mat", help='input file path')
    parser.add_argument('--output', '-O', type=str, default="pereira_data", help='output file directory')

    # Look for surprisal_dict.json
    args = parser.parse_args()
    transform_mat_data(args.input, args.output)

if __name__ == '__main__':
    main()
