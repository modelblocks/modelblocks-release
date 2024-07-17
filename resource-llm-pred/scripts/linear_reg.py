"""
Linear regression script for fMRI data (Blank2014).
Note: This script expects the response data generated with the modelblocks pipeline
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, log_loss
from scipy.stats import pearsonr
import sys

def get_froi_data(resp_data, froi):
    froi_data = resp_data.loc[resp_data["fROI"] == froi]
    froi_data = froi_data[["subject", "docid", "time", "BOLD"]]
    return froi_data

def map_data(froi_data, vec_data):
    """
    Map response data to vec data with docid + time (sampleids don't work)
    """
    X = []
    y = []
    for i in froi_data.index:
        
        # get doc id and time for mapping response data (froi data) and vec data
        docid = froi_data["docid"][i]
        time = froi_data["time"][i]

        # get the y from froi data
        y.append(froi_data["BOLD"][i])

        # get feat values from vec_data
        vec_line = vec_data.loc[(vec_data["docid"] == docid) & (vec_data["time"] == time)]
        only_vec = vec_line.drop(["sentpos", "rate", "time", "tr", "docid", "sampleid"], axis=1)

        feat_vals = only_vec.values.flatten().tolist()
        X.append(feat_vals)

    return np.array(X), np.array(y)

def get_pred(X_train, y_train, X_test):
    """
    [X_train]: list of feature vectors (2D numpy array)
    [y_train]: list of BOLD values (1D numpy array)
    [X_test]: list of feature vectors (2D numpy array)
    """
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)

    return y_pred

if __name__ == "__main__":

    trn_resp_data_fn = sys.argv[1]
    test_resp_data_fn = sys.argv[2]
    vec_fn = sys.argv[3]
    froi = sys.argv[4]
    
    # load data as dataframe
    trn_resp_data = pd.read_csv(trn_resp_data_fn, sep=' ', skipinitialspace=True)
    test_resp_data = pd.read_csv(test_resp_data_fn, sep=' ', skipinitialspace=True)
    vec_data = pd.read_csv(vec_fn, sep=' ', skipinitialspace=True)

    if froi == "ALL":
        all_frois = ['LangLIFGorb', 'LangLPostTemp', 'LangLMFG', 'LangLAntTemp', 'LangLAngG', 'LangLIFG']

        X_train = []
        y_train = []
        for one_froi in all_frois:
            trn_froi_data = get_froi_data(trn_resp_data, one_froi)
            froi_X_train, froi_y_train = map_data(trn_froi_data, vec_data)
            X_train += list(froi_X_train)
            y_train += list(froi_y_train)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        X_test = []
        y_test = []
        for one_froi in all_frois:
            test_froi_data = get_froi_data(test_resp_data, one_froi)
            froi_X_test, froi_y_test = map_data(test_froi_data, vec_data)
            X_test += list(froi_X_test)
            y_test += list(froi_y_test)
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
    else:
        # get data of specific fROI
        trn_froi_data = get_froi_data(trn_resp_data, froi)
        X_train, y_train = map_data(trn_froi_data, vec_data)

        test_froi_data = get_froi_data(test_resp_data, froi)
        X_test, y_test = map_data(test_froi_data, vec_data)

    print()
    print("trn_fn: ", trn_resp_data_fn)
    print("trn_resp_data.shape: ", trn_resp_data.shape)
    print("X_train.shape: ", X_train.shape)
    print("y_train.shape: ", y_train.shape, "\n")
    print("test_fn: ", test_resp_data_fn)
    print("test_resp_data.shape: ", test_resp_data.shape)
    print("X_test.shape: ", X_test.shape)
    print("y_test.shape: ", y_test.shape, "\n")
    print("vec_fn: ", vec_fn, "\n")
    print("froi: ", froi)
    print()

    # get prediction
    y_pred = get_pred(X_train, y_train, X_test)

    # calculate Pearson's correlation
    corr, _ = pearsonr(y_test, y_pred)
    print('Pearsons correlation: %.8f'%(corr))

    # r2 score
    r2 = r2_score(y_test, y_pred)
    print('r2 score: %.8f'%(r2))

    # mean square error
    print('mean squared error: %.8f'%(mean_squared_error(y_test, y_pred)))

    
    
    
    
    

