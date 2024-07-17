"""
Linear Regression script for self-paced reading data
Note: This script expects reponse data generatede with the modelblocks pipeline
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, log_loss
from scipy.stats import pearsonr
import sys

def map_data(resp_data, vec_data):
    """
    Map response data to vec data with docid, sentid, sentpos
    """
    X = []
    y = []

    for i in resp_data.index:

        # get docid, sentid, and sentpos from the datapoint
        docid = resp_data["docid"][i]
        sentid = resp_data["sentid"][i]
        sentpos = resp_data["sentpos"][i]

        # get fdur of the current datapoint
        y.append(resp_data["fdur"][i])

        # get corresponding vec representation in vec data
        vec_line = vec_data.loc[(vec_data["docid"] == docid) & (vec_data["sentid"] == sentid) & (vec_data["sentpos"] == sentpos)]
        only_vec = vec_line.drop(["word", "docid", "sentid", "sentpos"], axis=1)

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
    
    # load data as dataframe
    trn_resp_data = pd.read_csv(trn_resp_data_fn, sep=' ', skipinitialspace=True)
    test_resp_data = pd.read_csv(test_resp_data_fn, sep=' ', skipinitialspace=True)
    vec_data = pd.read_csv(vec_fn, sep=' ', skipinitialspace=True)

    X_train, y_train = map_data(trn_resp_data, vec_data)
    X_test, y_test = map_data(test_resp_data, vec_data)
    
    print("trn_fn: ", trn_resp_data_fn)
    print("trn_resp_data.shape: ", trn_resp_data.shape)
    print("X_train.shape: ", X_train.shape)
    print("y_train.shape: ", y_train.shape, "\n")
    print("test_fn: ", test_resp_data_fn)
    print("test_resp_data.shape: ", test_resp_data.shape)
    print("X_test.shape: ", X_test.shape)
    print("y_test.shape: ", y_test.shape, "\n")
    print("vec_fn: ", vec_fn, "\n")

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
