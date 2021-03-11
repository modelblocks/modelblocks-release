import sys, argparse, pandas as pd, numpy as np
from mvpa2.misc.data_generators import double_gamma_hrf as hrf


def compute_history_intervals(X, y, series_ids):
    """
    Compute row indices in **X** of initial and final impulses for each element of **y**.

    :param X: ``pandas`` ``DataFrame``; impulse (predictor) data.
    :param y: ``pandas`` ``DataFrame``; response data.
    :param series_ids: ``list`` of ``str``; column names whose jointly unique values define unique time series.
    :return: 2-tuple of ``numpy`` vectors; first and last impulse observations (respectively) for each response in **y**
    """

    m = len(X)
    n = len(y)

    time_X = np.array(X.time)
    time_y = np.array(y.time)

    id_vectors_X = []
    id_vectors_y = []

    for i in range(len(series_ids)):
        col = series_ids[i]
        id_vectors_X.append(np.array(X[col]))
        id_vectors_y.append(np.array(y[col]))
    id_vectors_X = np.stack(id_vectors_X, axis=1)
    id_vectors_y = np.stack(id_vectors_y, axis=1)

    y_cur_ids = id_vectors_y[0]

    first_obs = np.zeros(len(y)).astype('int32')
    last_obs = np.zeros(len(y)).astype('int32')

    # i iterates y
    i = 0
    # j iterates X
    j = 0
    start = 0
    end = 0
    epsilon = np.finfo(np.float32).eps
    while i < n:
        # Check if we've entered a new series in y
        if (id_vectors_y[i] != y_cur_ids).any():
            start = end = j
            X_cur_ids = id_vectors_X[j]
            y_cur_ids = id_vectors_y[i]

        # Move the X pointer forward until we are either in the same series as y or at the end of the table.
        # However, if we are already at the end of the current time series, stay put in case there are subsequent observations of the response.
        if j == 0 or (j > 0 and (id_vectors_X[j-1] != y_cur_ids).any()):
            while j < m and (id_vectors_X[j] != y_cur_ids).any():
                j += 1
                start = end = j

        # Move the X pointer forward until we are either at the end of the series or have moved later in time than y
        while j < m and time_X[j] <= (time_y[i] + epsilon) and (id_vectors_X[j] == y_cur_ids).all():
            j += 1
            end = j

        first_obs[i] = start
        last_obs[i] = end

        i += 1

    return first_obs, last_obs


def convolve(X, y, first_obs, last_obs, columns):
    """
    Convolve predictors in **X** into timestamps in **y** using canonical HRF.

    :param X: ``pandas`` ``DataFrame``; impulse (predictor) data.
    :param y: ``pandas`` ``DataFrame``; response data.
    :param y: ``numpy`` ``array``; integer indices of first observation in X from time series in y.
    :param y: ``numpy`` ``array``; integer indices of last observation in X preceding each y in the same time series..
    :param series_ids: ``list`` of ``str``; column names whose jointly unique values define unique time series.
    :param columns: ``list`` of ``str``; column names to convolve.
    :return: ``pandas`` ``DataFrame``; matrix of convolved predictors
    """

    time_X = np.array(X.time)
    time_y = np.array(y.time)

    X = np.array(X[columns])
    X_conv = np.zeros((y.shape[0], X.shape[1]))

    for i in range(len(y)):
        s, e = first_obs[i], last_obs[i]
        hrf_weights = hrf(time_y[i] - time_X[s:e])[..., None]
        X_conv[i] = (X[s:e] * hrf_weights).sum(axis=0)

    X_conv = pd.DataFrame(X_conv, columns=columns, index=y.index)
    y = pd.concat([y, X_conv], axis=1)

    return y


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Convolve predictors from one data table into response timesteps defined in another using canonical HRF')
    argparser.add_argument('preds', nargs='+', help='Path(s) to predictor data table(s)')
    argparser.add_argument('response', type=str, help='Path to response data table')
    argparser.add_argument('-c', '--columns', nargs='+', default=[], help='Columns from predictor table to convolve.')
    argparser.add_argument('-k', '--keys', nargs='+', default=['subject', 'docid', 'fROI'], help='Column names to use as keys to define time series.')
    args, unknown = argparser.parse_known_args()

    response = pd.read_csv(args.response,sep=' ',skipinitialspace=True)
    response.sort_values(args.keys + ['time'], inplace=True)

    if len(args.columns) > 0:
        cols_to_process = args.columns[:]
    else:
        cols_to_process = None

    for preds in args.preds:
        preds = pd.read_csv(preds,sep=' ',skipinitialspace=True)
        if cols_to_process is not None:
            preds = preds[args.keys + ['time'] + [c for c in cols_to_process if c in preds.columns]]
        preds.sort_values(args.keys + ['time'], inplace=True)
        if 'rate' in args.columns and 'rate' not in preds.columns:
            preds['rate'] = 1

        if cols_to_process > 0:
            columns = [c for c in cols_to_process if c in preds.columns]
        else:
            columns = [c for c in preds.columns if not c == 'time' and not c in args.keys]

        cols_to_process_new = []
        for c in cols_to_process:
            if not c in preds.columns:
                cols_to_process_new.append(c)
        cols_to_process = cols_to_process_new

        first_obs, last_obs = compute_history_intervals(preds, response, args.keys)
        response = convolve(preds, response, first_obs, last_obs, columns)
    response.to_csv(sys.stdout, ' ', index=False, na_rep='nan')

