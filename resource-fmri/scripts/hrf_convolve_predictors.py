import sys, argparse, pandas as pd, numpy as np


# Source: https://github.com/PyMVPA/PyMVPA/blob/master/mvpa2/misc/fx.py#L17
def single_gamma_hrf(t, A=5.4, W=5.2, K=1.0):
    """Hemodynamic response function model.
    The version consists of a single gamma function (also see
    double_gamma_hrf()).
    Parameters
    ----------
    t : float
      Time.
    A : float
      Time to peak.
    W : float
      Full-width at half-maximum.
    K : float
      Scaling factor.
    """
    A = float(A)
    W = float(W)
    K = float(K)
    return \
        K * (t / A) ** ((A ** 2) / (W ** 2) * 8.0 * np.log(2.0)) \
        * np.e ** ((t - A) / -((W ** 2) / A / 8.0 / np.log(2.0)))


# Source: https://github.com/PyMVPA/PyMVPA/blob/master/mvpa2/misc/fx.py#L43
def double_gamma_hrf(t, A1=5.4, W1=5.2, K1=1.0, A2=10.8, W2=7.35, K2=0.35):
    """Hemodynamic response function model.
    The version is using two gamma functions (also see single_gamma_hrf()).
    Parameters
    ----------
    t : float
      Time.
    A : float
      Time to peak.
    W : float
      Full-width at half-maximum.
    K : float
      Scaling factor.
    Parameters A, W and K exists individually for each of both gamma
    functions.
    """
    A1 = float(A1)
    W1 = float(W1)
    K1 = float(K1)
    A2 = float(A2)
    W2 = float(W2)
    K2 = float(K2)
    return single_gamma_hrf(t, A1, W1, K1) - single_gamma_hrf(t, A2, W2, K2)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Convolve data table using HRF')
    argparser.add_argument('data', type=str, help='Path to data table')
    argparser.add_argument('-s', '--step', type=float, default=2.0, help='Step size (in seconds) between fMRI samples')
    argparser.add_argument('-S', '--start', type=float, default=0.0, help='Start time (time of first scan)')
    argparser.add_argument('-g', '--grouping_columns', nargs='+', default=['docid'], help='Name(s) of column(s) that define(s) unique time series')
    argparser.add_argument('-t', '--extra_time', type=int, default=30, help='Number of seconds past the last word to calculate convolved predictors')
    args = argparser.parse_args()

    df = pd.read_csv(args.data,sep=' ',skipinitialspace=True)
    df['rate'] = 1.
   
    if 'sentid' in df.columns:
        df.sentid = df.sentid.astype('category')
    if 'docid' in df.columns:
        df.docid = df.docid.astype('category')
    else:
        df['docid'] = '1'
    if 'rolled' in df.columns:
        df.rolled = df.rolled.astype('category')

    cols = [x for x in df.select_dtypes([np.number]).columns if x != 'time']

    gb = df.groupby(args.grouping_columns)
    series = [x[1] for x in gb]
    series_names = [x[0] for x in gb]

    out = []
    for i, df_cur in enumerate(series):
        if 'duration' in df_cur.columns:
            duration = df_cur['duration']
        else:
            duration = None
        X = df_cur[cols]
        impulse_times = df_cur.time.values
        max_response_time = int(np.ceil(df_cur.time.max())) + args.extra_time
        if max_response_time % 2 != 0:
           max_response_time += 1
        tr = np.arange(1, 1 + (max_response_time // args.step))
        response_times = (tr-1) * args.step + args.start
        D = response_times[..., None] - impulse_times[None, ...]
        G_mask = D >= 0
        G = double_gamma_hrf(D)
        G = np.where(G_mask, G, 0)
        if duration is not None:
            X = X.multiply(duration, axis=0)
        X_conv = np.dot(G, X)
        X_conv = pd.DataFrame(X_conv, columns=cols)
        X_conv['time'] = response_times
        X_conv['tr'] = tr
        for col in args.grouping_columns:
            X_conv[col] = series_names[i]
        out.append(X_conv)

    out = pd.concat(out, axis=0)
    out.reset_index(drop=True, inplace=True)
    out['sampleid'] = 1
    out.sampleid = out.groupby(args.grouping_columns).sampleid.cumsum().apply('{0:05d}'.format)
    for col in args.grouping_columns[::-1]:
        out.sampleid = out[col].astype('str').str.cat(out.sampleid, sep='-')
    out.to_csv(sys.stdout, ' ', index=False, na_rep='NaN')

