import sys, argparse, pandas as pd, numpy as np
from mvpa2.testing.datasets import double_gamma_hrf as hrf

argparser = argparse.ArgumentParser(description='Convolve data table using HRF')
argparser.add_argument('data', type=str, help='Path to data table')
argparser.add_argument('predictors', type=str, nargs='+', default=[], help='List of predictors to convolve')
argparser.add_argument('--step', type=float, default=2.0, help='Step size (in seconds) between fMRI samples')
argparser.add_argument('--start', type=float, default=0.0, help='Time (in seconds) of first fMRI sample')
argparser.add_argument('--debug', action='store_true', help='Print verbose output')
args, unknown = argparser.parse_known_args()

def main():
    predictors = args.predictors
    assert len(predictors) > 0, 'ERROR: no predictors to convolve'
    data = pd.read_csv(args.data,sep=' ',skipinitialspace=True)
    step = args.step
    start = args.start
    end = data['timestamp'].iat[-1]
    n = int((end-start) / step) + 1
    assert n > 0, 'Time interval contains no fMRI samples'
    out = data[predictors].head(n)
    time = lambda x: start + (step*i)

    t_col = data[['timestamp']]

    for p in predictors:
        for i in xrange(n):
            t = time(i)
            tmp = data.loc[data['timestamp'] < t]
            n_prec = tmp.shape[0]
            
            if n_prec > 0:
                t_conv = tmp['timestamp'].apply(lambda x: hrf(t - x))
                p_conv = (t_conv*data.head(n_prec)[p].astype(float)).sum()
            else:
                p_conv = 0
            out.at[i,p] = p_conv
    
    out['sampleid'] = range(1, n+1)
    out.to_csv(sys.stdout, ' ', index=False, na_rep='nan')





    
      
main()   
