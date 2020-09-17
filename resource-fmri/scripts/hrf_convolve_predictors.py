import sys, argparse, pandas as pd, numpy as np
from mvpa2.misc.data_generators import double_gamma_hrf as hrf

argparser = argparse.ArgumentParser(description='Convolve data table using HRF')
argparser.add_argument('data', type=str, help='Path to data table')
argparser.add_argument('-s', '--step', type=float, default=2.0, help='Step size (in seconds) between fMRI samples')
argparser.add_argument('-d', '--doc_names', nargs='+', default=['Boar', 'Aqua', 'MatchstickSeller', 'KingOfBirds', 'Elvis', 'MrSticky', 'HighSchool', 'Roswell', 'Tulips', 'Tourettes'], help='List of document names in input data')
args, unknown = argparser.parse_known_args()

step = float(args.step)

convolve = np.vectorize(lambda x: hrf(time_cur + step - x))

doc_names = args.doc_names

def get_docid(timeseries):
    timeseries = np.array(timeseries)
    docid = [] 
    docix = 0
    for i in range(len(timeseries)):
        if i > 0 and timeseries[i] < timeseries[i-1]:
            docix += 1
        docid.append(doc_names[docix])
    return pd.Series(docid).astype('category')

def convolve_column(df, col):
    categorical = df[col].dtype.name == 'category'
    arr = np.array(df[col])
    time = np.array(df.time)
    time_cur = 0.
    start = 0
    out = [] 
    for i in range(len(df)):
        if i > 0 and time[i] < time[i-1]:
            start = i
            time_cur = 0.
        if time[i] > time_cur:
            time_cur += step
            if categorical:
                if i > start:
                    out.append(arr[i-1])
                else:
                    ## Take the initial category value from the first row in the series
                    out.append(arr[i])
            else:
                convolve = np.vectorize(lambda x: hrf(time_cur - x))
                if i > start:
                    t_conv = convolve(time[start:i])
                    p_conv = (t_conv*np.nan_to_num(arr[start:i])).sum()
                else:
                    p_conv = 0
                out.append(p_conv)
    if categorical:
        out = pd.Series(out, dtype='category')
    else:
        out = pd.Series(np.array(out), dtype='float')
    return out
            
         
	
    

def main():
    df = pd.read_csv(args.data,sep=' ',skipinitialspace=True)
    df['docid'] = get_docid(df.time)
    df['rate'] = 1.
   
    if 'sentid' in df.columns:
        df.sentid = df.sentid.astype('category')
    if 'rolled' in df.columns:
        df.rolled = df.rolled.astype('category')
    sys.stderr.write(df.sentid.dtype.name + '\n')

    n = len(df.columns)
    i = 1

    out = pd.DataFrame()

    for c in df.columns: 
        sys.stderr.write('\rConvolving column %d/%d' %(i,n))
        if df[c].dtype.name != 'category':
            if df[c].dtype.name == 'object':
                df[c] = df[c].astype('category')
            else:
                df[c] = df[c].astype('float')
        out[c] = convolve_column(df, c)
        i += 1

    sys.stderr.write('\n')

    out['sampleid'] = 1
    out.sampleid = out.groupby(['docid']).sampleid.cumsum()
    sampleid_format = '{0:05d}'
    out.sampleid = out.docid.astype('str').str.cat(out.sampleid.apply(sampleid_format.format), sep='-')
    out.to_csv(sys.stdout, ' ', index=False, na_rep='NaN')

main()   
