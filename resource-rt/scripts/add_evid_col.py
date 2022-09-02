import sys, argparse, pandas as pd


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Add evid (event ID) column to data table')
    #argparser.add_argument('data', type=str, help='Path to data table')
    argparser.add_argument('-i', '--id_columns', nargs='+', default=['docid', 'subject', 'sentid', 'resid'], help='Name(s) of column(s) that uniquely identify datapoints in table')
    args = argparser.parse_args()

    #df = pd.read_csv(args.data,sep=' ',skipinitialspace=True)
    df = pd.read_csv(sys.stdin,sep=' ',skipinitialspace=True)

    if 'sentid' in df.columns:
        df.sentid = df.sentid.astype('category')
    if 'docid' in df.columns:
        df.docid = df.docid.astype('category')
    else:
        df['docid'] = '1'
    if 'rolled' in df.columns:
        df.rolled = df.rolled.astype('category')
    if 'tr' in df.columns:
        df.tr = df.tr.astype('int')

    df['evid'] = ''
    #df.key = df.groupby(args.grouping_columns).key.cumsum().apply('{0:05d}'.format)
    for col in args.id_columns[::-1]:
        #df.key = df[col].astype('str').str.cat(df.key, sep='-')
        if col in df:
            if (df.evid == '').all():
                df.evid = col + "-" + df[col].astype("str")
            else:
                df.evid = df.evid + "_" + col + "-" + df[col].astype("str")
        #df.key = df[col].astype('str').str.cat(df.key, sep='-')
        
    df.to_csv(sys.stdout, ' ', index=False, na_rep='NaN')

