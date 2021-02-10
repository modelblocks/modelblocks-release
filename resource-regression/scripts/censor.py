import sys
import argparse
import numpy as np
import pandas as pd
import configparser


def compute_filter(y, field, cond):
    """
    Compute filter given a field and condition

    :param y: ``pandas`` ``DataFrame``; response data.
    :param field: ``str``; name of column on whose values to filter.
    :param cond: ``str``; string representation of condition to use for filtering.
    :return: ``numpy`` vector; boolean mask to use for ``pandas`` subsetting operations.
    """

    cond = cond.strip()
    if cond.startswith('<='):
        return y[field] <= (np.inf if cond[2:].strip() == 'inf' else float(cond[2:].strip()))
    if cond.startswith('>='):
        return y[field] >= (np.inf if cond[2:].strip() == 'inf' else float(cond[2:].strip()))
    if cond.startswith('<'):
        return y[field] < (np.inf if cond[1:].strip() == 'inf' else float(cond[1:].strip()))
    if cond.startswith('>'):
        return y[field] > (np.inf if cond[1:].strip() == 'inf' else float(cond[1:].strip()))
    if cond.startswith('=='):
        try:
            return y[field] == (np.inf if cond[2:].strip() == 'inf' else float(cond[2:].strip()))
        except:
            return y[field].astype('str') == cond[2:].strip()
    if cond.startswith('!='):
        try:
            return y[field] != (np.inf if cond[2:].strip() == 'inf' else float(cond[2:].strip()))
        except:
            return y[field].astype('str') != cond[2:].strip()
    if cond.startswith('in'):
        try:
            return y[field].isin([float(x) for x in cond[2:].strip().split(';')])
        except:
            return y[field].astype('str').isin(cond[2:].strip().split(';'))
    raise ValueError('Unsupported comparator in filter "%s"' %cond)


def compute_filters(y, censorship_params=None):
    """
    Compute filters given a filter map.

    :param y: ``pandas`` ``DataFrame``; response data.
    :param censorship_params: ``dict``; maps column names to filtering criteria for their values.
    :return: ``numpy`` vector; boolean mask to use for ``pandas`` subsetting operations.
    """

    if censorship_params is None:
        return y
    select = np.ones(len(y), dtype=bool)
    for field in censorship_params:
        if field in y.columns:
            for cond in censorship_params[field]:
                select &= compute_filter(y, field, cond)
    return select


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        A utility that reads a dataset from stdin and removes outliers given censorship criteria.
    ''')
    argparser.add_argument('-c', '--config_path', default=None, help='Path to config file with censoring instructions. If not provided, complete data will be returned uncensored.')
    args, unknown = argparser.parse_known_args()

    df = pd.read_csv(sys.stdin, sep=' ', skipinitialspace=True)

    censorship_params = {}

    if args.config_path is not None:
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(args.config_path)

        censor = config['censor']
        censorship_params = {}
        for c in censor:
            censorship_params[c] = [x.strip() for x in censor[c].strip().split(',')]

    select = compute_filters(df, censorship_params)

    df = df[select]

    df.to_csv(sys.stdout, sep=' ', index=False, na_rep='NaN')
