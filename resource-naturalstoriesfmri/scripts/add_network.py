import sys
import re
import pandas as pd
import argparse


network_names = ['Lang', 'MDlangloc', 'MDspatwm', 'DMNlangloc', 'DMNspatwm', 'AC', 'LHip', 'RHip']


def compute_network(name):
    if 'Hip' in name:
        return 'Hip'
    for network_name in network_names:
        if name.startswith(network_name):
            return network_name
    raise ValueError('Could not determine network name for unrecognized fROI "%s".' %name)


def compute_hemisphere(name):
    if name in ['LHip', 'RHip']:
        return name[0]
    for network_name in network_names:
        if name.startswith(network_name):
            return name[len(network_name)]
    raise ValueError('Could not determine network name for unrecognized fROI "%s".' %name)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        Add field "network" to data table, computed from column named "fROI".
    ''')
    argparser.add_argument('input', default='-', help='Path to input data frame (or "-" for pipe from stdin)')
    argparser.add_argument('-l' '--language_indicator', action='store_true', help='Encode network variable as indicator for language network, rather than network string ID.')
    args = argparser.parse_args()

    if args.input.strip() == '-':
        df_in = sys.stdin
    else:
        df_in = args.input

    df = pd.read_csv(df_in, sep=' ')
    df['network'] = df.fROI.apply(compute_network)
    df['network'] = (df.network == 'Lang').astype('int')
    df['hemisphere'] = df.fROI.apply(compute_hemisphere)
    df.to_csv(sys.stdout, index=False, sep=' ', na_rep='nan')

