import sys
import argparse
import numpy as np
import pandas as pd
import configparser

def compute_splitID(y, split_fields):
    """
    Map tuples in columns designated by **split_fields** into integer ID to use for data partitioning.

    :param y: ``pandas`` ``DataFrame``; response data.
    :param split_fields: ``list`` of ``str``; column names to use for computing split ID.
    :return: ``numpy`` vector; integer vector of split ID's.
    """

    splitID = np.zeros(len(y), dtype='int32')
    for col in split_fields:
        splitID += y[col].cat.codes
    return splitID

def compute_partition(y, modulus, n):
    """
    Given a ``splitID`` column, use modular arithmetic to partition data into **n** subparts.

    :param y: ``pandas`` ``DataFrame``; response data.
    :param modulus: ``int``; modulus to use for splitting, must be at least as large as **n**.
    :param n: ``int``; number of subparts in the partition.
    :return: ``list`` of ``numpy`` vectors; one boolean vector per subpart of the partition, selecting only those elements of **y** that belong.
    """

    partition = [((y.splitID) % modulus) <= (modulus - n)]
    for i in range(n-1, 0, -1):
        partition.append(((y.splitID) % modulus) == (modulus - i))
    return partition

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        A utility that reads a dataset from stdin and partitions it given splitting criteria.
    ''')
    argparser.add_argument('-c', '--config_path', default=None, help='Path to config file with partitioning instructions. If not provided, default params will be used. Defaults are not necessarily reasonable for all use cases.')
    argparser.add_argument('-p', '--partition', nargs='+', help='One or more space-delimited IDs of subsets of partition to send to stdout. If arity = 2, subset of ("train", "test"). If arity = 3, subset of {"train", "dev", "test"}. Otherwise, set of integers.')
    args, unknown = argparser.parse_known_args()

    if args.config_path is not None:
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(args.config_path)

        params = config['partition']

        mod = params.getint('mod', 4)
        arity = params.getint('arity', 3)
        fields = params.get('fields', 'subject sentid')
        fields = fields.strip().split()

    else:
        mod = 4
        arity = 3
        fields = ['subject', 'sentid']

    df = pd.read_csv(sys.stdin, sep=' ', skipinitialspace=True)
    for f in fields:
        df[f] = df[f].astype('category')
    cols = df.columns
    df['splitID'] = compute_splitID(df, fields)

    select = compute_partition(df, mod, arity)

    if arity == 3:
        names = ['train', 'dev', 'test']
    elif arity == 2:
        names = ['train', 'test']
    else:
        names = [str(x) for x in range(args.n)]

    select_new = None
    for name in args.partition:
        try:
            i = int(name)
        except:
            i = names.index(name)
        if select_new is None:
            select_new = select[i]
        else:
            select_new |= select[i]
    df[select_new].to_csv(sys.stdout, sep=' ', index=False, na_rep='nan', columns=cols)
