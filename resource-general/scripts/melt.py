import sys
import re
import pandas as pd
import argparse

def filter_names(names, filters):
    """
    Return elements of **names** permitted by **filters**, preserving order in which filters were matched.
    Filters can be ordinary strings, regular expression objects, or string representations of regular expressions.
    For a regex filter to be considered a match, the expression must entirely match the name.

    :param names: ``list`` of ``str``; pool of names to filter.
    :param filters: ``list`` of ``{str, SRE_Pattern}``; filters to apply in order
    :return: ``list`` of ``str``; names in **names** that pass at least one filter
    """

    filters_regex = [re.compile(f if f.endswith('$') else f + '$') for f in filters]

    out = []

    for i in range(len(filters)):
        filter = filters[i]
        filter_regex = filters_regex[i]
        for name in names:
            if name not in out:
                if name == filter:
                    out.append(name)
                elif filter_regex.match(name):
                    out.append(name)

    return out

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        Melts input table with multiple response columns into output table with single response column along with identifier column.
    ''')
    argparser.add_argument('input', default='-', help='Path to input data frame (or "-" for pipe from stdin)')
    argparser.add_argument('to_melt', nargs='+', help='Column names to melt (also supports regular expressions)')
    argparser.add_argument('-k', '--to_keep', nargs='*', default=[], help='Source column names to keep (also supports regular expressions, mutually exclusive with "-d")')
    argparser.add_argument('-d', '--to_drop', nargs='*', default=[], help='Source column names to drop (also supports regular expressions, mutually exclusive with "-k")')
    argparser.add_argument('-v', '--var_name', type=str, default='fROI', help='Name for new identifier column')
    argparser.add_argument('-V', '--value_name', type=str, default='BOLD', help='Name for new value (response) column')
    argparser.add_argument('-p', '--prefix_len', type=int, default=0, help='Length of prefix to delete from melted column names')
    argparser.add_argument('-s', '--suffix_len', type=int, default=0, help='Length of suffix to delete from melted column names')
    args = argparser.parse_args()

    if args.input.strip() == '-':
        df_in = sys.stdin
    else:
        df_in = args.input

    df = pd.read_csv(df_in, sep=' ')

    value_vars = filter_names(df.columns, args.to_melt)
    value_var_set = set(value_vars)
    id_vars = [x for x in df.columns if not x in value_vars]

    df = pd.melt(df, id_vars=id_vars, value_vars=value_vars, var_name=args.var_name, value_name=args.value_name)
    df = df[df[args.value_name].notnull()]

    if args.prefix_len > 0:
        df[args.var_name] = df[args.var_name].apply(lambda x: x[args.prefix_len:])
    if args.suffix_len > 0:
        df[args.var_name] = df[args.var_name].apply(lambda x: x[:-args.suffix_len])

    if len(args.to_keep) > 0:
        to_keep = filter_names(df[args.var_name].unique(), args.to_keep)
        df = df[df[args.var_name].isin(to_keep)]
    elif len(args.to_drop) > 0:
        to_drop = filter_names(df[args.var_name].unique(), args.to_drop)
        df = df[~df[args.var_name].isin(to_drop)]

    df.to_csv(sys.stdout, index=False, sep=' ', na_rep='nan')





