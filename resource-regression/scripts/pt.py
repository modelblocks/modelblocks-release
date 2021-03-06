import sys
import re
import numpy as np
import pandas as pd
import math
import argparse


def zscore(a):
    return (a - a.mean()) / a.std()

def nested(p1, p2, df1=True):
    preds_1 = set([x for x in p1.split('.')[-3].split('_')[2:-1] if not x.startswith('~')])
    preds_2 = set([x for x in p2.split('.')[-3].split('_')[2:-1] if not x.startswith('~')])

    if len(preds_1) < len(preds_2):
        a = preds_1
        a_path = p1
        b = preds_2
        b_path = p2
    else:
        a = preds_2
        a_path = p2
        b = preds_2
        b_path = p1

    if df1:
        is_nested = len(a - b) == 0 and len(b - a) == 1
    else:
        is_nested = len(a - b) == 0 and len(b - a) > 0

    variable = b - a

    return a_path, b_path, variable, is_nested


def permutation_test(err_1, err_2, n_iter=10000, n_tails=2, mode='loss', nested=False, verbose=False):
    """
    Perform a paired permutation test for significance.

    :param err_1: ``numpy`` vector; first error/loss vector.
    :param err_2: ``numpy`` vector; second error/loss vector.
    :param n_iter: ``int``; number of resampling iterations.
    :param n_tails: ``int``; number of tails.
    :param mode: ``str``; one of ``["loss", "loglik"]``, the type of error used (losses are averaged while loglik's are summed).
    :param nested: ``bool``; assume that the second model is nested within the first.
    :param verbose: ``bool``; report progress logs to standard error.
    :return:
    """

    err_table = np.stack([err_1, err_2], 1)
    if mode == 'loss':
        base_diff = err_table[:,0].mean() - err_table[:,1].mean()
        if nested and base_diff <= 0:
            return (1.0, base_diff, np.zeros((n_iter,)))
    elif mode == 'loglik':
        base_diff = err_table[:,0].sum() - err_table[:,1].sum()
        if nested and base_diff >= 0:
            return (1.0, base_diff, np.zeros((n_iter,)))
    elif mode == 'corr':
        denom = len(err_table) - 1
        base_diff = (err_table[:,0].sum() - err_table[:,1].sum()) / denom
        if nested and base_diff >= 0:
            return (1.0, base_diff, np.zeros((n_iter,)))
    else:
        raise ValueError('Unrecognized aggregation function "%s" in permutation test' %mode)

    if base_diff == 0:
        return (1.0, base_diff, np.zeros((n_iter,)))

    hits = 0
    if verbose:
        sys.stderr.write('Difference in test statistic: %s\n' %base_diff)
        sys.stderr.write('Permutation testing...\n')

    diffs = np.zeros((n_iter,))

    for i in range(n_iter):
        sys.stderr.write('\r%d/%d' %(i+1, n_iter))
        sys.stderr.flush()
        shuffle = (np.random.random((len(err_table))) > 0.5).astype('int')
        m1 = err_table[np.arange(len(err_table)),shuffle]
        m2 = err_table[np.arange(len(err_table)),1-shuffle]
        if mode == 'loss':
            cur_diff = m1.mean() - m2.mean()
        elif mode == 'loglik':
            cur_diff = m1.sum() - m2.sum()
        elif mode == 'corr':
            cur_diff = (m1.sum() - m2.sum()) / denom
        else:
            raise ValueError('Unrecognized aggregation function "%s" in permutation test' %mode)
        diffs[i] = cur_diff
        if n_tails == 1:
            if base_diff < 0 and cur_diff <= base_diff:
                hits += 1
            elif base_diff > 0 and cur_diff >= base_diff:
                hits += 1
        elif n_tails == 2:
            if math.fabs(cur_diff) > math.fabs(base_diff):
                if verbose:
                    sys.stderr.write('Hit on iteration %d: %s\n' %(i, cur_diff))
                hits += 1
        else:
            raise ValueError('Invalid bootstrap parameter n_tails: %s. Must be in {1, 2}.' %n_tails)

    p = float(hits+1)/(n_iter+1)

    sys.stderr.write('\n')

    return p, base_diff, diffs


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Perform paired permutation tests on one or more pairs of statistics generated by nested models.
    ''')
    argparser.add_argument('paths', nargs='+', help='Paths to input data. If mode == corr, N paths to predictions followed by N paths to observations. Otherwise, N paths to error/likelihood vectors.')
    argparser.add_argument('-n', '--n_iter', type=int, default=10000, help='Number of bootstrap iterations.')
    argparser.add_argument('-t', '--n_tails', type=int, default=2, help='Number of tails in the test (1 or 2).')
    argparser.add_argument('-m', '--mode', type=str, default='loss', help='Type of test statistic used. One of ["loss", "loglik"].')
    args = argparser.parse_args()

    paths = args.paths

    if args.mode == 'corr':
        assert len(paths) % 2 == 0, 'If mode == corr, must provide an even number of paths'
        obs_paths = paths[len(paths) // 2:]
        paths = paths[:len(paths) // 2]

    summary = ''

    for i in range(len(paths)):
        for j in range(i+1, len(paths)):
            a, b, pred, is_nested = nested(paths[i], paths[j])
            if is_nested:
                a_data = pd.read_csv(a, header=None).values
                b_data = pd.read_csv(b, header=None).values

                if args.mode == 'corr':
                    y = pd.read_csv(obs_paths[j], header=None).values
                    y = zscore(y)
                    denom = len(y) - 1
                    
                    a_data = zscore(a_data)
                    b_data = zscore(b_data)
                    
                    r_a_b = (a_data * b_data).sum() / denom
                    
                    a_data = a_data * y
                    b_data = b_data * y
                    
                    r_a_y = a_data.sum() / denom
                    r_b_y = b_data.sum() / denom

                p_value, base_diff, diffs = permutation_test(a_data, b_data, n_iter=args.n_iter, n_tails=args.n_tails, mode=args.mode, nested=True)

                select = np.logical_and(np.isfinite(np.array(a_data)), np.isfinite(np.array(b_data)))
                diff = float(len(a) - select.sum())
                summary += '='*50 + '\n'
                summary += 'Model comparison for predictor %s\n' % ', '.join(pred)
                summary += 'Baseline file: %s\n' %a
                summary += 'Test file: %s\n' %b
                summary += 'Metric: %s\n' % args.mode
                if args.mode == 'corr':
                    summary += 'r(a, y): %s\n' % r_a_y
                    summary += 'r(b, y): %s\n' % r_b_y
                    summary += 'r(a, b): %s\n' % r_a_b
                summary += 'Difference: %s\n' % base_diff
                summary += 'p: %.4e%s\n' % (p_value, '' if p_value > 0.05 else '*' if p_value > 0.01 else '**' if p_value > 0.001 else '***')
                summary += '='*50 + '\n\n'

    sys.stdout.write(summary)



