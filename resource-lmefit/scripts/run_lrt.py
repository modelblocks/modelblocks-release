import sys
import os

def nested(A, B, max_df=1):
    # Only checks nesting based on variable names and ablation configuration
    # Does not check for identity of data or formula
    a_vars = A.split('.')[-3].split('_')[2:-1]
    b_vars = B.split('.')[-3].split('_')[2:-1]
    if len(a_vars) != len(b_vars):
        return A, B, False
    a_vars = sorted([(x[1:],0) if x.startswith('~') else (x,1) for x in a_vars], key=lambda x:x[0])
    b_vars = sorted([(x[1:],0) if x.startswith('~') else (x,1) for x in b_vars], key=lambda x:x[0])
    if set([x[0] for x in a_vars]) != set([x[0] for x in b_vars]):
        return A, B, False
    baseline = None
    full = None
    df = 0
    for a, b in zip(a_vars, b_vars):
        if a[1] != b[1]:
            df += 1
            if df > max_df:
                return A, B, False
            if a[1] < b[1]:
                baseline_cur = A
                full_cur = B
            else:
                baseline_cur = B
                full_cur = A
            if baseline is None:
                baseline = baseline_cur
                full = full_cur
            elif baseline != baseline_cur:
                return A, B, False

    return baseline, full, True

paths = sorted(sys.argv[1:], key=len, reverse=True)

models = []

for path in paths:
    m = '.'.join(path.split('.')[:-2]) + '.fitmodel.rdata'
    fit_part = path.split('.')[-3].split('_')[0]
    eval_part = path.split('.')[-2].split('_')[0]
    assert fit_part == eval_part, 'Likelihood ratio testing is in-sample and requires matched training and evaluation partitions. Saw "%s", "%s"' % (fit_part, eval_part)
    models.append(m)


for i, m1 in enumerate(models):
    for m2 in models[i+1:]:
        baseline, full, is_nested = nested(m1, m2)
        if is_nested:
            os.system('../resource-lmefit/scripts/lmefit2lrtsignif.r %s %s' % (m1, m2))

