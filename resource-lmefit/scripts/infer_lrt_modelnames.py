import sys

err1, err2 = sys.argv[1:]

m1 = '.'.join(err1.split('.')[:-2]) + '.fitmodel.rdata'
fit_part1 = err1.split('.')[-3].split('_')[0]
eval_part1 = err1.split('.')[-2].split('_')[0]
assert fit_part1 == eval_part1, 'Likelihood ratio testing is in-sample and requires matched training and evaluation partitions. Saw "%s", "%s"' % (fit_part1, eval_part1)
m2 = '.'.join(err2.split('.')[:-2]) + '.fitmodel.rdata'
fit_part2 = err2.split('.')[-3].split('_')[0]
eval_part2 = err2.split('.')[-2].split('_')[0]
assert fit_part2 == eval_part2, 'Likelihood ratio testing is in-sample and requires matched training and evaluation partitions. Saw "%s", "%s"' % (fit_part2, eval_part2)

sys.stdout.write('%s %s' % (m1, m2))
