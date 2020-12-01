import sys
import configparser

sys.path.append(sys.argv[1])
from dtsr.formula import Formula


config = configparser.ConfigParser()
config.optionxform = str
config.readfp(sys.stdin)

cols = ['word', 'time', 'subject']
cols += config['data']['series_ids'].strip().split()
if 'split_ids' in config['data']:
    cols += config['data']['split_ids'].strip().split()
for m in [x for x in config if x.startswith('model_DTSR')]:
    formula = Formula(config[m]['formula'])
    cols.append(formula.dv_term.id)
    for impulse in formula.t.impulses():
        cols.append(impulse.id)

cols = sorted(list(set(cols)))
print(' '.join(cols))


