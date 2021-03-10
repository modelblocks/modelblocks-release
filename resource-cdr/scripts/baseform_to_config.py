import sys
import os
import configparser

from cdr.formula import Formula

X_path = sys.argv[1]
y_path = sys.argv[2]
outdir = sys.argv[3]
main_effects = sys.argv[4:]

config = configparser.ConfigParser()
config.optionxform = str
config.readfp(sys.stdin)

if 'data' in config and 'X_train' in config['data']:
    X_train = ';'.join([X_path, config['data']['X_train']])
else:
    X_train = X_path
if 'data' in config and 'X_dev' in config['data']:
    X_dev = ';'.join([X_path, config['data']['X_dev']])
else:
    X_dev = X_path
if 'data' in config and 'X_test' in config['data']:
    X_test = ';'.join([X_path, config['data']['X_test']])
else:
    X_test = X_path

config['data']['X_train'] = X_train
config['data']['X_dev'] = X_dev
config['data']['X_test'] = X_test
config['data']['y_train'] = y_path
config['data']['y_dev'] = y_path #.replace('fit_part','expl_part')
config['data']['y_test'] = y_path #.replace('fit_part', 'held_part')

impulse_to_irf = {}
if 'impulse_to_irf' in config:
    for key in config['impulse_to_irf']:
        impulse_to_irf[key] = config['impulse_to_irf'][key]

if 'default' in impulse_to_irf:
    default_irf_str = impulse_to_irf['default']
else:
    default_irf_str = 'ShiftedGammaShapeGT1(alpha=2, beta=5, delta=-0.2, ran=T)'

impulse_to_transform = {}
if 'impulse_to_transform' in config:
    for key in config['impulse_to_transform']:
        impulse_to_transform[key] = config['impulse_to_transform'][key]

newpred_rangf = ['subject']
if 'newpred_rangf' in config:
    newpred_rangf = config['newpred_rangf']['rangf'].strip().split()

baseline_found = False
for name in config:
    if (name.startswith('model_CDR') or name.startswith('model_DTSR')) and name.endswith('_BASELINE'):
        baseline_found = True
        new_name = name[:-9]
        model_template = config[name]
        f = Formula(model_template['formula'])

        for effect in main_effects:
            if effect.startswith('~'):
                ablate = True
                effect = effect[1:]
            else:
                ablate = False

            irf_str = impulse_to_irf.get(effect, default_irf_str)
            effect_name = effect
            transform = impulse_to_transform.get(effect, ['s'])
            if transform == 'None':
                transform = []
            elif isinstance(transform, str):
                transform = transform.strip().split()
            for t in reversed(transform):
                effect_name = t + '(' + effect_name + ')'
           
            f.insert_impulses(effect_name, irf_str, rangf=newpred_rangf)
            if ablate:
                f.ablate_impulses(effect)

        config[new_name] = dict(config[name]).copy()
        config[new_name]['formula'] = str(f)
        del config[name]

if not baseline_found and len(main_effects) > 0:
    sys.stderr.write('No CDR models in config file are flagged with suffix "_BASELINE". Effects %s will not be added/ablated in any models.\n' %main_effects)

config['global_settings'] = {'outdir': outdir}

if 'impulse_to_irf' in config:
    del config['impulse_to_irf']
if 'impulse_to_transform' in config:
    del config['impulse_to_transform']
if 'newpred_rangf' in config:
    del config['newpred_rangf']

if not os.path.exists(outdir):
    os.makedirs(outdir)

with open(outdir + '/config.ini', 'w') as f:
    config.write(f)
