import sys
import os
import configparser

X_path = sys.argv[1]
y_path = sys.argv[2]
outdir = sys.argv[3]

config = configparser.ConfigParser()
config.optionxform = str
config.readfp(sys.stdin)

config['data']['X_train'] = X_path
config['data']['X_dev'] = X_path
config['data']['X_test'] = X_path
config['data']['y_train'] = y_path + '.train'
config['data']['y_dev'] = y_path + '.dev'
config['data']['y_test'] = y_path + '.test'

config['global_settings'] = {'outdir': outdir}

if not os.path.exists(outdir):
    os.makedirs(outdir)

with open(outdir + '/config.ini', 'w') as f:
    config.write(f)
