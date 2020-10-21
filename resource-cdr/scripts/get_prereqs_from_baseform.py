import sys
import os
if sys.version_info[0] == 2:
    import ConfigParser as configparser
else:
    import configparser

config = configparser.ConfigParser()
config.optionxform = str
config.readfp(sys.stdin)

prereqs = []

if 'data' in config and 'X_train' in config['data']:
    for x in config['data']['X_train'].split(';'):
        if x not in prereqs:
            prereqs.append(x)
if 'data' in config and 'X_dev' in config['data']:
    for x in config['data']['X_dev'].split(';'):
        if x not in prereqs:
            prereqs.append(x)
if 'data' in config and 'X_test' in config['data']:
    for x in config['data']['X_test'].split(';'):
        if x not in prereqs:
            prereqs.append(x)

sys.stdout.write(' '.join(prereqs))

