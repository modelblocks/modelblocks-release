import sys
import configparser

config = configparser.ConfigParser()
config.optionxform = str
config.readfp(sys.stdin)

split_ids = config['data']['split_ids']
print(' '.join(split_ids.strip().split()))
