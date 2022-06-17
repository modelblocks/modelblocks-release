import sys
import re
import numpy as np
import pandas as pd
import librosa
import argparse

argparser = argparse.ArgumentParser('''
        Utility for computing sound power metrics from pre--Dr. Who audio stimuli
        (Tree, Jeanne, Dinner)
''')
argparser.add_argument('audiofile', help='Paths to audio file for processing')
argparser.add_argument('interval', type=float, help='Interval step at which to extract sound power measures (used for deconvolving sound power using DTSR). Power measures are rescaled by interval value to ensure valid convolution.')
args = argparser.parse_args()


# evmeasures
df  = pd.read_csv(sys.stdin, sep=' ', skipinitialspace=True)

# the evmeasures should only contain one document (e.g., Tree)
assert len(df.docid.unique()) == 1
docid = df.docid.unique()[0]

#ix2docid = df.docid.unique()
#docid2ix = {}
#for i in range(len(ix2docid)):
#    docid2ix[ix2docid[i]] = i

# soundpower will be tiled by key
keys = []
for x in ['subject', 'docid', 'fROI', 'run']:
    if x in df.columns:
        keys.append(x)

y, sr = librosa.load(args.audiofile)
# Newer versions of librosa (e.g., 0.7.1) use "rms" rather than "rmse"
if hasattr(librosa.feature, 'rms'):
    rmse = librosa.feature.rms(y, hop_length=50)
else:
    rmse = librosa.feature.rmse(y, hop_length=50)
power = rmse[0]

sys.stderr.write('\n')

# At a sampling rate of 44100 Hz, there are 882 rmse measurements
#  within a 1 second sampling window given the hop length of 50
ix = (np.arange(len(power)) / (882. * args.interval)).astype(int) 
splits = np.where(ix[1:] != ix[:-1])[0] + 1
time = splits.astype(float) / 882.
chunks = np.array([x.mean() for x in np.split(power, splits)][1:])
intervals = pd.DataFrame({'time': time, 'soundPower%sms' % int(args.interval * 1000): chunks})
intervals['docid'] = docid
    
df = df[keys].drop_duplicates()
df = df.merge(intervals, on='docid', how='left')
df.to_csv(sys.stdout, sep=' ', na_rep='NaN', index=False)
    
