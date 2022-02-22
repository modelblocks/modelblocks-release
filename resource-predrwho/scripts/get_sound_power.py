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

DEBUG = False
if DEBUG:
    from matplotlib import pyplot as plt

y, sr = librosa.load(args.audiofile)
# Newer versions of librosa (e.g., 0.7.1) use "rms" rather than "rmse"
if hasattr(librosa.feature, 'rms'):
    rmse = librosa.feature.rms(y, hop_length=50)
else:
    rmse = librosa.feature.rmse(y, hop_length=50)
power = rmse[0]

if DEBUG:
    wav, = plt.plot(y[:500000])
    power, = plt.plot(np.arange(0, 500000, 50), rmse[0,:10000])
    # plt.legend([wav, power], ['WAV form', 'Sound power'])
    # plt.show(block=False)
    # raw_input()
    # plt.clf()

sys.stderr.write('\n')

if args.interval:
    # At a sampling rate of 44100 Hz, there are 882 rmse measurements
    #  within a 1 second sampling window given the hop length of 50
    ix = (np.arange(len(power)) / (882. * args.interval)).astype(int) 
    splits = np.where(ix[1:] != ix[:-1])[0] + 1
    time = splits.astype(float) / 882.
    chunks = np.array([x.mean() for x in np.split(power, splits)][1:])
    df = pd.DataFrame({'time': time, 'soundPower%sms' % int(args.interval * 1000): chunks})
    
df.to_csv(sys.stdout, sep=' ', na_rep='NaN', index=False)
    
