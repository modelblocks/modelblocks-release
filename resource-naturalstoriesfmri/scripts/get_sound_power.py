import sys
import re
import numpy as np
import pandas as pd
import librosa

DEBUG = False
if DEBUG:
    from matplotlib import pyplot as plt

name = re.compile('^.*/?([^ /])+\.wav *$')
ix2name = ['Boar', 'Aqua', 'MatchstickSeller', 'KingOfBirds', 'Elvis', 'MrSticky', 'HighSchool', 'Roswell', 'Tulips', 'Tourettes']


df = pd.read_csv(sys.stdin, sep=' ', skipinitialspace=True)
power_ix = (df.time * 441).astype('int') # At a sampling rate of 22050 Hz, there are 441 rmse measurements within a 1 second sampling window
power_ix = np.array(power_ix)
docid = np.array(df.docid)

max_times = {}
for n in ix2name:
    max_times[n] = df[df.docid == n].time.max()

power = {}

for i in range(len(sys.argv[1:])):
    path = sys.argv[1+i]
    n = ix2name[int(name.match(path).group(1)) - 1]
    sys.stderr.write('\rProcessing audio file "%s" (%d/%d)        ' %(n, i + 1, 10))
    y, sr = librosa.load(path)
    support = np.arange(0, len(y), 50)
    rmse = librosa.feature.rmse(y, hop_length=50)
    rmse = rmse[0]
    power[n] = rmse
    if DEBUG:
        wav, = plt.plot(y[:500000])
        power, = plt.plot(np.arange(0, 500000, 50), rmse[0,:10000])
        plt.legend([wav, power], ['WAV form', 'Sound power'])
        plt.show(block=False)
        raw_input()
        plt.clf()


sys.stderr.write('\n')

soundPower = np.zeros(power_ix.shape)

for i in range(power_ix.shape[0]):
    if docid[i] in power:
        if power_ix[i] < len(power[docid[i]]):
            soundPower[i] = power[docid[i]][power_ix[i]]
        # else:
        #     sys.stderr.write('Event time exceeds length of audio file: %s vs. %s\n' %(float(power_ix[i]) / 441, float(len(power[docid[i]]) / 441 )))



df['soundPower'] = soundPower

df.to_csv(sys.stdout, sep=' ', na_rep='nan', index=False)
    
