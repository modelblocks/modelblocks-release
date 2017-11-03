import sys
import re
import numpy as np
import pandas as pd
import librosa
from matplotlib import pyplot as plt

DEBUG = False
name = re.compile('^.*/?([^ /])+\.wav *$')
ix2name = ['Boar', 'Aqua', 'MatchstickSeller', 'KingOfBirds', 'Elvis', 'MrSticky', 'HighSchool', 'Roswell', 'Tulips', 'Tourettes']

df = []

for path in sys.argv[1:]:
    n = ix2name[int(name.match(path).group(1))-1]
    y, sr = librosa.load(path)
    support = np.arange(0, len(y), 50)
    rmse = librosa.feature.rmse(y, hop_length=50)
    rmse = rmse[0]
    if DEBUG:
        wav, = plt.plot(y[:500000])
        power, = plt.plot(np.arange(0, 500000, 50), rmse[0,:10000])
        plt.legend([wav, power], ['WAV form', 'Sound power'])
        plt.show(block=False)
        raw_input()
        plt.clf()
    samples = np.arange(0, len(rmse), 882) # At a sampling rate of 22050 Hz, there are 441 rmse measurements within a 2 second sampling window
    df_new = pd.DataFrame({'soundPower': rmse[samples]})
    df_new['docid'] = n
    sample_number = np.arange(len(df_new)) + 1
    df_new['sampleid'] = sample_number
    df_new.sampleid = df_new.sampleid.astype('str').str.zfill(5)
    df_new.sampleid = df_new.docid.str.cat(df_new.sampleid, sep='-')
    df.append(df_new)

df = pd.concat(df, axis=0)
df.to_csv(sys.stdout, sep=' ', na_rep='nan', index=False)
    
