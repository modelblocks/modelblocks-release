import pandas as pd

df = pd.read_csv('~/data/ET_geco/EnglishMaterialInterestAreas.csv', keep_default_na=False)

df2 = df.WORD_ID.str.split('-', expand=True)
df2.columns = ['part', 'trialid', 'trialpos']
df2['sentid'] = df.SENTENCE_ID
df4 = df2.drop_duplicates(['trialid', 'sentid'])

trialid = df4.trialid.astype(int).values
sentid = df4.sentid.values

map = {}
for s, t in zip(sentid, trialid):
    if s not in map:
        map[s] = set()
    map[s].add(t)

del map['#N/A']

for s in map:
    if len(map[s]) > 1:
        map[s] = sorted(list(map[s]))[-1]
    else:
        map[s] = list(map[s])[0]

keys = list(map.keys())

for s in keys:
    part, ix = s.split('-')
    ix = int(ix)
    _s = '%s-%s' % (part, ix + 1)
    if _s not in map:
        map[_s] = map[s]

map['4-216'] = map['4-215']

with open('../resource-geco/srcmodel/geco_sent2trial.txt', 'w') as f:
    for s in map:
        f.write('%s,%s\n' % (s, map[s]))
            
