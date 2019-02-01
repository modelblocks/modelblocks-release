import sys, pandas as pd, matplotlib.pyplot as plt, seaborn as sns


# Load data
df = pd.read_csv(sys.stdin, sep=', ')

# Preprocess
df.rename(columns={'lmeargs': 'link', 'signif': 'p', 'estimate': 'beta', 't value': 't'}, inplace=True)
df['agg'] = ''
df.ix[df.effect.str.endswith('Sum'), 'agg'] = 'Sum'
df.ix[df.effect.str.endswith('Max'), 'agg'] = 'Max'
df.ix[df.effect.str.endswith('Mean'), 'agg'] = 'Mean'

df['effect'] = df.effect.str.replace(r'Sum$|Max$|Mean$','')
df['ROI'] = df.formname.str.replace(r'passages.*', '')
df['formname'] = df.formname.str.replace(r'.*passages', '')

df.link = df.link.astype('str')
df.ix[df.link!='-L', 'link'] = ''
df.ix[df.link=='-L', 'link'] = 'Log'

df['pinv'] = float(1) / df['p']
df['effect_full'] = df['effect'] + df['agg'] + df['link']
df['effect_link'] = df['effect'] + df['link']

link = ['', 'Log']
form = ['Basic', 'Nwrds', 'NwrdsWlen']
agg = ['Max', 'Mean', 'Sum', '']
value = ['p', 'pinv', 'beta']
color = ['Blues_r', 'Blues', 'PuOr']
clip = [(0.0, 0.05), (20, 10000), (-0.2, 0.2)]
effect = sorted(list(df.effect.unique()))
effect_full = sorted(list(df.effect_full.unique()))
roi = sorted(list(df.ROI.unique()))

for f in form:
    for i in range(len(value)):
        fig, ax = plt.subplots()
        df_cur = df[df['formname'] == f].pivot(index='effect_full', columns='ROI', values=value[i])
        if df_cur.shape[0] == 0:
            continue
        fig.set_size_inches(15, 0.25*df_cur.shape[0])
        sns.heatmap(df_cur, cmap=color[i], ax=ax, cbar=True, vmin=clip[i][0], vmax=clip[i][1])
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_ylabel('Effect')
        ax.set_xlabel(value[i] + ' by ROI (' + f + ')')
        plt.yticks(rotation=0)
        plt.savefig('alleffects_' + f + '_' + value[i] + '.jpg')
        plt.close(fig)

for f in form:
    for i in range(len(value)):
        for a in agg:
            fig, ax = plt.subplots()
            fig.patch.set_visible(False)
            df_cur = df[(df['formname'] == f) & (df['agg'] == a)].pivot(index='effect_link', columns='ROI', values=value[i])
            if df_cur.shape[0] == 0:
                continue
            fig.set_size_inches(15, 0.25*df_cur.shape[0])
            sns.heatmap(df_cur, cmap=color[i], ax=ax, cbar=True, vmin=clip[i][0], vmax=clip[i][1])
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')
            ax.set_ylabel('Effect')
            ax.set_xlabel(value[i] + ' by ROI (' + f + ', ' + a + ')')
            plt.yticks(rotation=0)
            plt.savefig(a + '_' + f + '_' + value[i] + '.jpg')
            plt.close(fig)

plt.close()
