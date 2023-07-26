import sys
import io
import numpy as np
import pandas as pd

def get_run(x):
    print(x)
    trial_start = x.TRIAL_START_TIME
    starts = sorted(list(trial_start.unique()))
    run = trial_start.apply(lambda x, starts=starts: starts.index(x))
    out = pandas.DataFrame({'run': run}, index=x.index)

    return out

def get_fdurs(x):
    subjects = x.subject.values
    docids = x.docid.values
    indices = x.trial_ix.values
    spdur = x.fdurSP
    blinkbefore = x.blinkbeforefix
    blinkafter = x.blinkafterfix
    offscreenbefore = x.offscreenbeforefix
    offscreenafter = x.offscreenafterfix
    fdurSPsummed_cur = None
    fdurFP_cur = None
    fdurGP_cur = None
    key_cur = None
    sp_i_cur = None
    fp_i_cur = None
    gp_i_cur = None
    blinkduringSP_cur = 0
    blinkduringFP_cur = 0
    blinkduringGP_cur = 0
    offscreenduringSP_cur = 0
    offscreenduringFP_cur = 0
    offscreenduringGP_cur = 0
    inregression_cur = 0
    fdurSPsummed = np.full_like(spdur, -1)
    fdurFP = np.full_like(spdur, -1)
    fdurGP = np.full_like(spdur, -1)
    blinkduringSP = np.zeros_like(blinkbefore)
    blinkduringFP = np.zeros_like(blinkbefore)
    blinkduringGP = np.zeros_like(blinkbefore)
    offscreenduringSP = np.zeros_like(offscreenbefore)
    offscreenduringFP = np.zeros_like(offscreenbefore)
    offscreenduringGP = np.zeros_like(offscreenbefore)
    inregression = np.zeros_like(offscreenbefore)
    wdelta = np.zeros_like(offscreenbefore)
    prevwasfix = np.zeros_like(offscreenbefore)
    nextwasfix = np.zeros_like(offscreenbefore)
    frontier = 0
    for i, (s, doc, bb, ob, ix, d) in enumerate(zip(subjects, docids, blinkbefore, offscreenbefore, indices, spdur)):
        key = (s, doc)
        if key != key_cur:
            if key_cur is not None:
                fdurSPsummed[sp_i_cur] = fdurSPsummed_cur
                blinkduringSP[sp_i_cur] = blinkduringSP_cur
                offscreenduringSP[sp_i_cur] = offscreenduringSP_cur
                
                if fdurFP_cur is not None:
                    fdurFP[fp_i_cur] = fdurFP_cur
                    blinkduringFP[fp_i_cur] = blinkduringFP_cur
                    offscreenduringFP[fp_i_cur] = offscreenduringFP_cur

                fdurGP[gp_i_cur] = fdurGP_cur
                blinkduringGP[gp_i_cur] = blinkduringGP_cur
                offscreenduringGP[gp_i_cur] = offscreenduringGP_cur

            sp_i_cur = None
            fp_i_cur = None
            gp_i_cur = None
            fdurSPsummed_cur = None
            fdurFP_cur = None
            fdurGP_cur = None
            key_cur = key
            blinkduringSP_cur = 0
            blinkduringFP_cur = 0
            blinkduringSP_cur = 0
            offscreenduringSP_cur = 0
            offscreenduringFP_cur = 0
            offscreenduringGP_cur = 0
            inregression_cur = 0
            frontier = 0

        if sp_i_cur is None:         # Start of sequence
            fdurSPsummed_cur = d
            fdurFP_cur = d
            fdurGP_cur = d
            sp_i_cur = i
            fp_i_cur = i
            gp_i_cur = i
            wdelta_cur = 0
        elif ix == indices[i-1]:  # Same word region
            fdurSPsummed_cur += d
            blinkduringSP_cur = max(blinkduringSP_cur, bb)
            offscreenduringSP_cur = max(offscreenduringSP_cur, ob)
           
            if fdurFP_cur is not None: 
                fdurFP_cur += d
                blinkduringFP_cur = max(blinkduringFP_cur, bb)
                offscreenduringFP_cur = max(offscreenduringFP_cur, ob)
            
            fdurGP_cur += d
            blinkduringGP_cur = max(blinkduringGP_cur, bb)
            offscreenduringGP_cur = max(offscreenduringGP_cur, ob)

            wdelta_cur = ix - indices[i-1]
            wdelta[i] = wdelta_cur
            prevwasfix[i] = int(wdelta_cur == 1)
            nextwasfix[i] = int(wdelta_cur == -1)
        else:                     # New word region
            fdurSPsummed[sp_i_cur] = fdurSPsummed_cur
            blinkduringSP[sp_i_cur] = blinkduringSP_cur
            offscreenduringSP[sp_i_cur] = offscreenduringSP_cur
           
            fdurSPsummed_cur = d
            sp_i_cur = i
            blinkduringSP_cur = bb
            offscreenduringSP_cur = ob

            if ix > frontier:
                if fdurFP_cur is not None:
                    fdurFP[fp_i_cur] = fdurFP_cur
                    blinkduringFP[fp_i_cur] = blinkduringFP_cur
                    offscreenduringFP[fp_i_cur] = offscreenduringFP_cur

                fdurGP[gp_i_cur] = fdurGP_cur
                blinkduringGP[gp_i_cur] = blinkduringGP_cur
                offscreenduringGP[gp_i_cur] = offscreenduringGP_cur

                fdurFP_cur = d
                fdurGP_cur = d
                fp_i_cur = i
                gp_i_cur = i
                blinkduringFP_cur = bb
                blinkduringGP_cur = bb
                offscreenduringFP_cur = ob
                offscreenduringGP_cur = ob
            else:
                if not inregression_cur:        
                    fdurFP[fp_i_cur] = fdurFP_cur
                    blinkduringFP[fp_i_cur] = blinkduringFP_cur
                    offscreenduringFP[fp_i_cur] = offscreenduringFP_cur
                    fdurFP_cur = None
                fdurGP_cur += d

            wdelta_cur = ix - indices[i-1]
            wdelta[i] = wdelta_cur
            prevwasfix[i] = int(wdelta_cur == 1)
            nextwasfix[i] = int(wdelta_cur == -1)

        if ix < frontier:
            inregression_cur = 1
        elif ix > frontier:
            inregression_cur = 0
        frontier = max(frontier, ix)
        inregression[i] = inregression_cur

    fdurSPsummed[sp_i_cur] = fdurSPsummed_cur
    blinkduringSP[sp_i_cur] = blinkduringSP_cur
    offscreenduringSP[sp_i_cur] = offscreenduringSP_cur

    if fdurFP_cur is not None:
        fdurFP[fp_i_cur] = fdurFP_cur
        blinkduringFP[fp_i_cur] = blinkduringFP_cur
        offscreenduringFP[fp_i_cur] = offscreenduringFP_cur

    fdurGP[gp_i_cur] = fdurGP_cur
    blinkduringGP[gp_i_cur] = blinkduringGP_cur
    offscreenduringGP[gp_i_cur] = offscreenduringGP_cur

    return {
        'fdurSPsummed': fdurSPsummed,
        'blinkduringSPsummed': blinkduringSP,
        'offscreenduringSPsummed': offscreenduringSP,
        'fdurFP': fdurFP,
        'blinkduringFP': blinkduringFP,
        'offscreenduringFP': offscreenduringFP,
        'fdurGP': fdurGP,
        'blinkduringGP': blinkduringGP,
        'offscreenduringGP': offscreenduringGP,
        'inregression': inregression,
        'wdelta': wdelta,
        'prevwasfix': prevwasfix,
    }



stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='latin-1')

df = pd.read_csv(stdin)

cols = [
    'RECORDING_SESSION_LABEL',
    'page',
    'TRIAL_START_TIME',
    'PREVIOUS_SAC_END_TIME',
    'CURRENT_FIX_INTEREST_AREA_LABEL',
    'CURRENT_FIX_INTEREST_AREA_INDEX',
    'CURRENT_FIX_BLINK_AROUND',
    'CURRENT_FIX_DURATION',
]

df = df[cols]

df['subject'] = 's' + df.RECORDING_SESSION_LABEL.str.replace('sub', '')
df['docid'] = 'd' + df.page.astype(str)
df['word'] = df.CURRENT_FIX_INTEREST_AREA_LABEL.str.strip()
df.word[df.word == '.'] = 'OFFSCREEN'
df['trial_ix'] = df.CURRENT_FIX_INTEREST_AREA_INDEX.str.replace('.', '0').astype(int) - 1
df['time'] = (df.TRIAL_START_TIME.astype(int) + df.PREVIOUS_SAC_END_TIME.str.replace('.', '0').astype(int)) / 1000
df.time = (df.time - df.groupby('RECORDING_SESSION_LABEL').time.transform('min')).round(3)
run = []
for key, x in df.groupby(['subject', 'docid']):
    trial_start = x.TRIAL_START_TIME
    starts = sorted(list(trial_start.unique()))
    _run = trial_start.apply(lambda x, starts=starts: starts.index(x))
    run.append(pd.Series(_run, index=x.index))
run = pd.concat(run)
run = run.sort_index() 
df['run'] = run + 1
df = df[df.run < 2]  # Delete repeat exposures
del df['run']
df = df.sort_values(['subject', 'time'])
df['blinkbeforefix'] = df.CURRENT_FIX_BLINK_AROUND.isin(['BEFORE', 'BOTH']).astype(int)
df['blinkafterfix'] = df.CURRENT_FIX_BLINK_AROUND.isin(['AFTER', 'BOTH']).astype(int)
gb = df.groupby(['subject', 'docid'])
df['offscreenbeforefix'] = (gb.word.shift(1) == 'OFFSCREEN').astype(int)
df['offscreenafterfix'] = (gb.word.shift(-1) == 'OFFSCREEN').astype(int)
df['startoffile'] = (df.docid != gb.docid.shift(1)).astype(int)
df['endoffile'] = (df.docid != gb.docid.shift(-1)).astype(int)


del df['RECORDING_SESSION_LABEL']
del df['page']
del df['TRIAL_START_TIME']
del df['PREVIOUS_SAC_END_TIME']
del df['CURRENT_FIX_INTEREST_AREA_LABEL']
del df['CURRENT_FIX_INTEREST_AREA_INDEX']
del df['CURRENT_FIX_BLINK_AROUND']

df = df[df.word != 'OFFSCREEN']
gb = df.groupby(['subject', 'docid'])
df['fdurSP'] = ((gb.time.shift(-1) - df.time) * 1000).astype(float)
df['fdurSPa'] = df.fdurSP
sel = df.fdurSP.isna()
df.fdurSP = np.where(sel, df.CURRENT_FIX_DURATION, df.fdurSP)
df.fdurSP = df.fdurSP.astype(int)
fdurs = get_fdurs(df)
for col in fdurs:
    df[col] = fdurs[col]
df.fdurSPsummed = df.fdurSPsummed.replace(-1, np.nan)
df.fdurFP = df.fdurFP.replace(-1, np.nan)
df.fdurGP = df.fdurGP.replace(-1, np.nan)
for col in fdurs:
    if 'SPsummed' in col and col != 'fdurSPsummed':
        sel = ~df.fdurSPsummed.isna()
        df[col] = df[col].where(sel, other=np.nan)
    elif 'FP' in col and col != 'fdurFP':
        sel = ~df.fdurFP.isna()
        df[col] = df[col].where(sel, other=np.nan)
    elif 'GP' in col and col != 'fdurGP':
        sel = ~df.fdurGP.isna()
        df[col] = df[col].where(sel, other=np.nan)

del df['CURRENT_FIX_DURATION']

df.to_csv(sys.stdout, sep=' ', index=False, na_rep='NaN')


