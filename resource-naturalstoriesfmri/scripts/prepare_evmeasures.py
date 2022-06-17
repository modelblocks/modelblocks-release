import sys, numpy, math, re, pandas as pd
from collections import defaultdict


T_MIN = 1
T_MAX = 226

MD_ROI_MAPPING = {
    "MD_LH_antParietal": "MDlanglocLAntPar",
    "MD_LH_insula": "MDlanglocLInsula",
    "MD_LH_medialFrontal": "MDlanglocLmPFC",
    "MD_LH_midFrontal": "MDlanglocLMFG",
    "MD_LH_midFrontalOrb": "MDlanglocLMFGorb",
    "MD_LH_midParietal": "MDlanglocLMidPar",
    "MD_LH_postParietal": "MDlanglocLPostPar",
    "MD_LH_Precental_B_IFGop": "MDlanglocLIFGop",
    "MD_LH_Precentral_A_PrecG": "MDlanglocLPrecG",
    "MD_LH_supFrontal": "MDlanglocLSFG",
    "MD_RH_antParietal": "MDlanglocRAntPar",
    "MD_RH_insula": "MDlanglocRInsula",
    "MD_RH_medialFrontal": "MDlanglocRmPFC",
    "MD_RH_midFrontal": "MDlanglocRMFG",
    "MD_RH_midFrontalOrb": "MDlanglocRMFGorb",
    "MD_RH_midParietal": "MDlanglocRMidPar",
    "MD_RH_postParietal": "MDlanglocRPostPar",
    "MD_RH_Precental_B_IFGop": "MDlanglocRIFGop",
    "MD_RH_Precentral_A_PrecG": "MDlanglocRPrecG",
    "MD_RH_supFrontal": "MDlanglocRSFG"
}


def correct_roi(roi):
    if roi in MD_ROI_MAPPING:
        return MD_ROI_MAPPING[roi]
    elif re.match("Lang_LH_", roi):
        return re.sub("Lang_LH_", "LangL", roi)
    elif re.match("Lang_RH_", roi):
        return re.sub("Lang_RH_", "LangR", roi)
    elif re.match("Aud_LH_", roi):
        return re.sub("Aud_LH_", "AudL", roi)
    elif re.match("Aud_RH_", roi):
        return re.sub("Aud_RH_", "AudR", roi)
    else: raise ValueError("Unfamiliar ROI: " + roi)


def earliest_nan(time_series):
    earliest_ix = 0
    for i, item in enumerate(time_series):
        if pd.isna(item):
            break
        earliest_ix += 1
    return earliest_ix
    

relevant_networks = ["Aud_Anat", "Lang_SN", "MD_HE"]

# wide-format dataframe
df = pd.read_csv(sys.stdin)

# key -> ROI -> time series
data = defaultdict(dict)

for _, row in df.iterrows():
    # ignore data from irrelevant networks
    if row["Network"] not in relevant_networks: continue
    # ignore empty time series
    #if pd.isna(row["T_{}".format(T_MIN)]): continue
    key = (row["UID"], row["Story"], row["Run"])
    roi = row["ROI"]
    roi = correct_roi(roi)
    roi_bold = list()
    for t in range(T_MIN, T_MAX+1):
        bold_t = row["T_{}".format(t)]
        # don't add padding to time series
        #if pd.isna(bold_t):
        #    break
        roi_bold.append(bold_t)
    data[key][roi] = roi_bold

# make sure the same set of per-ROI time series exists for each (UID, Story, Run)
roi_sets = [ set(data[k].keys()) for k in data ]
assert all( s == roi_sets[0] for s in roi_sets )

# trim padding from the end of each time series. the amount of padding trimmed
# off is based on the longest time series (the one with the latest first_nan_index)
for key in data:
    per_roi_time_series = data[key]
    max_first_nan_index = 0
    for series in per_roi_time_series.values():
        first_nan_index = earliest_nan(series)
        max_first_nan_index = max([max_first_nan_index, first_nan_index])
    for roi in per_roi_time_series:
        per_roi_time_series[roi] = per_roi_time_series[roi][:max_first_nan_index]

rois = sorted(roi_sets[0])
roi_bold_cols = ["bold_"+roi for roi in rois]
col_labels = ["subject", "docid", "run", "tr", "time", "splitVal15"] + roi_bold_cols
print(" ".join(col_labels))

for key in data:
    max_t_of_key = T_MIN + len(data[key][rois[0]]) - 1
    for t in range(T_MIN, max_t_of_key+1):
        # t is the index of the sample. Samples are taken two seconds apart
        tr = t-1
        actual_t = 2 * (t-1)
        subj_number = key[0]
        split_val_15 = int((subj_number + t) / 15)
        cols = list(key) + [tr, actual_t, split_val_15]
        for roi in rois:
            cols.append(data[key][roi][t-1])
        print(" ".join(str(c) for c in cols))
