import sys

roi_mapping = {
    "Lang_LH_AngG": "LangLAngG",
    "Lang_LH_AntTemp": "LangLAntTemp",
    "Lang_LH_IFG": "LangLIFG",
    "Lang_LH_IFGorb": "LangLIFGorb",
    "Lang_LH_MFG": "LangLMFG",
    "Lang_LH_PostTemp": "LangLPostTemp",
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

for l in sys.stdin:
    l = l.strip()
    for roi in roi_mapping:
        if roi in l:
            l = l.replace(roi, roi_mapping[roi])
            continue
    print(l)
