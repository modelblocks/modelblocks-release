#!/usr/bin/bash

MDL_VAR=$1

source ~/.bashrc
cd ~/modelblocks-release/workspace

#####################################################################
# (0) copy the lme formula for the natural stories corpus to the 
#     designated directory in the pipeline
#####################################################################

cp ~/modelblocks-release/linshuler_2026_eacl/scripts/fmri.lmerform ~/modelblocks-release/workspace/scripts/fmri.lmerform

#####################################################################
# (1) get token-level surprisal
#     model variants evaluated in this work:
#     * [gpt2 family]: gpt2, gpt2-medium, gpt2-large, gpt2-xl
#     * [gpt-neo family]: gpt-neo-125m, gpt-neo-1300m, gpt-neo-2700m, 
#                         gpt-j-6000m, gpt-neox-20000m
#     * [opt family]: opt-125m, opt-350m, opt-1300m, opt-2700m, 
#                     opt-6700m, opt-13000m, opt-30000m, opt-66000m
#####################################################################

conda deactivate
source ~/miniconda3/bin/activate hf_env

make genmode/naturalstories.${MDL_VAR}.tokmeasures

#####################################################################
# (2) get word level surprisal and collect dataset info 
#     (e.g., document id, word position within a sentence, 
#     BOLD signal for each data point, etc.)
#####################################################################

make genmodel/naturalstoriesfmri_Lang.t.${MDL_VAR}.t.all-itemmeasures

#####################################################################
# (3) convolve predictors with HRF
#####################################################################

make genmodel/naturalstoriesfmri_Lang.t.${MDL_VAR}.hrf.all-itemmeasures

#####################################################################
# (4) get loglikelihood for the held-out partition
#####################################################################

make genmodel/naturalstoriesfmri_Lang.t.${MDL_VAR}.hrf.fmri-bywrd.fit_fmri_totsurp_lmer.held_predicted.loglik
