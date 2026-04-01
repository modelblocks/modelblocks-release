#!/usr/bin/bash

MDL_VAR=$1

source ~/.bashrc
cd ~/modelblocks-release/workspace

#####################################################################
# (0) copy the lme formula for the pereira fmri corpus to the 
#     designated directory in the pipeline
#####################################################################

cp ~/modelblocks-release/linshuler_2026_eacl/scripts/fmri-sent.lmerform ~/modelblocks-release/workspace/scripts/fmri-sent.lmerform

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

make genmodel/pereira.${MDL_VAR}.tokmeasures

#####################################################################
# (2) get sentence-final surprisal and collect dataset info 
#     (e.g., passage position, sentence length, BOLD signals, etc.)
#####################################################################

make genmodel/pereira-sent.${MDL_VAR}.prdmeasures

#####################################################################
# (3) get loglikelihood for the held-out partition
#####################################################################

touch genmodel/pereira.sentitems
touch genmodel/pereira.${MDL_VAR}.tokmeasures
touch genmodel/pereira.${MDL_VAR}.tokmeasures.log
touch genmodel/pereira-sent.${MDL_VAR}.prdmeasures

make genmodel/pereira-sent.${MDL_VAR}.fit_fmri-sent_lastwordsurp_lmer.held_predicted.loglik
