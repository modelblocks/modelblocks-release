#!/usr/bin/bash

MDL_VAR=$1

source ~/.bashrc
cd ~/modelblocks-release/resource-llm-pred/scripts

conda deactivate
# source ~/miniconda3/bin/activate llm-pred
source ~/miniconda3/bin/activate fmri-pred

# for storing intermediate files (will be deleted)
mkdir ~/modelblocks-release/workspace/genmodel/llm_pred_temp

# input arguments:
# (1) naturalstories_corpus
# (2) model variant (please refer to the script for the full list of supported variants)
# (3) down-sampling method: std (standard; no down-sampling applied) / rand (randomization)
# (4) down-sampling size
# Note: This script outputs two json files to an intermediate directory, which will be deleted afterwards
echo "get_llm_vec.py: getting llm vector repres"
python get_llm_vec.py ~/modelblocks-release/workspace/genmodel/naturalstories.sentitems ${MDL_VAR} std 0

conda deactivate
source ~/miniconda3/bin/activate mb

# input arguments:
# (1) model variant: for GPT-Neo and OPT family, exclude anything before the slash (the LM family name) and the slash
# (2) down-sampling method: std/rand
# (3) down-sampling size
echo "generate_vec_csv.py: generating vec csv file"
python generate_vec_csv.py ${MDL_VAR} std 0

# input arguments:
# (1) model variant: for GPT-Neo and OPT family, exclude anything before the slash (the LM family name) and the slash
# (2) down-sampling method: std/rand
# (3) down-sampling size
# output file:
# std: vec_rep_std-%s'%(model_variant)
# down-sampled: vec_rep_%s-%s-%s'%(ds_mthd, model_variant, ds_vec_size)
echo "map_time.py: mapping time to vec csv file (for hrf convolution)"
python map_time.py ${MDL_VAR} std 0

# input argument:
# (1) file to be convolved
echo "hrf_convolve_predictors.py"
python ../../resource-fmri/scripts/hrf_convolve_predictors.py ../../workspace/genmodel/llm_pred_temp/vec_rep_w_time_std-${MDL_VAR} > ../../workspace/genmodel/vec_rep_hrf_${MDL_VAR}

# input arguments:
# (1) train response data
# (2) test response data
# (3) vector file (fmri; hrf convolved)
# (4) fROI (all + 6 fROIs): 'ALL', 'LangLIFGorb', 'LangLPostTemp', 'LangLMFG', 'LangLAntTemp', 'LangLAngG', 'LangLIFG'
echo "linear_reg.py"
python linear_reg.py ~/modelblocks-release/workspace/genmodel/naturalstoriesfmri_Lang.t.fmri-bywrd.expl+held_part.resmeasures ~/modelblocks-release/workspace/genmodel/naturalstoriesfmri_Lang.t.fmri-bywrd.expl+held_part.resmeasures ~/modelblocks-release/workspace/genmodel/vec_rep_hrf_${MDL_VAR} ALL

# # remove temp directory for storing intermediate files
# # rm -r ~/modelblocks-release/workspace/genmodel/llm_pred_temp