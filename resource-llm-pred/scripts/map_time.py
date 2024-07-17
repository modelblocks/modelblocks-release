"""
Supported Arguments for model variant:
* For GPT-Neo and OPT family, everything before the slash shown in the get_llm_vec.py 
  file (i.e., EleutherAI/ and facebook/) is not needed in any step after get_llm_vec.py

GPT-2 family:
"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"

GPT-Neo family:
"gpt-neo-125M", "gpt-neo-1.3B", "gpt-neo-2.7B",
"gpt-j-6B", "gpt-neox-20b"

OPT family:
"opt-125m", "opt-350m", "opt-1.3b", "opt-2.7b",
"opt-6.7b", "opt-13b", "opt-30b", "opt-66b"
"""

import pandas as pd
import sys

model_variant = sys.argv[1]
ds_mthd = sys.argv[2]
ds_vec_size = sys.argv[3]

if ds_mthd == "std":
    vec_rep_fn = '../../workspace/genmodel/llm_pred_temp/vec_rep_std-%s'%(model_variant)
else:
    vec_rep_fn = '../../workspace/genmodel/llm_pred_temp/vec_rep_%s-%s-%s'%(ds_mthd, model_variant, ds_vec_size)

all_item_fn = "~/modelblocks-release/workspace/genmodel/naturalstoriesfmri_Lang.t.all-itemmeasures"

vec_rep = pd.read_csv(vec_rep_fn, sep=' ', skipinitialspace=True)
all_item_meas = pd.read_csv(all_item_fn, sep=' ', skipinitialspace=True)

time_ids = vec_rep.loc[:,"timeid"]
all_item_meas_time = all_item_meas.loc[:,"time"]

vec_rep_time = []
for time_id in time_ids:
    vec_rep_time.append(all_item_meas_time[time_id])

vec_rep.insert(loc=4, column="time", value=vec_rep_time)
vec_rep.drop("timeid", axis=1, inplace = True)

if ds_mthd == "std":
    output_file = '../../workspace/genmodel/llm_pred_temp/vec_rep_w_time_std-%s'%(model_variant)
else:
    output_file = '../../workspace/genmodel/llm_pred_temp/vec_rep_w_time_%s-%s-%s'%(ds_mthd, model_variant, ds_vec_size)

vec_rep.to_csv(output_file, index=False, sep=' ')