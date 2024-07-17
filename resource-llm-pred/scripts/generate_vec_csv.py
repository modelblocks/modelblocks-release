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
import json
import sys

def generate_json_fn(ds_mthd, model_variant, ds_vec_size):
    if ds_mthd == "std": 
        json_token_fp = '../../workspace/genmodel/llm_pred_temp/tokens_std-%s.json'%(model_variant)
        json_output_vec_fp = '../../workspace/genmodel/llm_pred_temp/output_vec_std-%s.json'%(model_variant)
    else:
        json_token_fp = '../../workspace/genmodel/llm_pred_temp/tokens_%s-%s-%s.json'%(ds_mthd, model_variant, ds_vec_size)
        json_output_vec_fp = '../../workspace/genmodel/llm_pred_temp/output_vec_%s-%s-%s.json'%(ds_mthd, model_variant, ds_vec_size)
    return json_token_fp, json_output_vec_fp

if __name__ == '__main__':
    model_variant = sys.argv[1]
    ds_mthd = sys.argv[2]
    ds_vec_size = sys.argv[3]
    
    json_token_fp, json_output_vec_fp = generate_json_fn(ds_mthd, model_variant, ds_vec_size)

    if ds_mthd == "std":
        output_vec_rep = '../../workspace/genmodel/llm_pred_temp/vec_rep_std-%s'%(model_variant)
    else:
        output_vec_rep = '../../workspace/genmodel/llm_pred_temp/vec_rep_%s-%s-%s'%(ds_mthd, model_variant, ds_vec_size)

    with open(json_output_vec_fp, 'r') as f:
        output_vec = json.load(f)

    with open(json_token_fp, 'r') as f:
        tokens = json.load(f)

    df = pd.DataFrame(output_vec, index = tokens)
    df.to_csv(output_vec_rep, index=False, sep=' ')