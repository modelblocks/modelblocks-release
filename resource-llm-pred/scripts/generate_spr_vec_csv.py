"""
What this script does: Combining subwords and generating a new vec representation file for Futrell2018 dataset

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

# get the vec_rep file that is generated for the fmri exp
if ds_mthd == "std":
    orig_vec_rep_fn = '../../workspace/genmodel/llm_pred_temp/vec_rep_std-%s'%(model_variant)
else:
    orig_vec_rep_fn = '../../workspace/genmodel/llm_pred_temp/vec_rep_%s-%s-%s'%(ds_mthd, model_variant, ds_vec_size)

orig_vec_rep = pd.read_csv(orig_vec_rep_fn, sep=' ', skipinitialspace=True)
col_names = list(orig_vec_rep.columns.values)

# combine subwords and get averaged vec representations
spr_vec_rep_list = []
max_timeid = orig_vec_rep.iloc[-1]["timeid"]
for curr_timeid in range(max_timeid+1):

    vec_lines = orig_vec_rep.loc[orig_vec_rep["timeid"] == curr_timeid]
    only_vecs = vec_lines.drop(["word", "docid", "sentid", "sentpos", "timeid"], axis=1)
    avg_vec = only_vecs.mean(axis=0)
    avg_vec_list = avg_vec.tolist()

    vec_indices = list(vec_lines.index.values)
    vec_indices.sort()
    
    curr_word = ""
    for i in vec_indices:
        curr_word += vec_lines["word"][i]
    
    docid = vec_lines["docid"][vec_indices[0]]
    sentid = vec_lines["sentid"][vec_indices[0]]
    sentpos = vec_lines["sentpos"][vec_indices[0]]
    timeid = vec_lines["timeid"][vec_indices[0]]

    new_vec_line = [curr_word, docid, sentid, sentpos, timeid] # sentpos will be changed and timeid will be removed
    new_vec_line += avg_vec_list
    spr_vec_rep_list.append(new_vec_line)

spr_vec_rep = pd.DataFrame(spr_vec_rep_list, columns = col_names)

# adjust sentpos
docids = list(set(spr_vec_rep["docid"]))
for docid in docids:
    vec_lines = spr_vec_rep.loc[spr_vec_rep["docid"] == docid]
    doc_sentids = list(set(vec_lines["sentid"]))
    doc_sentids.sort()

    for doc_sentid in doc_sentids:
        doc_sentid_lines = vec_lines.loc[vec_lines["sentid"] == doc_sentid] # the words in the same sentence of the same doc
        
        doc_sentid_lines_indices = list(set(doc_sentid_lines.index.values))
        doc_sentid_lines_indices.sort()

        sentpos = 1
        for doc_sentid_line_id in doc_sentid_lines_indices:
            spr_vec_rep.at[doc_sentid_line_id, "sentpos"] = sentpos
            sentpos += 1

# remove timeid column
spr_vec_rep.drop("timeid", axis=1, inplace = True)

# output vec file for self-paced reading data
spr_vec_rep.to_csv(sys.stdout, index=False, sep=' ')