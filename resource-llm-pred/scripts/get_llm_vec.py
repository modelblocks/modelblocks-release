"""
GPT-2 family:
"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"

GPT-Neo family:
"EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B",
"EleutherAI/gpt-j-6B", "EleutherAI/gpt-neox-20b"

OPT family:
"facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b",
"facebook/opt-6.7b", "facebook/opt-13b", "facebook/opt-30b", "facebook/opt-66b"
"""

import os, sys, torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoXTokenizerFast
from collections import defaultdict
import sys
import json
import numpy as np
import random

def generate_stories(fn):

    f = open(fn).read()
    first_line = f.splitlines()[0]
    assert first_line.strip() == "!ARTICLE"

    # collect stories
    unproc_stor = f.split("!ARTICLE\n")
    unproc_stor = [story for story in unproc_stor if story != ""]

    split_stories = [] # each story, split by sent
    stories = [] # a list of stories, not split by sent
    for story in unproc_stor:
        sents = story.splitlines()
        split_stories.append(sents)
        stories.append(" ".join(sents))
    return stories, split_stories

def ds_rand(input_vec, rand_indices):
    """
    Down-sampling with randomization
    Note: The sampled indices are not sorted (e.g., can get a list of indices like: [6, 9, 3, 7, 2, 5, 4, 8])
    """
    rand_indices = torch.tensor(rand_indices)
    return input_vec[rand_indices]

def generate_json_fn(ds_mthd, model_variant, ds_vec_size):
    if ds_mthd == "std": 
        json_token_fp = '../../workspace/genmodel/llm_pred_temp/tokens_std-%s.json'%(model_variant)
        json_output_vec_fp = '../../workspace/genmodel/llm_pred_temp/output_vec_std-%s.json'%(model_variant)
    else:
        json_token_fp = '../../workspace/genmodel/llm_pred_temp/tokens_%s-%s-%d.json'%(ds_mthd, model_variant, ds_vec_size)
        json_output_vec_fp = '../../workspace/genmodel/llm_pred_temp/output_vec_%s-%s-%d.json'%(ds_mthd, model_variant, ds_vec_size)
    return json_token_fp, json_output_vec_fp

def main():
    docnames = ['Boar', 'Aqua', 'MatchstickSeller', 'KingOfBirds', 'Elvis', 'MrSticky', 'HighSchool', 'Roswell', 'Tulips', 'Tourettes']
    stories, split_stories = generate_stories(sys.argv[1])
    model_variant = sys.argv[2].split("/")[-1]
    ds_mthd = sys.argv[3]
    ds_vec_size = int(sys.argv[4])

    print()
    print("*model_variant: ", model_variant)
    print("*ds_mthd: ", ds_mthd)
    print("*ds_vec_size: ", ds_vec_size)
    print()

    if "gpt-neox" in model_variant:
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(sys.argv[2])
    elif "gpt" in model_variant:
        tokenizer = AutoTokenizer.from_pretrained(sys.argv[2], use_fast=False)
    elif "opt" in model_variant:
        tokenizer = AutoTokenizer.from_pretrained(sys.argv[2], use_fast=False)
    else:
        raise ValueError("Unsupported LLM variant")

    model = AutoModelForCausalLM.from_pretrained(sys.argv[2], output_hidden_states=True)
    model.eval()
    ctx_size = model.config.max_position_embeddings
    bos_id = model.config.bos_token_id

    batches = []
    words = []
    all_doc_ids = []
    all_story_sent_ids = []
    all_sent_pos = []
    curr_sent_id = 0
    for i, story in enumerate(stories):
        words.extend(story.split(" "))

        # get token ids and attn from nested list - a list of sentences
        split_sent_tok_output = tokenizer(split_stories[i])
        split_sent_tok_ids = split_sent_tok_output.input_ids
        split_sent_tok_attn = split_sent_tok_output.attention_mask
        
        # flatten the list
        ids = [idx for sent_token_ids in split_sent_tok_ids for idx in sent_token_ids]
        attn = [w for weights in split_sent_tok_attn for w in weights]
        start_idx = 0

        # generate doc_ids
        doc_ids = [docnames[i] for idx in ids]
        
        # generate sent_ids, and sentpos
        sent_ids = []
        sent_pos = []
        for sent in split_sent_tok_ids:
            for j, wd_id in enumerate(sent):
                sent_ids.append(curr_sent_id)
                sent_pos.append(j+1)
            curr_sent_id += 1

        # sliding windows with 50% overlap
        # start_idx is for correctly indexing the "later 50%" of sliding windows
        while len(ids) > ctx_size:
            # for GPT-NeoX (bos_id not appended by default)
            if "gpt-neox" in model_variant:
                batches.append((transformers.BatchEncoding({"input_ids": torch.tensor([bos_id] + ids[:ctx_size-1]).unsqueeze(0),
                                                           "attention_mask": torch.tensor([1] + attn[:ctx_size-1]).unsqueeze(0)}),
                                start_idx))

                # record doc_ids, sent_ids, and sentpos
                all_doc_ids.append([-1] + doc_ids[:ctx_size-1])
                all_story_sent_ids.append([-1] + sent_ids[:ctx_size-1])
                all_sent_pos.append([-1] + sent_pos[:ctx_size-1])

            # for GPT-2/GPT-Neo (bos_id not appended by default)
            elif "gpt" in model_variant:
                batches.append((transformers.BatchEncoding({"input_ids": torch.tensor([bos_id] + ids[:ctx_size-1]),
                                                            "attention_mask": torch.tensor([1] + attn[:ctx_size-1])}),
                                start_idx))
                # record doc_ids, sent_ids, and sentpos
                all_doc_ids.append([-1] + doc_ids[:ctx_size-1])
                all_story_sent_ids.append([-1] + sent_ids[:ctx_size-1])
                all_sent_pos.append([-1] + sent_pos[:ctx_size-1])

            # for OPT (bos_id appended by default)
            else:
                batches.append((transformers.BatchEncoding({"input_ids": torch.tensor(ids[:ctx_size]).unsqueeze(0),
                                                        "attention_mask": torch.tensor(attn[:ctx_size]).unsqueeze(0)}),
                                start_idx))

                # record doc_ids, sent_ids, and sentpos
                # still add a [-1] at the beginning bc for OPT, bos_id is appended by default
                all_doc_ids.append([-1] + doc_ids[:ctx_size])
                all_story_sent_ids.append([-1] + sent_ids[:ctx_size])
                all_sent_pos.append([-1] + sent_pos[:ctx_size])

            ids = ids[int(ctx_size/2):]
            attn = attn[int(ctx_size/2):]
            start_idx = int(ctx_size/2)-1

            # record doc_ids, sent_ids, and sentpos
            doc_ids = doc_ids[int(ctx_size/2):]
            sent_ids = sent_ids[int(ctx_size/2):]
            sent_pos = sent_pos[int(ctx_size/2):]

        # remaining tokens
        if "gpt-neox" in model_variant:
            batches.append((transformers.BatchEncoding({"input_ids": torch.tensor([bos_id] + ids).unsqueeze(0),
                                                       "attention_mask": torch.tensor([1] + attn).unsqueeze(0)}),
                           start_idx))
            
            # record doc_ids, sent_ids, and sentpos
            all_doc_ids.append([-1] + doc_ids)
            all_story_sent_ids.append([-1] + sent_ids)
            all_sent_pos.append([-1] + sent_pos)

        elif "gpt" in model_variant:
            batches.append((transformers.BatchEncoding({"input_ids": torch.tensor([bos_id] + ids),
                                                       "attention_mask": torch.tensor([1] + attn)}),
                           start_idx))
            # record doc_ids, sent_ids, and sentpos
            all_doc_ids.append([-1] + doc_ids)
            all_story_sent_ids.append([-1] + sent_ids)
            all_sent_pos.append([-1] + sent_pos)
        else:
            batches.append((transformers.BatchEncoding({"input_ids": torch.tensor(ids).unsqueeze(0),
                                                        "attention_mask": torch.tensor(attn).unsqueeze(0)}),
                            start_idx))
            # record doc_ids, sent_ids, and sentpos
            all_doc_ids.append([-1] + doc_ids)
            all_story_sent_ids.append([-1] + sent_ids)
            all_sent_pos.append([-1] + sent_pos)

    # for getting indices for time stamps
    curr_word_ix = 0
    output_vec = defaultdict(list)
    tokens = []
    model2vec_size = {"gpt2": 768, "gpt2-medium": 1024, "gpt2-large": 1280, "gpt2-xl": 1600, "gpt-neo-125M": 768, 
                      "gpt-neo-1.3B": 2048, "gpt-neo-2.7B": 2560, "gpt-j-6B": 4096, "gpt-neox-20b": 6144, "opt-125m": 768,
                      "opt-350m": 512, "opt-1.3b": 2048, "opt-2.7b": 2560, "opt-6.7b": 4096, "opt-13b": 5120, 
                      "opt-30b": 7168, "opt-66b": 9216}
    
    # for down-sampling, if specified
    rand_indices = random.sample(range(model2vec_size[model_variant]), ds_vec_size)
    for batch_idx, batch in enumerate(batches):

        batch_input, start_idx = batch

        with torch.no_grad():
            model_output = model(**batch_input)

        toks = tokenizer.convert_ids_to_tokens(batch_input.input_ids.squeeze(0))[1:]
        doc_ids = all_doc_ids[batch_idx][1:]
        sent_ids = all_story_sent_ids[batch_idx][1:]
        sent_pos = all_sent_pos[batch_idx][1:]

        for i in range(start_idx, len(toks)):
            # necessary for diacritics in Dundee
            cleaned_tok = toks[i].replace("Ġ", "", 1).encode("latin-1").decode("utf-8")

            # OPT adds </s> but we won't want it to be output
            if cleaned_tok == "</s>":
                continue
            
            tokens.append(cleaned_tok)
            output_vec["word"].append(cleaned_tok)
            output_vec["docid"].append(doc_ids[i])
            output_vec["sentid"].append(sent_ids[i])
            output_vec["sentpos"].append(sent_pos[i])

            # store word indices: for getting time stamps in later steps
            output_vec["timeid"].append(curr_word_ix)
            
            if ("gpt-neox" in model_variant) or ("opt" in model_variant):
                vec = model_output.hidden_states[-1][0][i+1] # because there is a bos at the beginning
            elif "gpt" in model_variant:
                vec = model_output.hidden_states[-1][i+1] # because there is a bos at the beginning

            # down-sample, if specified
            if ds_mthd == "rand":
                vec = ds_rand(vec, rand_indices)

            vec = vec.tolist()
            for feat_num, feat in enumerate(vec):
                col_name = "feat%d"%(feat_num)
                output_vec[col_name].append(feat)
            
            # update index for getting time stamp
            words[curr_word_ix] = words[curr_word_ix].replace(cleaned_tok, "", 1)
            if words[curr_word_ix] == "":
                curr_word_ix += 1

    json_token_fp, json_output_vec_fp = generate_json_fn(ds_mthd, model_variant, ds_vec_size)
    
    with open(json_token_fp, 'w') as f:
        json.dump(tokens, f, indent=4)
        
    with open(json_output_vec_fp, 'w') as f:
        json.dump(output_vec, f)

    print()
    print("json_token_fp: ", json_token_fp)
    print("json_output_vec_fp: ", json_output_vec_fp)
    print()
        

if __name__ == "__main__":
    main()
    
