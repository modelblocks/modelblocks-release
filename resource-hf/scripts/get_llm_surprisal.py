"""
Calculates LLM surprisal from the following LLM families:

GPT-2 family:
"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"

GPT-Neo family:
"EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B",
"EleutherAI/gpt-j-6B", "EleutherAI/gpt-neox-20b"

OPT family:
"facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b",
"facebook/opt-6.7b", "facebook/opt-13b", "facebook/opt-30b", "facebook/opt-66b"

Pythia family:
"EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "EleutherAI/pythia-1b",
"EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b", "EleutherAI/pythia-6.9b", "EleutherAI/pythia-12b",
each with checkpoints specified by training steps:
"step1", "step2", "step4", ..., "step142000", "step143000"
"""

import os, sys, torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoXTokenizerFast, GPTNeoXForCausalLM


def generate_stories(fn):
    stories = []
    f = open(fn)
    first_line = f.readline()
    assert first_line.strip() == "!ARTICLE"
    curr_story = ""

    for line in f:
        sentence = line.strip()
        if sentence == "!ARTICLE":
            stories.append(curr_story[:-1])
            curr_story = ""
        else:
            curr_story += line.strip() + " "

    stories.append(curr_story[:-1])
    return stories


def main():
    stories = generate_stories(sys.argv[1])
    model_variant = sys.argv[2].split("/")[-1]
    mode = sys.argv[-1]
    assert mode in {"token", "word"}, ValueError('Calculation mode must be "token" or "word"')

    if "gpt-neox" in model_variant:
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(sys.argv[2])
    elif "gpt" in model_variant:
        tokenizer = AutoTokenizer.from_pretrained(sys.argv[2], use_fast=False)
    elif "opt" in model_variant:
        tokenizer = AutoTokenizer.from_pretrained(sys.argv[2], use_fast=False)
    elif "pythia" in model_variant:
        tokenizer = AutoTokenizer.from_pretrained(sys.argv[2], revision=sys.argv[3])
    else:
        raise ValueError("Unsupported LLM variant")

    if "pythia" in model_variant:
        model = GPTNeoXForCausalLM.from_pretrained(sys.argv[2], revision=sys.argv[3])
    else:
        model = AutoModelForCausalLM.from_pretrained(sys.argv[2])

    model.eval()
    softmax = torch.nn.Softmax(dim=-1)
    ctx_size = model.config.max_position_embeddings
    bos_id = model.config.bos_token_id

    batches = []
    words = []
    for story in stories:
        words.extend(story.split(" "))
        tokenizer_output = tokenizer(story)
        ids = tokenizer_output.input_ids
        attn = tokenizer_output.attention_mask

        # these tokenizers do not append bos_id by default
        if "gpt" in model_variant or "pythia" in model_variant:
            ids = [bos_id] + ids
            attn = [1] + attn

        start_idx = 0

        # sliding windows with 50% overlap
        # start_idx is for correctly indexing the "later 50%" of sliding windows
        while len(ids) > ctx_size:
            # for models that explicitly require the first dimension (batch_size)
            if "gpt-neox" in model_variant or "pythia" in model_variant or "opt" in model_variant:
                batches.append((transformers.BatchEncoding({"input_ids": torch.tensor(ids[:ctx_size]).unsqueeze(0),
                                                            "attention_mask": torch.tensor(attn[:ctx_size]).unsqueeze(0)}),
                                torch.tensor(ids[1:ctx_size+1]), start_idx, True))
            # for other models
            elif "gpt" in model_variant:
                batches.append((transformers.BatchEncoding({"input_ids": torch.tensor(ids[:ctx_size]),
                                                            "attention_mask": torch.tensor(attn[:ctx_size])}),
                                torch.tensor(ids[1:ctx_size+1]), start_idx, True))

            ids = ids[int(ctx_size/2):]
            attn = attn[int(ctx_size/2):]
            start_idx = int(ctx_size/2)

        # remaining tokens
        if "gpt-neox" in model_variant or "pythia" in model_variant or "opt" in model_variant:
            batches.append((transformers.BatchEncoding({"input_ids": torch.tensor(ids[:-1]).unsqueeze(0),
                                                        "attention_mask": torch.tensor(attn[:-1]).unsqueeze(0)}),
                           torch.tensor(ids[1:]), start_idx, False))
        elif "gpt" in model_variant:
            batches.append((transformers.BatchEncoding({"input_ids": torch.tensor(ids[:-1]),
                                                        "attention_mask": torch.tensor(attn[:-1])}),
                           torch.tensor(ids[1:]), start_idx, False))

    print("word totsurp")
    curr_word_surp = []
    curr_toks = []
    curr_word_ix = 0
    is_continued = False
    for batch in batches:
        batch_input, output_ids, start_idx, will_continue = batch

        with torch.no_grad():
            model_output = model(**batch_input)

        toks = tokenizer.convert_ids_to_tokens(output_ids)
        index = torch.arange(0, output_ids.shape[0])
        surp = -1 * torch.log2(softmax(model_output.logits).squeeze(0)[index, output_ids])

        if mode == "token":
            # token-level surprisal
            for i in range(start_idx, len(toks)):
                cleaned_tok = tokenizer.convert_tokens_to_string([toks[i]]).replace(" ", "")
                print(cleaned_tok, surp[i].item())

        elif mode == "word":
            # word-level surprisal
            # if the batch starts a new story
            if not is_continued:
                curr_word_surp = []
                curr_toks = []
            for i in range(start_idx, len(toks)):
                curr_word_surp.append(surp[i].item())
                curr_toks += [toks[i]]
                curr_toks_str = tokenizer.convert_tokens_to_string(curr_toks)
                # summing token-level surprisal
                if words[curr_word_ix] == curr_toks_str.strip():
                    print(curr_toks_str.strip(), sum(curr_word_surp))
                    curr_word_surp = []
                    curr_toks = []
                    curr_word_ix += 1

        is_continued = will_continue

        del model_output


if __name__ == "__main__":
    main()