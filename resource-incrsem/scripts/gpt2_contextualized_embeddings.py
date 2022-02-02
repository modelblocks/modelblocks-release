import argparse, sys, torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


def get_subword_windows(encoded_text, context_size=1024):
    """Splits encoded_text, a list of subword IDs, into a list of
    overlapping windows of subwords, each of length <= context_size. These
    windows are what get fed into GPT-2."""

    # Need an even context_size for splitting the subwords into windows
    if context_size % 2 == 1:
        raise ValueError("context_size must be an even number")
    windows = list()
    while len(encoded_text) > context_size:
        windows.append(encoded_text[:context_size])
        # size of overlap between windows is context_size/2
        encoded_text = encoded_text[int(context_size/2):]
    windows.append(encoded_text)
    return windows


def combine_window_embeddings(window_embeddings, context_size=1024):
    # use all subword embeddings from the first window. For subsequent
    # windows, the first context_size/2 terms will overlap with the
    # previous window, so throw them out
    subword_embeddings = window_embeddings[0]
    for w in window_embeddings[1:]:
        subword_embeddings.append(w[int(context_size/2):])
    return subword_embeddings

def word_wordpiece_alignment(words, wordpieces):
    """Returns a list of length len(words), where position i contains
    the set of wordpieces that overlap with word i. These wordpieces'
    embeddings will be averaged downstream to get an embedding for word i"""

    # verify that words and wordpieces consist of the same character sequence,
    # other than \u0120
    wp_chars = "".join(wordpieces)
    # G with dot, \u0120, is used for spaces, which we don't care about
    wp_chars = wp_chars.replace("\u0120", "")
    w_chars = "".join(words)
    assert wp_chars == w_chars, "wp_chars: {}\nw_chars: {}".format(wp_chars, w_chars)

    # char2w[i] gives the index of the word to which w_chars[i] belongs
    char2w = list()
    for i, w in enumerate(words):
        char2w.extend([i]*len(w))

    # char2wp[i] gives the index of the wordpiece to which w_chars[i] belongs
    char2wp = list()
    for i, wp in enumerate(wordpieces):
        wp = wp.replace("\u0120", "")
        char2wp.extend([i]*len(wp))

    assert len(char2w) == len(char2wp)

    # w2wp[i] gives the set of wordpiece indices that overlap at all with
    # word i
    w2wp = [set()] * len(words)
    for i in range(len(char2w)):
        w2wp[char2w[i]].add(char2wp[i])

    return w2wp
        

#def word_wordpiece_alignment(words, wordpieces):
#    """Returns a list of length len(words), where the value at index
#    i is the index of the wordpiece corresponding to the start of
#    words[i]"""
#    # wordpiece index
#    wp_ix = 0
#    # word index
#    w_ix = 0
#    alignment = list()
#    while w_ix < len(words):
#        start = wp_ix
#        wp = wordpieces[wp_ix]
#        w = words[w_ix]
#        # strip away the leading G with dot, which is how spaces are
#        # represented in BPE
#        if wp.startswith("\u0120"):
#            wp = wp[1:]
#
#        if wp == w:
#            end = wp_ix + 1
#        elif wp.startswith(w):
#            end = wp_ix + 1
#            combined_w = w
#            while wp != combined_w:
#                assert wp.startswith(combined_w), "combined_w: {} wp: {}".format(combined_w, wp)
#                alignment.append((start, end))
#                w_ix += 1
#                combined_w += words[w_ix]
#        else:
#            assert w.startswith(wp)
#            combined_wp = wp
#            while combined_wp != w:
#                assert w.startswith(combined_wp), "word: {} combined_wp: {}".format(word, combined_wp)
#                wp_ix += 1
#                combined_wp += wordpieces[wp_ix]
#            end = wp_ix + 1
#
#        alignment.append((start, end))
#        w_ix += 1
#        wp_ix += 1
#
#    return alignment
            

def wordpiece_emb_2_word_emb(wordpiece_embs, wordpiece_text, words):
    """wordpiece_embs is a list of per-word piece embeddings, and 
    wordpiece_text is a list of the word pieces to which the embeddings
    correspond. words contains the same sequence of characters, but 
    tokenized into full words with punctuation separated. This function
    combines embeddings of word pieces in order to get one embedding per
    word in words"""
    assert len(wordpiece_embs) == len(wordpiece_text)
    alignment = word_wordpiece_alignment(
        words, wordpiece_text)
    assert len(alignment) == len(words)
    word_embs = list()
    for wp_set in alignment:
        subword_embs = list()
        for wp_ix in wp_set:
            subword_embs.append(wordpiece_embs[wp_ix])
        word_embs.append(sum(subword_embs)/len(subword_embs))

#    for start, end in alignment:
#        subword_embs = list()
#        for i in range(start, end):
#            subword_embs.append(wordpiece_embs[i])
#        # average the list of subword embeddings to get an embedding
#        # for the whole word
#        word_embs.append(sum(subword_embs)/len(subword_embs))
    return word_embs


def get_word_embeddings(tokenizer, model, input_str, input_tok, layer):
    input_ids = tokenizer(input_str, return_tensors="pt")["input_ids"][0]
    input_id_text = tokenizer.convert_ids_to_tokens(input_ids)
    windowed_input_ids = get_subword_windows(input_ids)
    #outputs = model(**inputs, output_hidden_states=True)
    #outputs = model(input_ids, output_hidden_states=True)
    #embs = list(outputs["hidden_states"][-1][0])
    per_window_embs = list()
    for window in windowed_input_ids:
        outputs = model(window, output_hidden_states=True)
        embs = list(outputs["hidden_states"][layer])
        per_window_embs.append(embs)
    combined_embs = combine_window_embeddings(per_window_embs)

    # combined_embs contains one embedding per word piece. Roll these
    # together so there's one embedding per token in the input linetoks
    word_embs = wordpiece_emb_2_word_emb(combined_embs, input_id_text,
         input_tok)
    return word_embs


def print_article_embeddings(toks, embeddings, article_ix):
    assert len(toks) == len(embeddings)
    for i, tok in enumerate(toks):
        emb = embeddings[i]
        emb_str = " ".join(str(x.item()) for x in emb)
        print("{} {} {}".format(article_ix, tok, emb_str))


def get_gpt2_embeddings(sentitems, senttoks, gpt2_version="gpt2", layer=-1):
    model = GPT2LMHeadModel.from_pretrained(gpt2_version)
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_version)

    items = open(sentitems)
    toks = open(senttoks)
    assert items.readline().strip() == "!ARTICLE" \
        and toks.readline().strip() == "!ARTICLE"

    all_toks = list()
    all_embeddings = list()
    all_article_ix = list()

    curr_input_str = ""
    curr_input_toks = list()
    curr_article_ix = 0
    
    for l_item in items:
        l_item = l_item.strip()
        l_tok = toks.readline().strip()
        if l_item == "!ARTICLE":
            assert l_tok == "!ARTICLE"
            embeddings = get_word_embeddings(tokenizer, model, curr_input_str,
                curr_input_toks, layer)
            all_embeddings.extend(embeddings)
            all_article_ix.extend([curr_article_ix]*len(curr_input_toks))
            all_toks.extend(curr_input_toks)
            curr_input_toks = list()
            curr_input_str = ""
            curr_article_ix += 1
        elif curr_input_str:
            curr_input_str += " " + l_item
            curr_input_toks.extend(l_tok.split())
        else:
            curr_input_str = l_item
            curr_input_toks = l_tok.split()
    embeddings = get_word_embeddings(tokenizer, model, curr_input_str,
        curr_input_toks, layer)
    all_embeddings.extend(embeddings)
    all_article_ix.extend([curr_article_ix]*len(curr_input_toks))
    all_toks.extend(curr_input_toks)
    return {
        "tokens": all_toks,
        "embeddings": all_embeddings,
        "article_indices": all_article_ix
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sentitems", type=str)
    parser.add_argument("senttoks", type=str)
    parser.add_argument("--gpt2_version", type=str, default="gpt2",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])
    parser.add_argument("--layer", type=int, default=-1)
    args = parser.parse_args()
    result = get_gpt2_embeddings(args.sentitems, args.senttoks,
        args.gpt2_version)
    print_article_embeddings(result["tokens"], result["embeddings"],
        result["article_indices"])



if __name__ == "__main__":
    main()
