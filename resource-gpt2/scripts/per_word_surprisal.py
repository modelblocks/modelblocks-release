#!/usr/bin/env python3

import fire, json, os
import tensorflow as tf

import encoder, surprisal, model

BATCH_SIZE = 1
DELIMITER = '!ARTICLE'


def get_delimited_texts(lineitems_fn):
    '''Returns the texts in the delim.linetimes file. Each text
    is preceded by a line with a special delimiter token. Output is a
    list of strings, where each string is one complete text'''
    texts = list()

    f = open(lineitems_fn)
    firstline = f.readline()
    assert firstline.strip() == DELIMITER

    curr_text = list()
    for l in f:
        sentence = l.strip()
        if sentence == DELIMITER:
            texts.append(' '.join(curr_text))
            curr_text = list()
        else:
            curr_text.append(sentence)
    # don't forget the last text
    texts.append(' '.join(curr_text))
        
    return texts


def get_subword_windows(encoded_text, context_size):
    '''Splits encoded_text, a list of subword IDs, into a list of
    overlapping windows of subwords, each of length <= context_size. These
    windows are what get fed into GPT-2.'''

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


def combine_window_surprisals(window_surprisals, context_size):
    '''Given the surprisal measurements for a list of overlapping windows,
    returns the recombined list of subword surprisals.'''
    # use all subword surprisals from the first window. For subsequent
    # windows, the first context_size/2 terms will overlap with the
    # previous window, so throw them out
    subword_surprisals = window_surprisals[0]
    for w in window_surprisals[1:]:
        subword_surprisals.extend(w[int(context_size/2):])
    return subword_surprisals


def get_per_char_surprisal(subwords, subword_surps):
    # subword_surps[i] is the surprisal for subwords[i]
    assert len(subwords) == len(subword_surps)
    chars = list()
    char_surps = list()
    for sub, surp in zip(subwords, subword_surps):
        per_char_surp = surp/len(sub)
        for char in sub:
            chars.append(char)
            char_surps.append(per_char_surp)
    assert len(chars) == len(char_surps)
    return chars, char_surps


def roll_subword_surprisal(subwords, subword_surps, words):
    chars, char_surps = get_per_char_surprisal(subwords, subword_surps)
    word_surps = list()
    char_index = 0
    curr_char = chars[char_index]
    curr_surp = char_surps[char_index]
    for w in words:
        word_surp = 0
        # roll surprisals from spaces before the word into the word
        while curr_char.isspace():
            word_surp += curr_surp
            char_index += 1
            curr_char = chars[char_index]
            curr_surp = char_surps[char_index]
        for c in w:
            assert c == curr_char
            word_surp += curr_surp
            char_index += 1
            # at the very end of the string we can't advance
            if char_index < len(chars):
                curr_char = chars[char_index]
                curr_surp = char_surps[char_index]
        word_surps.append(word_surp)
    return word_surps


def per_word_surprisal(
    lineitems, 
    model_name='124M',
    models_dir='models',
    context_size=1024
):
    """
    Use GPT-2 to calculate per-word surprisal for a provided lineitems file

    Parameters
    ----------
    lineitems :
        path to delimited lineitems file (*.delim.lineitems)

    model_name :
        which model to use

    models_dir :
        path to parent folder containing model subfolders
        (i.e. contains the <model_name> folder)

    context_size :
        the maximum context size allowed by the model (n_ctx). 
        If the length of the input text exceeds context_size, the text is 
        split into overlapping windows to calculate surprisal
    """

    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    # prepare each text as a single string to feed into GPT-2
    texts = get_delimited_texts(lineitems)

    all_words = list()
    all_word_surprisals = list()
    for text in texts:
        # get sequence of subword encodings
        enc = encoder.get_encoder(model_name, models_dir)
        enc_text = enc.encode(text)
        subwords = [enc.decode([x]) for x in enc_text]
        words = text.split()
    
        # split subword sequence into windows of size <= context_size
        windows = get_subword_windows(enc_text, context_size)
    
        # feed each window into GPT-2, get per-subword surprisals
        window_surprisals = list()
        for window in windows:
            with tf.Session(graph=tf.Graph()) as sess:
        
                output = surprisal.get_per_subword_surprisal(
                    corpus=window, hparams=hparams,
                    encoder=enc
                )
        
                saver = tf.train.Saver()
                ckpt = tf.train.latest_checkpoint(
                           os.path.join(models_dir, model_name))
                saver.restore(sess, ckpt)
        
                # out has dimension batch_size x num_subwords
                # containing per-subword surprisals
                assert BATCH_SIZE == 1
                out = sess.run(output)
                surps = list(out[0])
            window_surprisals.append(surps)
        subword_surprisals = combine_window_surprisals(
                                 window_surprisals, context_size)
    
        # "roll" surprisals together to get per-word surprisal
        word_surprisals = roll_subword_surprisal(
                              subwords, subword_surprisals, words)

        all_words.extend(words)
        all_word_surprisals.extend(word_surprisals)
    
    print("word gpt2surp")
    for i in range(len(all_words)):
        print('{} {}'.format(all_words[i], all_word_surprisals[i]))


if __name__ == '__main__':
    fire.Fire(per_word_surprisal)
