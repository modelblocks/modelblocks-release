# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import dictionary_corpus
from utils import repackage_hidden, batchify, get_batch
import numpy as np

parser = argparse.ArgumentParser(description='Generate surprisal at each word')

parser.add_argument('--data', type=str, default="src", help='location of the pretrained model vocabulary')
parser.add_argument('--checkpoint', type=str, help='model checkpoint to use')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--path', type=str, help='path to test file (text)')
args = parser.parse_args()

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def evaluate(data_source, tokens):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    hidden = model.init_hidden(eval_batch_size)

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, seq_len):
            list_len = min(seq_len, len(tokens)-1-i)
            batch_tokens = tokens[i:i+list_len]
            # keep continuous hidden state across all sentences in the input file
            data, targets = get_batch(data_source, i, seq_len)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, vocab_size)
            output_surprisal(output_flat, targets, batch_tokens)
            hidden = repackage_hidden(hidden)


def output_surprisal(output_flat, targets, batch_tokens):
    softmax_probs = F.softmax(output_flat, dim=1)

    log_probs_np = -1 * np.log2(softmax_probs.cpu().numpy())
    targets_np = targets.cpu().numpy()

    for scores, correct_label, token in zip(log_probs_np, targets_np, batch_tokens):
        lstmunk = 0
        if idx2word[correct_label] == "<eos>":
            assert idx2word[correct_label] == token
            continue
        elif idx2word[correct_label] == "<unk>":
            lstmunk = 1
            print(token + " " + str(scores[correct_label]) + " " + str(lstmunk))
        else:
            assert idx2word[correct_label] == token or (idx2word[correct_label] == "(" and token == "-LRB-") or (idx2word[correct_label] == ")" and token == "-RRB-")
            print(idx2word[correct_label] + " " + str(scores[correct_label]) + " " + str(lstmunk))


# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        eprint("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

with open(args.checkpoint, 'rb') as f:
    eprint("Loading model from {}".format(args.checkpoint))
    if args.cuda:
        model = torch.load(f)
    else:
        # to convert model trained on cuda to cpu model
        model = torch.load(f, map_location = lambda storage, loc: storage)

model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()

eval_batch_size = 1
seq_len = 20

dictionary = dictionary_corpus.Dictionary(args.data)
vocab_size = len(dictionary)
idx2word = dictionary.idx2word
eprint("GLSTM vocab size", vocab_size)

id_tensor, tokens = dictionary_corpus.tokenize(dictionary, args.path)
test_data = batchify(id_tensor, eval_batch_size, args.cuda)

eprint("Computing surprisal for target words in {}".format(args.path))
print("word totsurp lstmunk")
# GLSTM cannot make predictions for the first token
print(tokens[0]+" inf 1")
evaluate(test_data, tokens[1:])
