import sys, configparser, torch, re, os, time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
import numpy as np
from collections import Counter


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def prepare_data():
    punct = ["-LCB-", "-LRB-", "-RCB-", "-RRB-"]
    exs, preds, cats, rules, lcats, lchars = ([] for _ in range(6))
    lchar2ekpsk = {}

    for line in sys.stdin.readlines():
        ex, pred, cat, rule, lcat, lchar, word = line.split(" ")
        exs.append(ex)
        preds.append(pred)
        cats.append(cat)
        rules.append(rule)
        # lcat, lchar = lemma.rsplit(":", 1)
        lcats.append(lcat)

        if len(list(lchar)) == 0:
            eprint("invalid lemma character sequence:", ex, pred, cat, rule, lcat, lchar, word)
            continue

        if rule == "%|%" or rule == "%|":
            primcat = "All"
        else:
            primcat = rule.split("|")[1][0]

        if rule == "%|":
            lchar = word.strip()
            if lchar not in punct:
                lchar = lchar.lower()

        if lchar in punct:
            lchars.append([lchar])
        else:
            lchars.append(list(lchar))

        if re.search(":!unk!", pred):
            lchar2ekpsk[("UNK", primcat)] = lchar2ekpsk.get(("UNK", primcat), [])
            lchar2ekpsk[("UNK", primcat)].append((ex, pred, cat, lcat))
        elif re.search(":!num!", pred):
            lchar2ekpsk[("NUM", "All")] = lchar2ekpsk.get(("NUM", "All"), [])
            lchar2ekpsk[("NUM", "All")].append((ex, pred, cat, lcat))
        else:
            lchar2ekpsk[(lchar, primcat)] = lchar2ekpsk.get((lchar, primcat), [])
            lchar2ekpsk[(lchar, primcat)].append((ex, pred, cat, lcat))

    # lchar2ekpsk["UNK"] = set(lchar2ekpsk["UNK"])
    # lchar2ekpsk["NUM"] = set(lchar2ekpsk["NUM"])
    for k in lchar2ekpsk:
        # if k == "UNK" or k == "NUM":
        #     continue
        lchar2ekpsk[k] = set(lchar2ekpsk[k])

    # for k, v in Counter(rules).most_common():
    #     print(k, v)
    # assert 0 == 1

    ex_to_ix = {ex: i for i, ex in enumerate(sorted(set(exs)))}
    pred_to_ix = {pred: i for i, pred in enumerate(sorted(set(preds)))}
    cat_to_ix = {cat: i for i, cat in enumerate(sorted(set(cats)))}
    rule_to_ix = {rule: i for i, rule in enumerate(sorted(set(rules)))}

    lcat_to_ix = {lcat: i for i, lcat in enumerate(sorted(set(lcats)))}

    flat_chars = [char for sublist in lchars for char in sublist]
    char_to_ix = {char: i for i, char in enumerate(sorted(set(flat_chars)))}
    char_to_ix["<E>"] = len(char_to_ix)
    char_to_ix["<S>"] = len(char_to_ix)
    char_to_ix["<P>"] = len(char_to_ix)

    input_ix, target_ix = [], []

    for i in range(len(lchars)):
        char_ix = [char_to_ix[char] for char in lchars[i]]
        # input_ix: <S> c a t
        # target_ix: c a t <E>
        input_ix.append([char_to_ix["<S>"]] + char_ix)
        target_ix.append(char_ix + [char_to_ix["<E>"]])

    input_lens = torch.LongTensor(list(map(len, input_ix)))
    max_len = input_lens.max().item()

    # ex, pred, and cat doesn't change throughout sequence
    ex_ix = [[ex_to_ix[ex]] * max_len for ex in exs]
    ex_ix = torch.LongTensor(ex_ix)
    pred_ix = [[pred_to_ix[pred]] * max_len for pred in preds]
    pred_ix = torch.LongTensor(pred_ix)
    cat_ix = [[cat_to_ix[cat]] * max_len for cat in cats]
    cat_ix = torch.LongTensor(cat_ix)
    lcat_ix = [[lcat_to_ix[lcat]] * max_len for lcat in lcats]
    lcat_ix = torch.LongTensor(lcat_ix)
    rule_ix = [rule_to_ix[rule] for rule in rules]
    rule_ix = torch.LongTensor(rule_ix)

    # embed both <S>, cat and then concatenate
    input_ix_padded = torch.zeros(1, dtype=torch.int64)
    input_ix_padded = input_ix_padded.new_full((len(input_ix), max_len), char_to_ix["<P>"])

    for ix, (seq, seqlen) in enumerate(zip(input_ix, input_lens)):
        input_ix_padded[ix, :seqlen] = torch.LongTensor(seq)

    target_ix_padded = torch.zeros(1, dtype=torch.int64)
    target_ix_padded = target_ix_padded.new_full((len(input_ix), max_len), char_to_ix["<P>"])

    for ix, (seq, seqlen) in enumerate(zip(target_ix, input_lens)):
        target_ix_padded[ix, :seqlen] = torch.LongTensor(seq)

    eprint("Number of training examples: {}".format(len(input_ix)))
    eprint("Number of input extraction operators (e): {}".format(len(ex_to_ix)))
    eprint("Number of input predicates (k): {}".format(len(pred_to_ix)))
    eprint("Number of input syntactic categories (p): {}".format(len(cat_to_ix)))
    eprint("Number of input morph rules: {}".format(len(rule_to_ix)))
    eprint("Number of input lemma syntactic categories: {}".format(len(lcat_to_ix)))
    eprint("Number of input lemma characters: {}".format(len(char_to_ix)))

    # eprint("===== Dimensions =====")
    # eprint("ex_ix:", ex_ix.shape)
    # eprint("pred_ix:", pred_ix.shape)
    # eprint("cat_ix:", cat_ix.shape)
    # eprint("lcat_ix:", lcat_ix.shape)
    # eprint("input_ix_padded:", input_ix_padded.shape)
    # eprint("target_ix_padded:", target_ix_padded.shape)

    return ex_to_ix, pred_to_ix, cat_to_ix, lcat_to_ix, char_to_ix, rule_to_ix, ex_ix, pred_ix, cat_ix, lcat_ix,\
           input_ix_padded, target_ix_padded, input_lens, rule_ix, lchar2ekpsk


def prepare_data_dev(dev_file, ex_to_ix, pred_to_ix, cat_to_ix, lcat_to_ix, char_to_ix, rule_to_ix):
    punct = ["-LCB-", "-LRB-", "-RCB-", "-RRB-"]
    dev_exs, dev_preds, dev_cats, dev_rules, dev_lcats, dev_lchars = ([] for _ in range(6))

    with open(dev_file, "r") as f:
        for line in f.readlines():
            ex, pred, cat, rule, lcat, lchar, word = line.split(" ")
            if ex not in ex_to_ix:
                eprint("Unseen extraction operator {} found in dev file!".format(ex))
                continue
            elif pred not in pred_to_ix:
                eprint("Unseen predicate {} found in dev file!".format(pred))
                continue
            elif cat not in cat_to_ix:
                eprint("Unseen syntactic category {} found in dev file!".format(cat))
                continue
            elif rule not in rule_to_ix:
                eprint("Unseen morph rule {} found in dev file!".format(rule))
                continue

            if rule == "%|":
                lchar = word.strip()
                if lchar not in punct:
                    lchar = lchar.lower()

            # lcat, lchar = lemma.split(":", 1)
            if lcat not in lcat_to_ix:
                eprint("Unseen lemma syntactic category {} found in dev file!".format(lcat))
                continue
            elif lchar not in punct and not all(char in char_to_ix for char in list(lchar)):
                eprint("Unseen character found in lemma {} in dev file!".format(lchar))
                continue

            dev_exs.append(ex)
            dev_preds.append(pred)
            dev_cats.append(cat)
            dev_rules.append(rule)
            dev_lcats.append(lcat)
            if lchar in punct:
                dev_lchars.append([lchar])
            else:
                dev_lchars.append(list(lchar))

    dev_input_ix, dev_target_ix = [], []

    for i in range(len(dev_lchars)):
        char_ix = [char_to_ix[char] for char in dev_lchars[i]]
        # input_ix: <S> c a t
        # target_ix: c a t <E>
        dev_input_ix.append([char_to_ix["<S>"]] + char_ix)
        dev_target_ix.append(char_ix + [char_to_ix["<E>"]])

    dev_input_lens = torch.LongTensor(list(map(len, dev_input_ix)))
    dev_max_len = dev_input_lens.max().item()

    # ex, pred, and cat doesn't change throughout sequence
    dev_ex_ix = [[ex_to_ix[ex]] * dev_max_len for ex in dev_exs]
    dev_ex_ix = torch.LongTensor(dev_ex_ix)
    dev_pred_ix = [[pred_to_ix[pred]] * dev_max_len for pred in dev_preds]
    dev_pred_ix = torch.LongTensor(dev_pred_ix)
    dev_cat_ix = [[cat_to_ix[cat]] * dev_max_len for cat in dev_cats]
    dev_cat_ix = torch.LongTensor(dev_cat_ix)
    dev_lcat_ix = [[lcat_to_ix[lcat]] * dev_max_len for lcat in dev_lcats]
    dev_lcat_ix = torch.LongTensor(dev_lcat_ix)
    dev_rule_ix = [rule_to_ix[rule] for rule in dev_rules]
    dev_rule_ix = torch.LongTensor(dev_rule_ix)

    dev_input_ix_padded = torch.zeros(1, dtype=torch.int64)
    dev_input_ix_padded = dev_input_ix_padded.new_full((len(dev_input_ix), dev_max_len), char_to_ix["<P>"])

    for ix, (seq, seqlen) in enumerate(zip(dev_input_ix, dev_input_lens)):
        dev_input_ix_padded[ix, :seqlen] = torch.LongTensor(seq)

    dev_target_ix_padded = torch.zeros(1, dtype=torch.int64)
    dev_target_ix_padded = dev_target_ix_padded.new_full((len(dev_input_ix), dev_max_len), char_to_ix["<P>"])

    for ix, (seq, seqlen) in enumerate(zip(dev_target_ix, dev_input_lens)):
        dev_target_ix_padded[ix, :seqlen] = torch.LongTensor(seq)

    return dev_ex_ix, dev_pred_ix, dev_cat_ix, dev_lcat_ix, dev_input_ix_padded, dev_target_ix_padded,\
           dev_input_lens, dev_rule_ix


class XModel(nn.Module):
    def __init__(self, ex_vocab_size, pred_vocab_size, cat_vocab_size, char_vocab_size, ex_size, pred_size, cat_size,
                 char_size, hidden_dim, n_layers, dropout_prob):
        super(XModel, self).__init__()
        self.ex_embeds = nn.Embedding(ex_vocab_size, ex_size)
        self.pred_embeds = nn.Embedding(pred_vocab_size, pred_size)
        self.cat_embeds = nn.Embedding(cat_vocab_size, cat_size)

        # lcat is part of the sequence, and therefore needs to have the same dimension as characters
        # self.lcat_embeds = nn.Embedding(lcat_vocab_size, char_size)
        self.char_embeds = nn.Embedding(char_vocab_size, char_size)
        self.rnn = nn.RNN(ex_size + pred_size + cat_size + char_size, hidden_dim, n_layers, batch_first=True,
                          nonlinearity="relu", dropout=dropout_prob)

        # after first RNN step - predict lemma syntactic category
        # self.fc1 = nn.Linear(hidden_dim, lcat_vocab_size-1, bias=True)
        # rest of RNN steps - predict next character of lemma sequence
        # -2 because <S> and <P> are never predicted
        self.fc = nn.Linear(hidden_dim, char_vocab_size-2, bias=True)
        # self.relu = F.relu

    def forward(self, ex_ix, pred_ix, cat_ix, input_ix_padded, input_lens, use_gpu):
        ex_embed = self.ex_embeds(ex_ix)
        pred_embed = self.pred_embeds(pred_ix)
        cat_embed = self.cat_embeds(cat_ix)
        # lcat_embed = self.lcat_embeds(lcat_ix)
        char_embed = self.char_embeds(input_ix_padded)
        # input_embed = torch.cat((lcat_embed, char_embed), dim=1)
        final_input = torch.cat((ex_embed, pred_embed, cat_embed, char_embed), dim=2)
        packed_input = pack_padded_sequence(final_input, input_lens.numpy(), batch_first=True, enforce_sorted=False)

        x, _ = self.rnn(packed_input)
        x = self.fc(x.data)
        # x1 = x.data[:lcat_ix.shape[0]]
        # x2 = x.data[lcat_ix.shape[0]:]
        # x1 = self.fc1(x1)
        # x2 = self.fc2(x2)
        return F.log_softmax(x, dim=1)

    # def get_prob(self, kvec_ix, cat_ix, char_ix):
    #     char_embed = self.char_embeds(char_ix)
    #     kvec_embed = self.kvec_embeds(kvec_ix)
    #     cat_embed = self.cat_embeds(cat_ix)
    #     print(input_ix_embed.shape)
    #     print(kvec_embed.shape)
    #     print(cat_embed.shape)
    #     kcin = torch.cat((kvec_embed.unsqueeze(1), cat_embed.unsqueeze(1), char_embed.unsqueeze(1)), dim=2)
    #     x, _ = self.rnn(kcin)
    #     x = self.fc(x)
    #     x = F.log_softmax(x, dim=1)
    #     eprint(x.shape)
    #     return F.log_softmax(x, dim=1)


class MModel(nn.Module):
    def __init__(self, ex_vocab_size, cat_vocab_size, lcat_vocab_size, char_vocab_size, ex_size, cat_size, lcat_size,
                 char_size, hidden_dim, n_layers, dropout_prob, rule_vocab_size):
        super(MModel, self).__init__()
        self.ex_embeds = nn.Embedding(ex_vocab_size, ex_size)
        self.cat_embeds = nn.Embedding(cat_vocab_size, cat_size)
        self.lcat_embeds = nn.Embedding(lcat_vocab_size, lcat_size)
        self.char_embeds = nn.Embedding(char_vocab_size, char_size)
        self.rnn = nn.RNN(ex_size + cat_size + lcat_size + char_size, hidden_dim, n_layers, batch_first=True,
                          nonlinearity="relu", dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, rule_vocab_size, bias=True)

    def forward(self, ex_ix, cat_ix, lcat_ix, input_ix_padded, input_lens, use_gpu):
        # sequence without <S>, sequence length = n-1
        ex_embed = self.ex_embeds(ex_ix[:,1:])
        cat_embed = self.cat_embeds(cat_ix[:,1:])
        lcat_embed = self.lcat_embeds(lcat_ix[:,1:])
        char_embed = self.char_embeds(input_ix_padded[:,1:])
        # input_embed = torch.cat((lcat_embed, char_embed), dim=1)
        final_input = torch.cat((ex_embed, cat_embed, lcat_embed, char_embed), dim=2)
        packed_input = pack_padded_sequence(final_input, input_lens.numpy()-1, batch_first=True, enforce_sorted=False)

        _, h = self.rnn(packed_input)
        h = self.fc(h[0])
        return F.log_softmax(h, dim=1)


def train(use_dev, dev_file, use_gpu,
          ex_size_x, pred_size_x, cat_size_x, char_size_x, hidden_dim_x, n_layers_x, dropout_prob_x,
          num_epochs_x, batch_size_x, learning_rate_x, weight_decay_x, l2_reg_x,
          ex_size_m, cat_size_m, lcat_size_m, char_size_m, hidden_dim_m, n_layers_m, dropout_prob_m,
          num_epochs_m, batch_size_m, learning_rate_m, weight_decay_m, l2_reg_m):

    ex_to_ix, pred_to_ix, cat_to_ix, lcat_to_ix, char_to_ix, rule_to_ix, ex_ix, pred_ix, cat_ix, lcat_ix, \
    input_ix_padded, target_ix_padded, input_lens, rule_ix, lchar2ekpsk = prepare_data()

    modX = XModel(len(ex_to_ix), len(pred_to_ix), len(cat_to_ix), len(char_to_ix), ex_size_x, pred_size_x, cat_size_x,
                  char_size_x, hidden_dim_x, n_layers_x, dropout_prob_x)
    modM = MModel(len(ex_to_ix), len(cat_to_ix), len(lcat_to_ix), len(char_to_ix), ex_size_m, cat_size_m, lcat_size_m,
                  char_size_m, hidden_dim_m, n_layers_m, dropout_prob_m, len(rule_to_ix))

    if use_gpu >= 0:
        input_ix_padded = input_ix_padded.to("cuda")
        target_ix_padded = target_ix_padded.to("cuda")
        ex_ix = ex_ix.to("cuda")
        pred_ix = pred_ix.to("cuda")
        cat_ix = cat_ix.to("cuda")
        lcat_ix = lcat_ix.to("cuda")
        rule_ix = rule_ix.to("cuda")
        modX = modX.cuda()
        modM = modM.cuda()

    if use_dev >= 0:
        dev_ex_ix, dev_pred_ix, dev_cat_ix, dev_lcat_ix, dev_input_ix_padded, dev_target_ix_padded, dev_input_lens,\
        dev_rule_ix = prepare_data_dev(dev_file, ex_to_ix, pred_to_ix, cat_to_ix, lcat_to_ix, char_to_ix, rule_to_ix)

        if use_gpu >= 0:
            dev_input_ix_padded = dev_input_ix_padded.to("cuda")
            dev_target_ix_padded = dev_target_ix_padded.to("cuda")
            dev_ex_ix = dev_ex_ix.to("cuda")
            dev_pred_ix = dev_pred_ix.to("cuda")
            dev_cat_ix = dev_cat_ix.to("cuda")
            dev_lcat_ix = dev_lcat_ix.to("cuda")
            dev_rule_ix = dev_rule_ix.to("cuda")

    modX_optim = optim.Adam(modX.parameters(), lr=learning_rate_x, weight_decay=weight_decay_x)
    modM_optim = optim.Adam(modM.parameters(), lr=learning_rate_m, weight_decay=weight_decay_m)
    criterion = nn.NLLLoss()

    # XModel training loop
    eprint("Start XModel training...")
    epoch = 0

    while True:
        c0 = time.time()
        modX.train()
        epoch += 1
        perm = torch.randperm(input_lens.shape[0])
        # total_cat_ex = 0
        # total_cat_correct = 0
        total_seq_ex = 0
        total_seq_correct = 0
        total_train_loss = 0
        total_dev_loss = 0

        for i in range(0, input_lens.shape[0], batch_size_x):
            idx = perm[i:i+batch_size_x]
            b_ex_ix, b_pred_ix, b_cat_ix, b_input_ix, b_target_ix, b_input_lens = \
                ex_ix[idx], pred_ix[idx], cat_ix[idx], input_ix_padded[idx], target_ix_padded[idx], input_lens[idx]

            if use_gpu >= 0:
                l2_loss = torch.cuda.FloatTensor([0])
            else:
                l2_loss = torch.FloatTensor([0])

            for param in modX.parameters():
                l2_loss += torch.mean(param.pow(2))

            lemma_seq = modX(b_ex_ix, b_pred_ix, b_cat_ix, b_input_ix, b_input_lens, use_gpu)
            packed_target = pack_padded_sequence(b_target_ix, b_input_lens.numpy(), batch_first=True, enforce_sorted=False)
            # _, cat_pred = torch.max(lemma_cat, 1)
            _, seq_pred = torch.max(lemma_seq, 1)

            # total_cat_ex += b_lcat_ix.shape[0]
            # cat_correct = (cat_pred == b_lcat_ix[:,1]).sum().item()
            # total_cat_correct += cat_correct

            total_seq_ex += packed_target.data.shape[0]
            seq_correct = (seq_pred == packed_target.data).sum().item()
            total_seq_correct += seq_correct

            nll_loss = criterion(lemma_seq, packed_target.data)
            loss = nll_loss + l2_reg_x * l2_loss
            total_train_loss += loss.item()
            loss.backward()

            # for name, param in modL.named_parameters():
            #     if param.requires_grad:
            #         eprint(name, param.grad)
            # for key in modL.state_dict():
            #     if "weight" in key:
            #         print(key, modL.state_dict()[key])

            # eprint(modL.state_dict())
            modX_optim.step()
            modX_optim.zero_grad()

        if use_dev >= 0:
            with torch.no_grad():
                modX.eval()
                dev_lemma_seq = modX(dev_ex_ix, dev_pred_ix, dev_cat_ix, dev_input_ix_padded, dev_input_lens, use_gpu)
                dev_packed_target = pack_padded_sequence(dev_target_ix_padded, dev_input_lens.numpy(), batch_first=True, enforce_sorted=False)
                # _, dev_cat_pred = torch.max(dev_lemma_cat, 1)
                _, dev_seq_pred = torch.max(dev_lemma_seq, 1)

                # dev_cat_correct = (dev_cat_pred == dev_lcat_ix[:,1]).sum().item()
                # dev_cat_acc = 100 * (dev_cat_correct / dev_lcat_ix.shape[0])

                dev_seq_correct = (dev_seq_pred == dev_packed_target.data).sum().item()
                dev_seq_acc = 100 * (dev_seq_correct / dev_packed_target.data.shape[0])

                dev_loss = criterion(dev_lemma_seq, dev_packed_target.data)
                total_dev_loss += dev_loss.item()
        else:
            # dev_cat_acc = 0
            dev_seq_acc = 0

        eprint("Epoch {:04d} | AvgTrainLoss {:.4f} | TrainSeqAcc {:.4f} | DevLoss {:.4f} | DevSeqAcc {:.4f} | Time {:.4f}".
               format(epoch, total_train_loss / (input_lens.shape[0] // batch_size_x), 100 * (total_seq_correct / total_seq_ex),
                      total_dev_loss, dev_seq_acc, time.time() - c0))

        if epoch == num_epochs_x:
            # if use_gpu:
            #     model.cpu()
            # torch.save({"model_state_dict": model.state_dict(), "wpred_to_ix": wpred_to_ix, "char_to_ix": char_to_ix}, "wsj22_wmodel_{}epochs.pt".format(num_epochs))
            break

    # MModel training loop
    eprint("Start MModel training...")
    epoch = 0

    while True:
        c0 = time.time()
        modM.train()
        epoch += 1
        perm = torch.randperm(input_lens.shape[0])
        total_ex = 0
        total_correct = 0
        total_train_loss = 0
        total_dev_loss = 0

        for i in range(0, input_lens.shape[0], batch_size_m):
            idx = perm[i:i + batch_size_m]
            b_ex_ix, b_cat_ix, b_lcat_ix, b_input_ix, b_rule_ix, b_input_lens = \
                ex_ix[idx], cat_ix[idx], lcat_ix[idx], input_ix_padded[idx], rule_ix[idx], \
                input_lens[idx]

            if use_gpu >= 0:
                l2_loss = torch.cuda.FloatTensor([0])
            else:
                l2_loss = torch.FloatTensor([0])

            for param in modM.parameters():
                l2_loss += torch.mean(param.pow(2))

            rule = modM(b_ex_ix, b_cat_ix, b_lcat_ix, b_input_ix, b_input_lens, use_gpu)
            _, rule_pred = torch.max(rule, 1)

            total_ex += b_rule_ix.shape[0]
            correct = (rule_pred == b_rule_ix).sum().item()
            total_correct += correct

            # eprint("cat loss", criterion(lemma_cat, b_lcat_ix[:,1]).item())
            # eprint("seq loss", criterion(lemma_seq, packed_target.data).item())
            nll_loss = criterion(rule, b_rule_ix)
            loss = nll_loss + l2_reg_m * l2_loss
            total_train_loss += loss.item()
            loss.backward()
            modM_optim.step()
            modM_optim.zero_grad()

        if use_dev >= 0:
            with torch.no_grad():
                modM.eval()
                dev_rule = modM(dev_ex_ix, dev_cat_ix, dev_lcat_ix, dev_input_ix_padded, dev_input_lens, use_gpu)
                _, dev_rule_pred = torch.max(dev_rule, 1)

                dev_correct = (dev_rule_pred == dev_rule_ix).sum().item()
                dev_acc = 100 * (dev_correct / dev_rule_ix.shape[0])

                dev_loss = criterion(dev_rule, dev_rule_ix)
                total_dev_loss += dev_loss.item()
        else:
            dev_acc = 0

        eprint(
            "Epoch {:04d} | AvgTrainLoss {:.4f} | TrainAcc {:.4f} | DevLoss {:.4f} | DevAcc {:.4f} | Time {:.4f}".
            format(epoch, total_train_loss / (input_lens.shape[0] // batch_size_m), 100 * (total_correct / total_ex),
                   total_dev_loss, dev_acc, time.time() - c0))

        if epoch == num_epochs_m:
            # if use_gpu:
            #     model.cpu()
            # torch.save({"model_state_dict": model.state_dict(), "wpred_to_ix": wpred_to_ix, "char_to_ix": char_to_ix}, "wsj22_wmodel_{}epochs.pt".format(num_epochs))
            break

    return ex_to_ix, pred_to_ix, cat_to_ix, lcat_to_ix, char_to_ix, rule_to_ix, modX, modM, lchar2ekpsk


def main(config):
    w_config = config["WModel"]
    x_config = config["XModel"]
    m_config = config["MModel"]
    torch.manual_seed(w_config.getint("Seed"))
    ex_to_ix, pred_to_ix, cat_to_ix, lcat_to_ix, char_to_ix, rule_to_ix, modX, modM, lchar2ekpsk =\
        train(w_config.getint("Dev"), w_config.get("DevFile"), w_config.getint("GPU"),
        x_config.getint("ExSize"), x_config.getint("PredSize"), x_config.getint("CatSize"), x_config.getint("CharSize"),
        x_config.getint("HiddenSize"), x_config.getint("NLayers"), x_config.getfloat("DropoutProb"), x_config.getint("NEpochs"),
        x_config.getint("BatchSize"), x_config.getfloat("LearningRate"), x_config.getfloat("WeightDecay"), x_config.getfloat("L2Reg"),
        m_config.getint("ExSize"), m_config.getint("CatSize"), m_config.getint("LCatSize"), m_config.getint("CharSize"),
        m_config.getint("HiddenSize"), m_config.getint("NLayers"), m_config.getfloat("DropoutProb"), m_config.getint("NEpochs"),
        m_config.getint("BatchSize"), m_config.getfloat("LearningRate"), m_config.getfloat("WeightDecay"), m_config.getfloat("L2Reg"))

    modX.eval()
    modM.eval()

    if w_config.getint("GPU") >= 0:
        # XModel parameters
        ex_embeds_x = modX.state_dict()["ex_embeds.weight"].data.cpu().numpy()
        pred_embeds_x = modX.state_dict()["pred_embeds.weight"].data.cpu().numpy()
        cat_embeds_x = modX.state_dict()["cat_embeds.weight"].data.cpu().numpy()
        char_embeds_x = modX.state_dict()["char_embeds.weight"].data.cpu().numpy()
        ih_weights_x = modX.state_dict()["rnn.weight_ih_l0"].data.cpu().numpy()
        hh_weights_x = modX.state_dict()["rnn.weight_hh_l0"].data.cpu().numpy()
        ih_bias_x = modX.state_dict()["rnn.bias_ih_l0"].data.cpu().numpy()
        hh_bias_x = modX.state_dict()["rnn.bias_hh_l0"].data.cpu().numpy()
        fc_weights_x = modX.state_dict()["fc.weight"].data.cpu().numpy()
        fc_bias_x = modX.state_dict()["fc.bias"].data.cpu().numpy()
        # MModel parameters
        ex_embeds_m = modM.state_dict()["ex_embeds.weight"].data.cpu().numpy()
        cat_embeds_m = modM.state_dict()["cat_embeds.weight"].data.cpu().numpy()
        lcat_embeds_m = modM.state_dict()["lcat_embeds.weight"].data.cpu().numpy()
        char_embeds_m = modM.state_dict()["char_embeds.weight"].data.cpu().numpy()
        ih_weights_m = modM.state_dict()["rnn.weight_ih_l0"].data.cpu().numpy()
        hh_weights_m = modM.state_dict()["rnn.weight_hh_l0"].data.cpu().numpy()
        ih_bias_m = modM.state_dict()["rnn.bias_ih_l0"].data.cpu().numpy()
        hh_bias_m = modM.state_dict()["rnn.bias_hh_l0"].data.cpu().numpy()
        fc_weights_m = modM.state_dict()["fc.weight"].data.cpu().numpy()
        fc_bias_m = modM.state_dict()["fc.bias"].data.cpu().numpy()
    else:
        # XModel parameters
        ex_embeds_x = modX.state_dict()["ex_embeds.weight"].data.numpy()
        pred_embeds_x = modX.state_dict()["pred_embeds.weight"].data.numpy()
        cat_embeds_x = modX.state_dict()["cat_embeds.weight"].data.numpy()
        char_embeds_x = modX.state_dict()["char_embeds.weight"].data.numpy()
        ih_weights_x = modX.state_dict()["rnn.weight_ih_l0"].data.numpy()
        hh_weights_x = modX.state_dict()["rnn.weight_hh_l0"].data.numpy()
        ih_bias_x = modX.state_dict()["rnn.bias_ih_l0"].data.numpy()
        hh_bias_x = modX.state_dict()["rnn.bias_hh_l0"].data.numpy()
        fc_weights_x = modX.state_dict()["fc.weight"].data.numpy()
        fc_bias_x = modX.state_dict()["fc.bias"].data.numpy()
        # MModel parameters
        ex_embeds_m = modM.state_dict()["ex_embeds.weight"].data.numpy()
        cat_embeds_m = modM.state_dict()["cat_embeds.weight"].data.numpy()
        lcat_embeds_m = modM.state_dict()["lcat_embeds.weight"].data.numpy()
        char_embeds_m = modM.state_dict()["char_embeds.weight"].data.numpy()
        ih_weights_m = modM.state_dict()["rnn.weight_ih_l0"].data.numpy()
        hh_weights_m = modM.state_dict()["rnn.weight_hh_l0"].data.numpy()
        ih_bias_m = modM.state_dict()["rnn.bias_ih_l0"].data.numpy()
        hh_bias_m = modM.state_dict()["rnn.bias_hh_l0"].data.numpy()
        fc_weights_m = modM.state_dict()["fc.weight"].data.numpy()
        fc_bias_m = modM.state_dict()["fc.bias"].data.numpy()

    # XModel parameters
    print("W X I " + ",".join(map(str, ih_weights_x.flatten('F').tolist())))
    print("W X i " + ",".join(map(str, ih_bias_x.flatten('F').tolist())))
    print("W X H " + ",".join(map(str, hh_weights_x.flatten('F').tolist())))
    print("W X h " + ",".join(map(str, hh_bias_x.flatten('F').tolist())))
    print("W X F " + ",".join(map(str, fc_weights_x.flatten('F').tolist())))
    print("W X f " + ",".join(map(str, fc_bias_x.flatten('F').tolist())))
    # MModel parameters
    print("W M I " + ",".join(map(str, ih_weights_m.flatten('F').tolist())))
    print("W M i " + ",".join(map(str, ih_bias_m.flatten('F').tolist())))
    print("W M H " + ",".join(map(str, hh_weights_m.flatten('F').tolist())))
    print("W M h " + ",".join(map(str, hh_bias_m.flatten('F').tolist())))
    print("W M F " + ",".join(map(str, fc_weights_m.flatten('F').tolist())))
    print("W M f " + ",".join(map(str, fc_bias_m.flatten('F').tolist())))

    for ex, ix in sorted(ex_to_ix.items()):
        print("E X " + str(ex) + " [" + ",".join(map(str, ex_embeds_x[ix])) + "]")
        print("E M " + str(ex) + " [" + ",".join(map(str, ex_embeds_m[ix])) + "]")
        # print("e " + str(ex) + " " + str(ix))
    for pred, ix in sorted(pred_to_ix.items()):
        print("K " + str(pred) + " [" + ",".join(map(str, pred_embeds_x[ix])) + "]")
        # print("k " + str(pred) + " " + str(ix))
    for cat, ix in sorted(cat_to_ix.items()):
        print("P X " + str(cat) + " [" + ",".join(map(str, cat_embeds_x[ix])) + "]")
        print("P M " + str(cat) + " [" + ",".join(map(str, cat_embeds_m[ix])) + "]")
        # print("p " + str(cat) + " " + str(ix))
    for lcat, ix in sorted(lcat_to_ix.items()):
        print("L " + str(lcat) + " [" + ",".join(map(str, lcat_embeds_m[ix])) + "]")
        # print("l " + str(lcat) + " " + str(ix))
    for char, ix in sorted(char_to_ix.items()):
        print("C X " + str(char) + " [" + ",".join(map(str, char_embeds_x[ix])) + "]")
        print("C M " + str(char) + " [" + ",".join(map(str, char_embeds_m[ix])) + "]")
        print("C I " + str(char) + " " + str(ix))
    for rule, ix in sorted(rule_to_ix.items()):
        print("R " + str(rule) + " " + str(ix))
    for k in lchar2ekpsk:
        print("X " + k[0] + " " + k[1], end=" [")
        for v in list(lchar2ekpsk[k])[:-1]:
            print(v[0], v[1], v[2], sep="|", end=" ")
        print(list(lchar2ekpsk[k])[-1][0], list(lchar2ekpsk[k])[-1][1], list(lchar2ekpsk[k])[-1][2], sep="/", end="]\n")
    for k in lchar2ekpsk:
        print("M " + k[0] + " " + k[1], end=" [")
        for v in list(lchar2ekpsk[k])[:-1]:
            print(v[0], v[2], v[3], sep="|", end=" ")
        print(list(lchar2ekpsk[k])[-1][0], list(lchar2ekpsk[k])[-1][2], list(lchar2ekpsk[k])[-1][3], sep="/", end="]\n")


if __name__ == "__main__":
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(sys.argv[1])
    for section in config:
        eprint(section, dict(config[section]))
    main(config)
