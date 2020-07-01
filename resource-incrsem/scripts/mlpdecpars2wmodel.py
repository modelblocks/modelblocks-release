import sys, configparser, torch, re, os, time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
import numpy as np


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def prepare_data():
    punct = ["-LCB-", "-LRB-", "-RCB-", "-RRB-"]
    wpreds, chars = ([] for _ in range(2))

    for line in sys.stdin.readlines():
        wpred, word = line.rstrip().rsplit(" : ", 1)
        wpreds.append(wpred)

        if word in punct:
            chars.append([word])
        else:
            chars.append(list(word))
    
    flat_chars = [char for sublist in chars for char in sublist]
    wpred_to_ix = {wpred: i for i, wpred in enumerate(sorted(set(wpreds)))}
    char_to_ix = {char: i for i, char in enumerate(sorted(set(flat_chars)))}
    char_to_ix["<E>"] = len(char_to_ix)
    char_to_ix["<S>"] = len(char_to_ix)
    char_to_ix["<P>"] = len(char_to_ix)

    input_ix, target_ix = [], []

    for i in range(len(chars)):
        chars[i] = ["<S>"] + chars[i] + ["<E>"]
        input = chars[i][:-1]
        target = chars[i][1:]
        input_ix.append([char_to_ix[char] for char in input])
        target_ix.append([char_to_ix[char] for char in target])

    input_lens = torch.LongTensor(list(map(len, input_ix)))
    max_len = input_lens.max().item()

    # wpred doesn't change throughout sequence
    wpred_ix = [[wpred_to_ix[wpred]] * max_len for wpred in wpreds]
    wpred_ix = torch.LongTensor(wpred_ix)

    input_ix_padded = torch.zeros(1, dtype=torch.int64)
    input_ix_padded = input_ix_padded.new_full((len(input_ix), max_len), char_to_ix["<P>"])

    for ix, (seq, seqlen) in enumerate(zip(input_ix, input_lens)):
        input_ix_padded[ix, :seqlen] = torch.LongTensor(seq)

    target_ix_padded = torch.zeros(1, dtype=torch.int64)
    target_ix_padded = target_ix_padded.new_full((len(input_ix), max_len), char_to_ix["<P>"])

    for ix, (seq, seqlen) in enumerate(zip(target_ix, input_lens)):
        target_ix_padded[ix, :seqlen] = torch.LongTensor(seq)

    eprint("Number of training examples: {}".format(len(input_ix)))
    eprint("Number of input WPredictors: {}".format(len(wpred_to_ix)))
    eprint("Number of input characters: {}".format(len(char_to_ix)))

    return wpred_to_ix, char_to_ix, wpred_ix, input_ix_padded, target_ix_padded, input_lens


def prepare_data_dev(dev_file, wpred_to_ix, char_to_ix):
    punct = ["-LCB-", "-LRB-", "-RCB-", "-RRB-"]
    dev_wpreds, dev_chars = ([] for _ in range(2))

    with open(dev_file, "r") as f:
        for line in f.readlines():
            wpred, word = line.rstrip().rsplit(" : ", 1)
            if wpred not in wpred_to_ix:
                eprint("Unseen WPredictors {} found in dev file!".format(wpred))
                continue

            dev_wpreds.append(wpred)

            if word not in punct and not all(char in char_to_ix for char in list(word)):
                eprint("Unseen char found in word {} in dev file!".format(word))
                continue

            if word in punct:
                dev_chars.append([word])
            else:
                dev_chars.append(list(word))

    dev_input_ix, dev_target_ix = [], []

    for i in range(len(dev_chars)):
        dev_chars[i] = ["<S>"] + dev_chars[i] + ["<E>"]
        input = dev_chars[i][:-1]
        target = dev_chars[i][1:]
        dev_input_ix.append([char_to_ix[char] for char in input])
        dev_target_ix.append([char_to_ix[char] for char in target])

    dev_input_lens = torch.LongTensor(list(map(len, dev_input_ix)))
    dev_max_len = dev_input_lens.max().item()

    # wpred doesn't change throughout sequence
    dev_wpred_ix = [[wpred_to_ix[wpred]] * dev_max_len for wpred in dev_wpreds]
    dev_wpred_ix = torch.LongTensor(dev_wpred_ix)

    dev_input_ix_padded = torch.zeros(1, dtype=torch.int64)
    dev_input_ix_padded = dev_input_ix_padded.new_full((len(dev_input_ix), dev_max_len), char_to_ix["<P>"])

    for ix, (seq, seqlen) in enumerate(zip(dev_input_ix, dev_input_lens)):
        dev_input_ix_padded[ix, :seqlen] = torch.LongTensor(seq)

    dev_target_ix_padded = torch.zeros(1, dtype=torch.int64)
    dev_target_ix_padded = dev_target_ix_padded.new_full((len(dev_input_ix), dev_max_len), char_to_ix["<P>"])

    for ix, (seq, seqlen) in enumerate(zip(dev_target_ix, dev_input_lens)):
        dev_target_ix_padded[ix, :seqlen] = torch.LongTensor(seq)

    return dev_wpred_ix, dev_input_ix_padded, dev_target_ix_padded, dev_input_lens


class WModel(nn.Module):
    def __init__(self, wpred_vocab_size, char_vocab_size, wpred_size, char_size, hidden_dim, n_layers, output_dim, dropout_prob):
        super(WModel, self).__init__()
        self.wpred_embeds = nn.Embedding(wpred_vocab_size, wpred_size)
        self.char_embeds = nn.Embedding(char_vocab_size, char_size)
        self.rnn = nn.RNN(wpred_size + char_size, hidden_dim, n_layers, batch_first=True, nonlinearity="relu", dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim, bias=True)
        self.relu = F.relu

    def forward(self, wpred_ix, input_ix_padded, input_lens, use_gpu):
        input_ix_embed = self.char_embeds(input_ix_padded)
        wpred_embed = self.wpred_embeds(wpred_ix)
        wpred_input = torch.cat((wpred_embed, input_ix_embed), dim=2)
        packed_wpin = pack_padded_sequence(wpred_input, input_lens.numpy(), batch_first=True, enforce_sorted=False)

        x, _ = self.rnn(packed_wpin)
        x = self.fc(x.data)
        return F.log_softmax(x, dim=1)

    def get_prob(self, kvec_ix, cat_ix, char_ix):
        char_embed = self.char_embeds(char_ix)
        kvec_embed = self.kvec_embeds(kvec_ix)
        cat_embed = self.cat_embeds(cat_ix)
        print(input_ix_embed.shape)
        print(kvec_embed.shape)
        print(cat_embed.shape)
        kcin = torch.cat((kvec_embed.unsqueeze(1), cat_embed.unsqueeze(1), char_embed.unsqueeze(1)), dim=2)
        x, _ = self.rnn(kcin)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        eprint(x.shape)
        return F.log_softmax(x, dim=1)


def train(use_dev, dev_file, use_gpu, wpred_size, char_size, hidden_dim, n_layers, dropout_prob,
          num_epochs, batch_size, learning_rate, weight_decay, l2_reg):
    wpred_to_ix, char_to_ix, wpred_ix, input_ix_padded, target_ix_padded, input_lens = prepare_data()
    model = WModel(len(wpred_to_ix), len(char_to_ix), wpred_size, char_size, hidden_dim, n_layers, len(char_to_ix)-2, dropout_prob)

    if use_gpu >= 0:
        input_ix_padded = input_ix_padded.to("cuda")
        target_ix_padded = target_ix_padded.to("cuda")
        wpred_ix = wpred_ix.to("cuda")
        model = model.cuda()

    if use_dev >= 0:
        dev_wpred_ix, dev_input_ix_padded, dev_target_ix_padded, dev_input_lens = prepare_data_dev(dev_file, wpred_to_ix, char_to_ix)

        if use_gpu >= 0:
            dev_input_ix_padded = dev_input_ix_padded.to("cuda")
            dev_target_ix_padded = dev_target_ix_padded.to("cuda")
            dev_wpred_ix = dev_wpred_ix.to("cuda")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.NLLLoss()

    # training loop
    eprint("Start WModel training...")
    epoch = 0

    while True:
        c0 = time.time()
        model.train()
        epoch += 1
        permutation = torch.randperm(input_lens.shape[0])
        total_train_ex = 0
        total_train_correct = 0
        total_train_loss = 0
        total_dev_loss = 0

        for i in range(0, input_lens.shape[0], batch_size):
            indices = permutation[i:i+batch_size]
            b_wpred_ix, b_input_ix, b_target_ix, b_input_lens = \
                wpred_ix[indices], input_ix_padded[indices], target_ix_padded[indices], input_lens[indices]

            if use_gpu >= 0:
                l2_loss = torch.cuda.FloatTensor([0])
            else:
                l2_loss = torch.FloatTensor([0])

            for param in model.parameters():
                l2_loss += torch.mean(param.pow(2))

            output = model(b_wpred_ix, b_input_ix, b_input_lens, use_gpu)
            packed_target = pack_padded_sequence(b_target_ix, b_input_lens.numpy(), batch_first=True, enforce_sorted=False)
            _, pred = torch.max(output.data, 1)
            total_train_ex += packed_target.data.shape[0]
            train_correct = (pred == packed_target.data).sum().item()
            total_train_correct += train_correct
            nll_loss = criterion(output, packed_target.data)
            loss = nll_loss + l2_reg * l2_loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if use_dev >= 0:
            with torch.no_grad():
                model.eval()
                dev_output = model(dev_wpred_ix, dev_input_ix_padded, dev_input_lens, use_gpu)
                dev_packed_target = pack_padded_sequence(dev_target_ix_padded, dev_input_lens.numpy(), batch_first=True, enforce_sorted=False)
                _, dev_pred = torch.max(dev_output.data, 1)
                dev_correct = (dev_pred == dev_packed_target.data).sum().item()
                dev_loss = criterion(dev_output, dev_packed_target.data)
                total_dev_loss += dev_loss.item()
                dev_acc = 100 * (dev_correct / dev_packed_target.data.shape[0])
        else:
            dev_acc = 0

        eprint("Epoch {:04d} | AvgTrainLoss {:.4f} | TrainAcc {:.4f} | DevLoss {:.4f} | DevAcc {:.4f} | Time {:.4f}".
               format(epoch, total_train_loss / (input_lens.shape[0] // batch_size), 100 * (total_train_correct / total_train_ex),
                      total_dev_loss, dev_acc, time.time() - c0))

        if epoch == num_epochs:
            if use_gpu:
                model.cpu()
            torch.save({"model_state_dict": model.state_dict(), "wpred_to_ix": wpred_to_ix, "char_to_ix": char_to_ix}, "wsj22_wmodel_{}epochs.pt".format(num_epochs))
            break

    return model, wpred_to_ix, char_to_ix


def main(config):
    w_config = config["WModel"]
    torch.manual_seed(w_config.getint("Seed"))
    model, wpred_to_ix, char_to_ix = train(w_config.getint("Dev"), w_config.get("DevFile"), w_config.getint("GPU"),
                                           w_config.getint("WPredSize"), w_config.getint("CharSize"),
                                           w_config.getint("HiddenSize"), w_config.getint("NLayers"),
                                           w_config.getfloat("DropoutProb"), w_config.getint("NEpochs"),
                                           w_config.getint("BatchSize"), w_config.getfloat("LearningRate"),
                                           w_config.getfloat("WeightDecay"), w_config.getfloat("L2Reg"))

    model.eval()

    if w_config.getint("GPU") >= 0:
        wpred_embeds = model.state_dict()["wpred_embeds.weight"].data.cpu().numpy()
        char_embeds = model.state_dict()["char_embeds.weight"].data.cpu().numpy()
        ih_weights = model.state_dict()["rnn.weight_ih_l0"].data.cpu().numpy()
        hh_weights = model.state_dict()["rnn.weight_hh_l0"].data.cpu().numpy()
        ih_bias = model.state_dict()["rnn.bias_ih_l0"].data.cpu().numpy()
        hh_bias = model.state_dict()["rnn.bias_hh_l0"].data.cpu().numpy()
        fc_weights = model.state_dict()["fc.weight"].data.cpu().numpy()
        fc_bias = model.state_dict()["fc.bias"].data.cpu().numpy()
    else:
        wpred_embeds = model.state_dict()["wpred_embeds.weight"].data.numpy()
        char_embeds = model.state_dict()["char_embeds.weight"].data.numpy()
        ih_weights = model.state_dict()["rnn.weight_ih_l0"].data.numpy()
        hh_weights = model.state_dict()["rnn.weight_hh_l0"].data.numpy()
        ih_bias = model.state_dict()["rnn.bias_ih_l0"].data.numpy()
        hh_bias = model.state_dict()["rnn.bias_hh_l0"].data.numpy()
        fc_weights = model.state_dict()["fc.weight"].data.numpy()
        fc_bias = model.state_dict()["fc.bias"].data.numpy()

    print("W I " + ",".join(map(str, ih_weights.flatten('F').tolist())))
    print("W i " + ",".join(map(str, ih_bias.flatten('F').tolist())))
    print("W H " + ",".join(map(str, hh_weights.flatten('F').tolist())))
    print("W h " + ",".join(map(str, hh_bias.flatten('F').tolist())))
    print("W F " + ",".join(map(str, fc_weights.flatten('F').tolist())))
    print("W f " + ",".join(map(str, fc_bias.flatten('F').tolist())))
    print("W P " + ",".join(map(str, wpred_embeds.transpose().flatten('F').tolist())))

    # for wpred, ix in sorted(wpred_to_ix.items()):
    #     print("P " + str(wpred) + " [" + ",".join(map(str, wpred_embeds[ix])) + "]")
    for wpred, ix in sorted(wpred_to_ix.items()):
        print("p " + str(wpred) + " " + str(ix))
    for char, ix in sorted(char_to_ix.items()):
        print("A " + str(char) + " [" + ",".join(map(str, char_embeds[ix])) + "]")
    for char, ix in sorted(char_to_ix.items()):
        print("a " + str(char) + " " + str(ix))


if __name__ == "__main__":
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(sys.argv[1])
    for section in config:
        eprint(section, dict(config[section]))
    main(config)
