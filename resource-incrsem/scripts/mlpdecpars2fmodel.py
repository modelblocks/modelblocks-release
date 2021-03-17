import sys, configparser, torch, re, os, time, random, math
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
import numpy as np
from scipy.sparse import csr_matrix


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def random_seed(seed_value, use_cuda):
    torch.manual_seed(seed_value) # cpu vars
    random.seed(seed_value) # Python
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars


# returns list of kvecs for each word
def get_first_kvec(hvec):
    match = re.findall(r"^\[(.*?)\]", hvec)
    return match[0].split(",")


# creates sparse Torch tensor for hvec indexing
def get_hvec_ix_tensor(hv_list, hv_to_ix):
    row, col, top = ([] for _ in range(3))
    for i, sublist in enumerate(hv_list):
        top_count = 0
        for hvec in sublist:
            if hvec == "Top":
                top_count += 1
            elif hvec == "" or hvec == "Bot" or hvec not in hv_to_ix:
                continue
            else:
                row.append(i)
                col.append(hv_to_ix[hvec])
        top.append([top_count])
    top = torch.FloatTensor(top)
    mat = csr_matrix((np.ones(len(row), dtype=np.int32), (row, col)), shape=(len(hv_list), len(hv_to_ix)))
    mat = mat.tocoo()
    mat = torch.sparse.FloatTensor(torch.LongTensor([mat.row.tolist(), mat.col.tolist()]),
                                   torch.FloatTensor(mat.data.astype(np.float32)), torch.Size(mat.shape))
    return mat, top


def prepare_data():
    data = [line.strip() for line in sys.stdin]
    depth, catbase, hvbase, hvfiller, fdecs, wordseq = ([] for _ in range(6))
    sent_depth, sent_catbase, sent_hvbase, sent_hvfiller, sent_fdecs, sent_wordseq = ([] for _ in range(6))
    flat_catbase, flat_hvbase, flat_hvfiller, flat_fdecs, flat_wordseq = ([] for _ in range(5))
    for line in data[1:]:
        if line.startswith("TREE") and "FAIL" in line:
            continue
        elif line.startswith("F"):
            _, d, cb, hvb, hvf, fd = line.split(" ")
            sent_depth.append(int(d))
            sent_catbase.append(cb)
            flat_catbase.append(cb)
            sent_hvbase.append(get_first_kvec(hvb))
            flat_hvbase += get_first_kvec(hvb)
            sent_hvfiller.append(get_first_kvec(hvf))
            flat_hvfiller += get_first_kvec(hvf)
            sent_fdecs.append(fd)
            flat_fdecs.append(fd)
        elif line.startswith("W"):
            _, _, _, _, _, _, _, word = line.split(" ")
            sent_wordseq.append(word.lower())
            flat_wordseq.append(word.lower())
        elif line.startswith("TREE"):
            depth.append(sent_depth)
            catbase.append(sent_catbase)
            hvbase.append(sent_hvbase)
            hvfiller.append(sent_hvfiller)
            fdecs.append(sent_fdecs)
            wordseq.append(sent_wordseq)
            # re-initialize sentence-level lists
            sent_depth, sent_catbase, sent_hvbase, sent_hvfiller, sent_fdecs, sent_wordseq = ([] for _ in range(6))
    depth.append(sent_depth)
    catbase.append(sent_catbase)
    hvbase.append(sent_hvbase)
    hvfiller.append(sent_hvfiller)
    fdecs.append(sent_fdecs)
    wordseq.append(sent_wordseq)
    eprint("Training file processing complete")

    # Mapping from category & HVec to index
    catb_to_ix = {cat: i for i, cat in enumerate(sorted(set(flat_catbase)))}
    fdecs_to_ix = {fdecs: i for i, fdecs in enumerate(sorted(set(flat_fdecs)))}
    hvecb_to_ix = {hvec: i for i, hvec in enumerate(sorted(set([hvec for hvec in flat_hvbase if hvec not in ["", "Bot", "Top"]])))}
    hvecf_to_ix = {hvec: i for i, hvec in enumerate(sorted(set([hvec for hvec in flat_hvfiller if hvec not in ["", "Bot", "Top"]])))}
    word_ctr = Counter(flat_wordseq)
    filtered_word_ctr = Counter({k: c for k, c in word_ctr.items() if c >= 2})
    word_to_ix = {word[0]: i for i, word in enumerate(filtered_word_ctr.items())}
    word_to_ix["<UNK>"] = len(word_to_ix)
    word_to_ix["<PAD>"] = len(word_to_ix)

    eprint("Number of input base CVecs: {}".format(len(catb_to_ix)))
    eprint("Number of input base KVecs: {}".format(len(hvecb_to_ix)))
    eprint("Number of input filler KVecs: {}".format(len(hvecf_to_ix)))
    eprint("Number of words in vocabulary: {}".format(len(word_to_ix)-2))
    eprint("Number of output F categories: {}".format(len(fdecs_to_ix)))

    return (depth, catbase, hvbase, hvfiller, fdecs, wordseq, catb_to_ix, hvecb_to_ix, hvecf_to_ix, fdecs_to_ix, word_to_ix)


def prepare_data_dev(dev_decpars_file, catb_to_ix, fdecs_to_ix):
    with open(dev_decpars_file, "r") as f:
        data = f.readlines()
        data = [line.strip() for line in data]
    curr_fid = 0
    depth, catbase, hvbase, hvfiller, fdecs, wordseq, fid = ([] for _ in range(7))
    sent_depth, sent_catbase, sent_hvbase, sent_hvfiller, sent_fdecs, sent_wordseq, sent_fid = ([] for _ in range(7))
    for line in data[1:]:
        if line.startswith("TREE") and "FAIL" in line:
            continue
        elif line.startswith("F"):
            _, d, cb, hvb, hvf, fd = line.split(" ")
            if cb not in catb_to_ix or fd not in fdecs_to_ix:
                eprint("Unseen syntactic category or F decision found in dev file ({}, {})!".format(cb, fd))
                curr_fid += 1
                continue
            else:
                sent_depth.append(int(d))
                sent_catbase.append(cb)
                sent_hvbase.append(get_first_kvec(hvb))
                sent_hvfiller.append(get_first_kvec(hvf))
                sent_fdecs.append(fd)
                sent_fid.append(curr_fid)
                curr_fid += 1
        elif line.startswith("W"):
            _, _, _, _, _, _, _, word = line.split(" ")
            sent_wordseq.append(word.lower())
        elif line.startswith("TREE"):
            depth.append(sent_depth)
            catbase.append(sent_catbase)
            hvbase.append(sent_hvbase)
            hvfiller.append(sent_hvfiller)
            fdecs.append(sent_fdecs)
            wordseq.append(sent_wordseq)
            fid.append(sent_fid)
            # re-initialize sentence-level lists and index
            curr_fid = 0
            sent_depth, sent_catbase, sent_hvbase, sent_hvfiller, sent_fdecs, sent_wordseq, sent_fid = ([] for _ in range(7))
    depth.append(sent_depth)
    catbase.append(sent_catbase)
    hvbase.append(sent_hvbase)
    hvfiller.append(sent_hvfiller)
    fdecs.append(sent_fdecs)
    wordseq.append(sent_wordseq)
    fid.append(sent_fid)
    eprint("Dev file processing complete")

    return (depth, catbase, hvbase, hvfiller, fdecs, wordseq, fid)


def create_one_batch(training_data, catb_to_ix, hvecb_to_ix, hvecf_to_ix, fdecs_to_ix, word_to_ix):
    depth, catbase, hvbase, hvfiller, fdecs, wordseq = training_data
    depth_ix = [d for sublist in depth for d in sublist]
    depth_ix = F.one_hot(torch.LongTensor(depth_ix), 7).float()
    catb_ix = [catb_to_ix[cb] for sublist in catbase for cb in sublist]
    catb_ix = torch.LongTensor(catb_ix)
    flat_hvb = [hvb for sublist in hvbase for hvb in sublist]
    flat_hvf = [hvf for sublist in hvfiller for hvf in sublist]
    hvb_mat, hvb_top = get_hvec_ix_tensor(flat_hvb, hvecb_to_ix)
    hvf_mat, hvf_top = get_hvec_ix_tensor(flat_hvf, hvecf_to_ix)
    fdec_ix = [fdecs_to_ix[fd] for sublist in fdecs for fd in sublist]
    fdec_ix = torch.LongTensor(fdec_ix)
    wordseq_ix = [[word_to_ix[word] if word in word_to_ix else word_to_ix["<UNK>"] for word in sublist] for sublist in wordseq]
    wordseq_lens = torch.LongTensor(list(map(len, wordseq_ix)))
    wordseq_max_len = wordseq_lens.max().item()
    wordseq_ix_padded = torch.zeros(1, dtype=torch.int64)
    wordseq_ix_padded = wordseq_ix_padded.new_full((len(wordseq_ix), wordseq_max_len), word_to_ix["<PAD>"])

    for ix, (seq, seqlen) in enumerate(zip(wordseq_ix, wordseq_lens)):
        wordseq_ix_padded[ix, :seqlen] = torch.LongTensor(seq)

    return depth_ix, catb_ix, hvb_mat, hvb_top, hvf_mat, hvf_top, fdec_ix, wordseq_ix_padded, wordseq_lens


class FModel(nn.Module):
    def __init__(self,
        device,
        catb_vocab_size,
        syn_dim,
        hvecb_vocab_size,
        hvecf_vocab_size,
        sem_dim,
        word_vocab_size,
        word_dim,
        lstm_hidden_dim,
        lstm_num_layers,
        lstm_dropout,
        ff_input_dropout,
        ff_hidden_dim,
        ff_hidden_dropout,
        ff_output_dim):

        super(FModel, self).__init__()
        self.device = device
        self.catb_embeds = nn.Embedding(catb_vocab_size, syn_dim)
        self.hvecb_embeds = nn.Embedding(hvecb_vocab_size, sem_dim)
        self.hvecf_embeds = nn.Embedding(hvecf_vocab_size, sem_dim)
        self.word_embeds = nn.Embedding(word_vocab_size, word_dim)
        self.lstm = nn.LSTM(word_dim, lstm_hidden_dim, lstm_num_layers, batch_first=True, dropout=lstm_dropout)
        self.dropout_1 = nn.Dropout(ff_input_dropout)
        self.fc1 = nn.Linear(7 + syn_dim + 2*sem_dim + lstm_hidden_dim, ff_hidden_dim, bias=True)
        self.dropout_2 = nn.Dropout(ff_hidden_dropout)
        self.relu = F.relu
        self.fc2 = nn.Linear(ff_hidden_dim, ff_output_dim, bias=True)

    def forward(self,
        d_onehot,
        catb_ix,
        hvb_mat,
        hvf_mat,
        hvb_top,
        hvf_top,
        wordseq_ix_padded,
        wordseq_lens,
        ablate_syn,
        ablate_sem,
        ablate_word,
        dev_mode,
        valid_ids=[]):

        if ablate_syn:
            catb_embed = torch.zeros([len(catb_ix), self.syn_size], dtype=torch.float, device=self.device)
        else:
            catb_embed = self.catb_embeds(catb_ix)

        if ablate_sem:
            hvb_embed = torch.zeros([hvb_top.shape[0], self.sem_size], dtype=torch.float, device=self.device) + hvb_top
            hvf_embed = torch.zeros([hvb_top.shape[0], self.sem_size], dtype=torch.float, device=self.device) + hvf_top
        else:
            hvb_embed = torch.sparse.mm(hvb_mat, self.hvecb_embeds.weight) + hvb_top
            hvf_embed = torch.sparse.mm(hvf_mat, self.hvecf_embeds.weight) + hvf_top

        word_embed = self.word_embeds(wordseq_ix_padded)
        packed_word_embed = pack_padded_sequence(word_embed, wordseq_lens.numpy(), batch_first=True, enforce_sorted=False)
        packed_output, (_, _) = self.lstm(packed_word_embed)
        padded_output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        # manual packing to ensure word embeddings line up with other features
        lstm_output_tensors = []

        if dev_mode:
            for i in range(len(valid_ids)):
                for j in valid_ids[i]:
                    lstm_output_tensors.append(padded_output[i][j].view(1, -1))
        else:
            for i in range(len(input_sizes)):
                lstm_output_tensors.append(padded_output[i][:int(input_sizes[i])])

        lstm_output = torch.cat(lstm_output_tensors, 0)
        # eprint(input_sizes)
        # eprint(catb_embed.shape, hvb_embed.shape, hvf_embed.shape, lstm_output.shape, d_onehot.shape)

        x = torch.cat((catb_embed, hvb_embed, hvf_embed, lstm_output, d_onehot), 1)
        x = self.dropout_1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(config):
    if config.getboolean("GPU") and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    random_seed(config.getint("Seed"), use_cuda=device == "cuda")

    depth, catbase, hvbase, hvfiller, fdecs, wordseq, catb_to_ix, hvecb_to_ix, hvecf_to_ix, fdecs_to_ix, word_to_ix = prepare_data()

    if config.getboolean("Dev"):
        d_depth, d_catbase, d_hvbase, d_hvfiller, d_fdecs, d_wordseq, d_fid = prepare_data_dev(config.get("DevFile"), catb_to_ix, fdecs_to_ix)

    model = FModel(device,
                   len(catb_to_ix),
                   config.getint("SynSize"),
                   len(hvecb_to_ix),
                   len(hvecf_to_ix),
                   config.getint("SemSize"),
                   len(word_to_ix),
                   config.getint("WordSize"),
                   config.getint("LSTMHiddenSize"),
                   config.getint("LSTMNLayers"),
                   config.getfloat("LSTMDropout"),
                   config.getfloat("FFInputDropout"),
                   config.getint("HiddenSize"),
                   config.getfloat("FFHiddenDropout"),
                   len(fdecs_to_ix))

    eprint(str(model))
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    eprint("Topmost model has {} parameters".format(num_params))

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.getfloat("LearningRate"), weight_decay=config.getfloat("WeightDecay"))
    criterion = nn.NLLLoss()

    # training loop
    eprint("Start FModel training...")
    start_time = time.time()

    for epoch in range(config.getint("NEpochs")):
        vars = list(zip(depth, catbase, hvbase, hvfiller, fdecs, wordseq))
        random.shuffle(vars)
        depth, catbase, hvbase, hvfiller, fdecs, wordseq = zip(*vars)
        model.train()
        epoch_data_points = 0
        epoch_correct = 0
        epoch_loss = 0
        dev_data_points = 100
        dev_correct = 0
        dev_loss = 0

        for j in range(0, len(depth), config.getint("BatchSize")):
            b_depth = depth[j:j+config.getint("BatchSize")]
            b_catbase = catbase[j:j+config.getint("BatchSize")]
            b_hvbase = hvbase[j:j+config.getint("BatchSize")]
            b_hvfiller = hvfiller[j:j+config.getint("BatchSize")]
            b_fdecs = fdecs[j:j+config.getint("BatchSize")]
            b_wordseq = wordseq[j:j+config.getint("BatchSize")]
            depth_ix, catb_ix, hvb_mat, hvb_top, hvf_mat, hvf_top, fdec_ix, wordseq_ix_padded, wordseq_lens = \
            create_one_batch((b_depth, b_catbase, b_hvbase, b_hvfiller, b_fdecs, b_wordseq),
                              catb_to_ix, hvecb_to_ix, hvecf_to_ix, fdecs_to_ix, word_to_ix)

            # maybe move these to create_one_batch function
            depth_ix = depth_ix.to(device)
            catb_ix = catb_ix.to(device)
            hvb_mat = hvb_mat.to(device)
            hvb_top = hvb_top.to(device)
            hvf_mat = hvf_mat.to(device)
            hvf_top = hvf_top.to(device)
            fdec_ix = fdec_ix.to(device)
            wordseq_ix_padded = wordseq_ix_padded.to(device)

            optimizer.zero_grad()
            if config.getfloat("L2Reg") > 0:
                l2_loss = torch.cuda.FloatTensor([0]) if device == "cuda" else torch.FloatTensor([0])
                for param in model.parameters():
                    l2_loss += torch.mean(param.pow(2))
            else:
                l2_loss = 0

            output = model(depth_ix, catb_ix, hvb_mat, hvf_mat, hvb_top, hvf_top, wordseq_ix_padded, wordseq_lens,
                           config.getboolean("AblateSyn"), config.getboolean("AblateSem"), config.getboolean("AblateWord"),
                           dev_mode=False)
            _, fdec = torch.max(output.data, 1)

            epoch_data_points += len(fdec)
            batch_correct = (fdec == fdec_ix).sum().item()
            epoch_correct += batch_correct
            nll_loss = criterion(output, fdec_ix)
            loss = nll_loss + config.getfloat("L2Reg") * l2_loss
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        if config.getboolean("Dev"):
            eprint("Entering Dev...")
            model.eval()
            dev_data_points = 0
            dev_correct = 0
            dev_loss = 0
            for j in range(0, len(b_depth), config.getint("DevBatchSize")):
                b_depth = d_depth[j:j+config.getint("DevBatchSize")]
                b_catbase = d_catbase[j:j+config.getint("DevBatchSize")]
                b_hvbase = d_hvbase[j:j+config.getint("DevBatchSize")]
                b_hvfiller = d_hvfiller[j:j+config.getint("DevBatchSize")]
                b_fdecs = d_fdecs[j:j+config.getint("DevBatchSize")]
                b_wordseq = d_wordseq[j:j+config.getint("DevBatchSize")]
                b_fid = d_fid[j:j+config.getint("DevBatchSize")]

                depth_ix, catb_ix, hvb_mat, hvb_top, hvf_mat, hvf_top, fdec_ix, wordseq_ix_padded, wordseq_lens = create_one_batch(
                    (b_depth, b_catbase, b_hvbase, b_hvfiller, b_fdecs, b_wordseq), catb_to_ix, hvecb_to_ix,
                    hvecf_to_ix, fdecs_to_ix, word_to_ix)
                depth_ix = depth_ix.to(device)
                catb_ix = catb_ix.to(device)
                hvb_mat = hvb_mat.to(device)
                hvb_top = hvb_top.to(device)
                hvf_mat = hvf_mat.to(device)
                hvf_top = hvf_top.to(device)
                fdec_ix = fdec_ix.to(device)
                wordseq_ix_padded = wordseq_ix_padded.to(device)

                output = model(depth_ix, catb_ix, hvb_mat, hvf_mat, hvb_top, hvf_top, wordseq_ix_padded, wordseq_lens,
                               config.getboolean("AblateSyn"), config.getboolean("AblateSem"), config.getboolean("AblateWord"),
                               dev_mode=True, valid_ids=b_fid)
                _, fdec = torch.max(output.data, 1)

                dev_data_points += len(fdec)
                batch_correct = (fdec == fdec_ix).sum().item()
                dev_correct += batch_correct

                loss = criterion(output, fdec_ix)
                dev_loss += loss.item() * len(fdec)

        eprint("Epoch {:04d} | AvgTrainLoss {:.4f} | TrainAcc {:.4f} | DevLoss {:.4f} | DevAcc {:.4f} | Time {:.4f}".format(
                epoch+1, epoch_loss/math.ceil(len(depth)/config.getint("BatchSize")), 100*(epoch_correct/epoch_data_points),
                dev_loss/dev_data_points, 100*(dev_correct/dev_data_points), time.time()-start_time))
        start_time = time.time()

    return model, catb_to_ix, hvecb_to_ix, hvecf_to_ix, fdecs_to_ix, word_to_ix


def main(config):
    f_config = config["FModel"]
    model, catb_to_ix, hvecb_to_ix, hvecf_to_ix, fdecs_to_ix, word_to_ix = train(f_config)
    # for param in model.state_dict():
    #     eprint(param)

    model.eval()
    if f_config.getboolean("GPU"):
        catb_embeds = model.state_dict()["catb_embeds.weight"].data.cpu().numpy()
        hvecb_embeds = model.state_dict()["hvecb_embeds.weight"].data.cpu().numpy()
        hvecf_embeds = model.state_dict()["hvecf_embeds.weight"].data.cpu().numpy()
        word_embeds = model.state_dict()["word_embeds.weight"].data.cpu().numpy()
        ih_weights_0 = model.state_dict()["lstm.weight_ih_l0"].data.cpu().numpy()
        hh_weights_0 = model.state_dict()["lstm.weight_hh_l0"].data.cpu().numpy()
        ih_bias_0 = model.state_dict()["lstm.bias_ih_l0"].data.cpu().numpy()
        hh_bias_0 = model.state_dict()["lstm.bias_hh_l0"].data.cpu().numpy()
        ih_weights_1 = model.state_dict()["lstm.weight_ih_l1"].data.cpu().numpy()
        hh_weights_1 = model.state_dict()["lstm.weight_hh_l1"].data.cpu().numpy()
        ih_bias_1 = model.state_dict()["lstm.bias_ih_l1"].data.cpu().numpy()
        hh_bias_1 = model.state_dict()["lstm.bias_hh_l1"].data.cpu().numpy()
        first_weights = model.state_dict()["fc1.weight"].data.cpu().numpy()
        first_biases = model.state_dict()["fc1.bias"].data.cpu().numpy()
        second_weights = model.state_dict()["fc2.weight"].data.cpu().numpy()
        second_biases = model.state_dict()["fc2.bias"].data.cpu().numpy()
    else:
        catb_embeds = model.state_dict()["catb_embeds.weight"].data.numpy()
        hvecb_embeds = model.state_dict()["hvecb_embeds.weight"].data.numpy()
        hvecf_embeds = model.state_dict()["hvecf_embeds.weight"].data.numpy()
        word_embeds = model.state_dict()["word_embeds.weight"].data.numpy()
        ih_weights_0 = model.state_dict()["lstm.weight_ih_l0"].data.numpy()
        hh_weights_0 = model.state_dict()["lstm.weight_hh_l0"].data.numpy()
        ih_bias_0 = model.state_dict()["lstm.bias_ih_l0"].data.numpy()
        hh_bias_0 = model.state_dict()["lstm.bias_hh_l0"].data.numpy()
        ih_weights_1 = model.state_dict()["lstm.weight_ih_l1"].data.numpy()
        hh_weights_1 = model.state_dict()["lstm.weight_hh_l1"].data.numpy()
        ih_bias_1 = model.state_dict()["lstm.bias_ih_l1"].data.numpy()
        hh_bias_1 = model.state_dict()["lstm.bias_hh_l1"].data.numpy()
        first_weights = model.state_dict()["fc1.weight"].data.numpy()
        first_biases = model.state_dict()["fc1.bias"].data.numpy()
        second_weights = model.state_dict()["fc2.weight"].data.numpy()
        second_biases = model.state_dict()["fc2.bias"].data.numpy()

    # eprint(model.word_embeds(torch.LongTensor([word_to_ix["in"]]).to("cuda")))
    # eprint(model.lstm(model.word_embeds(torch.LongTensor([word_to_ix["in"], word_to_ix["an"]]).to("cuda")).view(1, 2, 20)))

    # Final classifier parameters
    print("F F " + ",".join(map(str, first_weights.flatten("F").tolist())))
    print("F f " + ",".join(map(str, first_biases.flatten("F").tolist())))
    print("F S " + ",".join(map(str, second_weights.flatten("F").tolist())))
    print("F s " + ",".join(map(str, second_biases.flatten("F").tolist())))

    # Word LSTM parameters
    print("L 0 I " + ",".join(map(str, ih_weights_0.flatten("F").tolist())))
    print("L 0 i " + ",".join(map(str, ih_bias_0.flatten("F").tolist())))
    print("L 0 H " + ",".join(map(str, hh_weights_0.flatten("F").tolist())))
    print("L 0 h " + ",".join(map(str, hh_bias_0.flatten("F").tolist())))
    print("L 1 I " + ",".join(map(str, ih_weights_1.flatten("F").tolist())))
    print("L 1 i " + ",".join(map(str, ih_bias_1.flatten("F").tolist())))
    print("L 1 H " + ",".join(map(str, hh_weights_1.flatten("F").tolist())))
    print("L 1 h " + ",".join(map(str, hh_bias_1.flatten("F").tolist())))

    # Embedding parameters (syncat, hvec, word)
    if not f_config.getboolean("AblateSyn"):
        for cat, ix in sorted(catb_to_ix.items()):
            print("C B " + str(cat) + " [" + ",".join(map(str, catb_embeds[ix])) + "]")
    if not f_config.getboolean("AblateSem"):
        for hvec, ix in sorted(hvecb_to_ix.items()):
            print("K B " + str(hvec) + " [" + ",".join(map(str, hvecb_embeds[ix])) + "]")
        for hvec, ix in sorted(hvecf_to_ix.items()):
            print("K F " + str(hvec) + " [" + ",".join(map(str, hvecf_embeds[ix])) + "]")
    if not f_config.getboolean("AblateWord"):
        for word, ix in sorted(word_to_ix.items()):
            print("w " + str(word) + " [" + ",".join(map(str, word_embeds[ix])) + "]")

    # F decisions
    for fdec, ix in sorted(fdecs_to_ix.items()):
        print("f " + str(ix) + " " + str(fdec))


if __name__ == "__main__":
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(sys.argv[1])
    for section in config:
        eprint(section, dict(config[section]))
    main(config)
