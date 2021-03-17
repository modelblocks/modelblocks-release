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
    depth, catanc, hvanc, hvfiller, catlchild, hvlchild, jdecs, wordseq = ([] for _ in range(8))
    sent_depth, sent_catanc, sent_hvanc, sent_hvfiller, sent_catlchild, sent_hvlchild, sent_jdecs, sent_wordseq = ([] for _ in range(8))
    flat_catanc, flat_hvanc, flat_hvfiller, flat_catlchild, flat_hvlchild, flat_jdecs, flat_wordseq = ([] for _ in range(7))
    for line in data[1:]:
        if line.startswith("TREE") and "FAIL" in line:
            continue
        elif line.startswith("J"):
            _, d, ca, hva, hvf, cl, hvl, jd = line.split(" ")
            sent_depth.append(int(d))
            sent_catanc.append(ca)
            flat_catanc.append(ca)
            sent_hvanc.append(get_first_kvec(hva))
            flat_hvanc += get_first_kvec(hva)
            sent_hvfiller.append(get_first_kvec(hvf))
            flat_hvfiller += get_first_kvec(hvf)
            sent_catlchild.append(cl)
            flat_catlchild.append(cl)
            sent_hvlchild.append(get_first_kvec(hvl))
            flat_hvlchild += get_first_kvec(hvl)
            sent_jdecs.append(jd)
            flat_jdecs.append(jd)
        elif line.startswith("W"):
            _, _, _, _, _, _, _, word = line.split(" ")
            sent_wordseq.append(word.lower())
            flat_wordseq.append(word.lower())
        elif line.startswith("TREE"):
            depth.append(sent_depth)
            catanc.append(sent_catanc)
            hvanc.append(sent_hvanc)
            hvfiller.append(sent_hvfiller)
            catlchild.append(sent_catlchild)
            hvlchild.append(sent_hvlchild)
            jdecs.append(sent_jdecs)
            wordseq.append(sent_wordseq)
            # re-initialize sentence-level lists
            sent_depth, sent_catanc, sent_hvanc, sent_hvfiller, sent_catlchild, sent_hvlchild, sent_jdecs, sent_wordseq = ([] for _ in range(8))
    depth.append(sent_depth)
    catanc.append(sent_catanc)
    hvanc.append(sent_hvanc)
    hvfiller.append(sent_hvfiller)
    catlchild.append(sent_catlchild)
    hvlchild.append(sent_hvlchild)
    jdecs.append(sent_jdecs)
    wordseq.append(sent_wordseq)
    eprint("Training file processing complete")

    # Mapping from category & HVec to index
    cata_to_ix = {cat: i for i, cat in enumerate(sorted(set(flat_catanc)))}
    catl_to_ix = {cat: i for i, cat in enumerate(sorted(set(flat_catlchild)))}
    jdecs_to_ix = {jdecs: i for i, jdecs in enumerate(sorted(set(flat_jdecs)))}
    hveca_to_ix = {hvec: i for i, hvec in enumerate(sorted(set([hvec for hvec in flat_hvanc if hvec not in ["", "Bot", "Top"]])))}
    hvecf_to_ix = {hvec: i for i, hvec in enumerate(sorted(set([hvec for hvec in flat_hvfiller if hvec not in ["", "Bot", "Top"]])))}
    hvecl_to_ix = {hvec: i for i, hvec in enumerate(sorted(set([hvec for hvec in flat_hvlchild if hvec not in ["", "Bot", "Top"]])))}
    word_ctr = Counter(flat_wordseq)
    filtered_word_ctr = Counter({k: c for k, c in word_ctr.items() if c >= 2})
    word_to_ix = {word[0]: i for i, word in enumerate(filtered_word_ctr.items())}
    word_to_ix["<UNK>"] = len(word_to_ix)
    word_to_ix["<PAD>"] = len(word_to_ix)

    eprint("Number of input ancestor CVecs: {}".format(len(cata_to_ix)))
    eprint("Number of input ancestor KVecs: {}".format(len(hveca_to_ix)))
    eprint("Number of input left child CVecs: {}".format(len(catl_to_ix)))
    eprint("Number of input left child KVecs: {}".format(len(hvecl_to_ix)))
    eprint("Number of input filler KVecs: {}".format(len(hvecf_to_ix)))
    eprint("Number of words in vocabulary: {}".format(len(word_to_ix)-2))
    eprint("Number of output J categories: {}".format(len(jdecs_to_ix)))

    return (depth, catanc, hvanc, hvfiller, catlchild, hvlchild, jdecs, wordseq, cata_to_ix, hveca_to_ix, hvecf_to_ix, catl_to_ix, hvecl_to_ix, jdecs_to_ix, word_to_ix)


def prepare_data_dev(dev_decpars_file, cata_to_ix, catl_to_ix, jdecs_to_ix):
    with open(dev_decpars_file, "r") as f:
        data = f.readlines()
        data = [line.strip() for line in data]
    curr_jid = 0
    depth, catanc, hvanc, hvfiller, catlchild, hvlchild, jdecs, wordseq, jid = ([] for _ in range(9))
    sent_depth, sent_catanc, sent_hvanc, sent_hvfiller, sent_catlchild, sent_hvlchild, sent_jdecs, sent_wordseq, sent_jid = ([] for _ in range(9))
    for line in data[1:]:
        if line.startswith("TREE") and "FAIL" in line:
            continue
        elif line.startswith("J"):
            _, d, ca, hva, hvf, cl, hvl, jd = line.split(" ")
            if ca not in cata_to_ix or cl not in catl_to_ix or jd not in jdecs_to_ix:
                eprint("Unseen syntactic category or J decision found in dev file ({}, {}, {})!".format(ca, cl, jd))
                curr_jid += 1
                continue
            else:
                sent_depth.append(int(d))
                sent_catanc.append(ca)
                sent_hvanc.append(get_first_kvec(hva))
                sent_hvfiller.append(get_first_kvec(hvf))
                sent_catlchild.append(cl)
                sent_hvlchild.append(get_first_kvec(hvl))
                sent_jdecs.append(jd)
                sent_jid.append(curr_jid)
                curr_jid += 1
        elif line.startswith("W"):
            _, _, _, _, _, _, _, word = line.split(" ")
            sent_wordseq.append(word.lower())
        elif line.startswith("TREE"):
            depth.append(sent_depth)
            catanc.append(sent_catanc)
            hvanc.append(sent_hvanc)
            hvfiller.append(sent_hvfiller)
            catlchild.append(sent_catlchild)
            hvlchild.append(sent_hvlchild)
            jdecs.append(sent_jdecs)
            wordseq.append(sent_wordseq)
            jid.append(sent_jid)
            # re-initialize sentence-level lists and index
            curr_jid = 0
            sent_depth, sent_catanc, sent_hvanc, sent_hvfiller, sent_catlchild, sent_hvlchild, sent_jdecs, sent_wordseq, sent_jid = ([] for _ in range(9))
    depth.append(sent_depth)
    catanc.append(sent_catanc)
    hvanc.append(sent_hvanc)
    hvfiller.append(sent_hvfiller)
    catlchild.append(sent_catlchild)
    hvlchild.append(sent_hvlchild)
    jdecs.append(sent_jdecs)
    wordseq.append(sent_wordseq)
    jid.append(sent_jid)
    eprint("Dev file processing complete")

    return (depth, catanc, hvanc, hvfiller, catlchild, hvlchild, jdecs, wordseq, jid)


def create_one_batch(training_data, cata_to_ix, hveca_to_ix, hvecf_to_ix, catl_to_ix, hvecl_to_ix, jdecs_to_ix, word_to_ix):
    depth, catanc, hvanc, hvfiller, catlchild, hvlchild, jdecs, wordseq = training_data
    depth_ix = [d for sublist in depth for d in sublist]
    depth_ix = F.one_hot(torch.LongTensor(depth_ix), 7).float()
    cata_ix = [cata_to_ix[ca] for sublist in catanc for ca in sublist]
    cata_ix = torch.LongTensor(cata_ix)
    catl_ix = [catl_to_ix[cl] for sublist in catlchild for cl in sublist]
    catl_ix = torch.LongTensor(catl_ix)
    flat_hva = [hva for sublist in hvanc for hva in sublist]
    flat_hvf = [hvf for sublist in hvfiller for hvf in sublist]
    flat_hvl = [hvl for sublist in hvlchild for hvl in sublist]
    hva_mat, hva_top = get_hvec_ix_tensor(flat_hva, hveca_to_ix)
    hvf_mat, hvf_top = get_hvec_ix_tensor(flat_hvf, hvecf_to_ix)
    hvl_mat, hvl_top = get_hvec_ix_tensor(flat_hvl, hvecl_to_ix)
    jdec_ix = [jdecs_to_ix[jd] for sublist in jdecs for jd in sublist]
    jdec_ix = torch.LongTensor(jdec_ix)
    wordseq_ix = [[word_to_ix[word] if word in word_to_ix else word_to_ix["<UNK>"] for word in sublist] for sublist in wordseq]
    wordseq_lens = torch.LongTensor(list(map(len, wordseq_ix)))
    wordseq_max_len = wordseq_lens.max().item()
    wordseq_ix_padded = torch.zeros(1, dtype=torch.int64)
    wordseq_ix_padded = wordseq_ix_padded.new_full((len(wordseq_ix), wordseq_max_len), word_to_ix["<PAD>"])

    for ix, (seq, seqlen) in enumerate(zip(wordseq_ix, wordseq_lens)):
        wordseq_ix_padded[ix, :seqlen] = torch.LongTensor(seq)

    return depth_ix, cata_ix, hva_mat, hva_top, hvf_mat, hvf_top, catl_ix, hvl_mat, hvl_top, jdec_ix, wordseq_ix_padded, wordseq_lens


class JModel(nn.Module):
    def __init__(self,
        device,
        cata_vocab_size,
        catl_vocab_size,
        syn_dim,
        hveca_vocab_size,
        hvecf_vocab_size,
        hvecl_vocab_size,
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

        super(JModel, self).__init__()
        self.device = device
        self.cata_embeds = nn.Embedding(cata_vocab_size, syn_dim)
        self.catl_embeds = nn.Embedding(catl_vocab_size, syn_dim)
        self.hveca_embeds = nn.Embedding(hveca_vocab_size, sem_dim)
        self.hvecf_embeds = nn.Embedding(hvecf_vocab_size, sem_dim)
        self.hvecl_embeds = nn.Embedding(hvecl_vocab_size, sem_dim)
        self.word_embeds = nn.Embedding(word_vocab_size, word_dim)
        self.lstm = nn.LSTM(word_dim, lstm_hidden_dim, lstm_num_layers, batch_first=True, dropout=lstm_dropout)
        self.dropout_1 = nn.Dropout(ff_input_dropout)
        self.fc1 = nn.Linear(7 + 2*syn_dim + 3*sem_dim + lstm_hidden_dim, ff_hidden_dim, bias=True)
        self.dropout_2 = nn.Dropout(ff_hidden_dropout)
        self.relu = F.relu
        self.fc2 = nn.Linear(ff_hidden_dim, ff_output_dim, bias=True)

    def forward(self,
        d_onehot,
        cata_ix,
        catl_ix,
        hva_mat,
        hvf_mat,
        hvl_mat,
        hva_top,
        hvf_top,
        hvl_top,
        wordseq_ix_padded,
        wordseq_lens,
        ablate_syn,
        ablate_sem,
        ablate_word,
        dev_mode,
        valid_ids=[]):

        if ablate_syn:
            cata_embed = torch.zeros([len(cata_ix), self.syn_size], dtype=torch.float, device=self.device)
            catl_embed = torch.zeros([len(catl_ix), self.syn_size], dtype=torch.float, device=self.device)
        else:
            cata_embed = self.cata_embeds(cata_ix)
            catl_embed = self.catl_embeds(catl_ix)

        if ablate_sem:
            hva_embed = torch.zeros([hva_top.shape[0], self.sem_size], dtype=torch.float, device=self.device) + hva_top
            hvf_embed = torch.zeros([hvf_top.shape[0], self.sem_size], dtype=torch.float, device=self.device) + hvf_top
            hvl_embed = torch.zeros([hvl_top.shape[0], self.sem_size], dtype=torch.float, device=self.device) + hvl_top
        else:
            hva_embed = torch.sparse.mm(hva_mat, self.hveca_embeds.weight) + hva_top
            hvf_embed = torch.sparse.mm(hvf_mat, self.hvecf_embeds.weight) + hvf_top
            hvl_embed = torch.sparse.mm(hvl_mat, self.hvecl_embeds.weight) + hvl_top

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

        x = torch.cat((cata_embed, hva_embed, hvf_embed, catl_embed, hvl_embed, lstm_output, d_onehot), 1)
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

    depth, catanc, hvanc, hvfiller, catlchild, hvlchild, jdecs, wordseq, cata_to_ix, hveca_to_ix, hvecf_to_ix, catl_to_ix, hvecl_to_ix, jdecs_to_ix, word_to_ix = prepare_data()

    if config.getboolean("Dev"):
        d_depth, d_catanc, d_hvanc, d_hvfiller, d_catlchild, d_hvlchild, d_jdecs, d_wordseq, d_jid = prepare_data_dev(config.get("DevFile"), cata_to_ix, catl_to_ix, jdecs_to_ix)

    model = JModel(device,
                   len(cata_to_ix),
                   len(catl_to_ix),
                   config.getint("SynSize"),
                   len(hveca_to_ix),
                   len(hvecf_to_ix),
                   len(hvecl_to_ix),
                   config.getint("SemSize"),
                   len(word_to_ix),
                   config.getint("WordSize"),
                   config.getint("LSTMHiddenSize"),
                   config.getint("LSTMNLayers"),
                   config.getfloat("LSTMDropout"),
                   config.getfloat("FFInputDropout"),
                   config.getint("HiddenSize"),
                   config.getfloat("FFHiddenDropout"),
                   len(jdecs_to_ix))

    eprint(str(model))
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    eprint("Topmost model has {} parameters".format(num_params))

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.getfloat("LearningRate"), weight_decay=config.getfloat("WeightDecay"))
    criterion = nn.NLLLoss()

    # training loop
    eprint("Start JModel training...")
    start_time = time.time()

    for epoch in range(config.getint("NEpochs")):
        vars = list(zip(depth, catanc, hvanc, hvfiller, catlchild, hvlchild, jdecs, wordseq))
        random.shuffle(vars)
        depth, catanc, hvanc, hvfiller, catlchild, hvlchild, jdecs, wordseq = zip(*vars)
        model.train()
        epoch_data_points = 0
        epoch_correct = 0
        epoch_loss = 0
        dev_data_points = 100
        dev_correct = 0
        dev_loss = 0

        for j in range(0, len(depth), config.getint("BatchSize")):
            b_depth = depth[j:j+config.getint("BatchSize")]
            b_catanc = catanc[j:j+config.getint("BatchSize")]
            b_hvanc = hvanc[j:j+config.getint("BatchSize")]
            b_hvfiller = hvfiller[j:j+config.getint("BatchSize")]
            b_catlchild = catlchild[j:j+config.getint("BatchSize")]
            b_hvlchild = hvlchild[j:j+config.getint("BatchSize")]
            b_jdecs = jdecs[j:j+config.getint("BatchSize")]
            b_wordseq = wordseq[j:j+config.getint("BatchSize")]
            depth_ix, cata_ix, hva_mat, hva_top, hvf_mat, hvf_top, catl_ix, hvl_mat, hvl_top, jdec_ix, wordseq_ix_padded, wordseq_lens = \
            create_one_batch((b_depth, b_catanc, b_hvanc, b_hvfiller, b_catlchild, b_hvlchild, b_jdecs, b_wordseq),
                             cata_to_ix, hveca_to_ix, hvecf_to_ix, catl_to_ix, hvecl_to_ix, jdecs_to_ix, word_to_ix)

            # maybe move these to create_one_batch function
            depth_ix = depth_ix.to(device)
            cata_ix = cata_ix.to(device)
            hva_mat = hva_mat.to(device)
            hva_top = hva_top.to(device)
            hvf_mat = hvf_mat.to(device)
            hvf_top = hvf_top.to(device)
            catl_ix = catl_ix.to(device)
            hvl_mat = hvl_mat.to(device)
            hvl_top = hvl_top.to(device)
            jdec_ix = jdec_ix.to(device)
            wordseq_ix_padded = wordseq_ix_padded.to(device)

            optimizer.zero_grad()

            if config.getfloat("L2Reg") > 0:
                l2_loss = torch.cuda.FloatTensor([0]) if device == "cuda" else torch.FloatTensor([0])
                for param in model.parameters():
                    l2_loss += torch.mean(param.pow(2))
            else:
                l2_loss = 0

            output = model(depth_ix, cata_ix, catl_ix, hva_mat, hvf_mat, hvl_mat, hva_top, hvf_top, hvl_top, wordseq_ix_padded, wordseq_lens,
                           config.getboolean("AblateSyn"), config.getboolean("AblateSem"), config.getboolean("AblateWord"),
                           dev_mode=False)
            _, jdec = torch.max(output.data, 1)

            epoch_data_points += len(jdec)
            batch_correct = (jdec == jdec_ix).sum().item()
            epoch_correct += batch_correct
            nll_loss = criterion(output, jdec_ix)
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
                b_catanc = d_catanc[j:j+config.getint("DevBatchSize")]
                b_hvanc = d_hvanc[j:j+config.getint("DevBatchSize")]
                b_hvfiller = d_hvfiller[j:j+config.getint("DevBatchSize")]
                b_catlchild = d_catlchild[j:j+config.getint("DevBatchSize")]
                b_hvlchild = d_hvlchild[j:j+config.getint("DevBatchSize")]
                b_jdecs = d_jdecs[j:j+config.getint("DevBatchSize")]
                b_wordseq = d_wordseq[j:j+config.getint("DevBatchSize")]
                b_jid = d_jid[j:j+config.getint("DevBatchSize")]

                depth_ix, cata_ix, hva_mat, hva_top, hvf_mat, hvf_top, catl_ix, hvl_mat, hvl_top, jdec_ix, wordseq_ix_padded, wordseq_lens = \
                create_one_batch((b_depth, b_catanc, b_hvanc, b_hvfiller, b_catlchild, b_hvlchild, b_jdecs, b_wordseq),
                                 cata_to_ix, hveca_to_ix, hvecf_to_ix, catl_to_ix, hvecl_to_ix, jdecs_to_ix, word_to_ix)
                depth_ix = depth_ix.to(device)
                cata_ix = cata_ix.to(device)
                hva_mat = hva_mat.to(device)
                hva_top = hva_top.to(device)
                hvf_mat = hvf_mat.to(device)
                hvf_top = hvf_top.to(device)
                catl_ix = catl_ix.to(device)
                hvl_mat = hvl_mat.to(device)
                hvl_top = hvl_top.to(device)
                jdec_ix = jdec_ix.to(device)
                wordseq_ix_padded = wordseq_ix_padded.to(device)

                output = model(depth_ix, cata_ix, catl_ix, hva_mat, hvf_mat, hvl_mat, hva_top, hvf_top, hvl_top, wordseq_ix_padded, wordseq_lens,
                               config.getboolean("AblateSyn"), config.getboolean("AblateSem"), config.getboolean("AblateWord"),
                               dev_mode=True, valid_ids=b_jid)
                _, jdec = torch.max(output.data, 1)

                dev_data_points += len(jdec)
                batch_correct = (jdec == jdec_ix).sum().item()
                dev_correct += batch_correct

                loss = criterion(output, jdec_ix)
                dev_loss += loss.item() * len(jdec)

        eprint("Epoch {:04d} | AvgTrainLoss {:.4f} | TrainAcc {:.4f} | DevLoss {:.4f} | DevAcc {:.4f} | Time {:.4f}".format(
                epoch+1, epoch_loss/math.ceil(len(depth)/config.getint("BatchSize")), 100*(epoch_correct/epoch_data_points),
                dev_loss/dev_data_points, 100*(dev_correct/dev_data_points), time.time()-start_time))
        start_time = time.time()

    return model, cata_to_ix, catl_to_ix, hveca_to_ix, hvecf_to_ix, hvecl_to_ix, jdecs_to_ix, word_to_ix


def main(config):
    j_config = config["JModel"]
    model, cata_to_ix, catl_to_ix, hveca_to_ix, hvecf_to_ix, hvecl_to_ix, jdecs_to_ix, word_to_ix = train(j_config)

    model.eval()

    if j_config.getboolean("GPU"):
        cata_embeds = model.state_dict()["cata_embeds.weight"].data.cpu().numpy()
        catl_embeds = model.state_dict()["catl_embeds.weight"].data.cpu().numpy()
        hveca_embeds = model.state_dict()["hveca_embeds.weight"].data.cpu().numpy()
        hvecf_embeds = model.state_dict()["hvecf_embeds.weight"].data.cpu().numpy()
        hvecl_embeds = model.state_dict()["hvecl_embeds.weight"].data.cpu().numpy()
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
        cata_embeds = model.state_dict()["cata_embeds.weight"].data.numpy()
        catl_embeds = model.state_dict()["catl_embeds.weight"].data.numpy()
        hveca_embeds = model.state_dict()["hveca_embeds.weight"].data.numpy()
        hvecf_embeds = model.state_dict()["hvecf_embeds.weight"].data.numpy()
        hvecl_embeds = model.state_dict()["hvecl_embeds.weight"].data.numpy()
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
    eprint(model.lstm(model.word_embeds(torch.LongTensor([word_to_ix["in"], word_to_ix["an"]]).to("cuda")).view(1, 2, 20)))

    # Final classifier parameters
    print("J F " + ",".join(map(str, first_weights.flatten("F").tolist())))
    print("J f " + ",".join(map(str, first_biases.flatten("F").tolist())))
    print("J S " + ",".join(map(str, second_weights.flatten("F").tolist())))
    print("J s " + ",".join(map(str, second_biases.flatten("F").tolist())))

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
    if not j_config.getboolean("AblateSyn"):
        for cat, ix in sorted(cata_to_ix.items()):
            print("C A " + str(cat) + " [" + ",".join(map(str, cata_embeds[ix])) + "]")
        for cat, ix in sorted(catl_to_ix.items()):
            print("C L " + str(cat) + " [" + ",".join(map(str, catl_embeds[ix])) + "]")
    if not j_config.getboolean("AblateSem"):
        for hvec, ix in sorted(hveca_to_ix.items()):
            print("K A " + str(hvec) + " [" + ",".join(map(str, hveca_embeds[ix])) + "]")
        for hvec, ix in sorted(hvecf_to_ix.items()):
            print("K F " + str(hvec) + " [" + ",".join(map(str, hvecf_embeds[ix])) + "]")
        for hvec, ix in sorted(hvecl_to_ix.items()):
            print("K L " + str(hvec) + " [" + ",".join(map(str, hvecl_embeds[ix])) + "]")
    if not j_config.getboolean("AblateWord"):
        for word, ix in sorted(word_to_ix.items()):
            print("w " + str(word) + " [" + ",".join(map(str, word_embeds[ix])) + "]")

    # J decisions
    for jdec, ix in sorted(jdecs_to_ix.items()):
        print("j " + str(ix) + " " + str(jdec))


if __name__ == "__main__":
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(sys.argv[1])
    for section in config:
        eprint(section, dict(config[section]))
    main(config)
