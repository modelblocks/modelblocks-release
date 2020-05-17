import sys, configparser, torch, re, os, time
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.sparse import csr_matrix


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def prepare_data():
    data = [line.strip() for line in sys.stdin]
    depth, catAncstr, hvAncstr, hvFiller, catLchild, hvLchild, jDecs, hvAFirst, hvFFirst, hvLFirst = ([] for _ in
                                                                                                      range(10))

    for line in data:
        d, ca, hva, hvf, cl, hvl, jd = line.split(" ")
        depth.append(int(d))
        catAncstr.append(ca)
        hvAncstr.append(hva)
        hvFiller.append(hvf)
        catLchild.append(cl)
        hvLchild.append(hvl)
        jDecs.append(jd)
    eprint("Linesplit complete")

    # Extract first KVec from sparse HVec
    for kvec in hvAncstr:
        match = re.findall(r"^\[(.*?)\]", kvec)
        hvAFirst.append(match[0].split(","))
    eprint("hvAncstr ready")

    for kvec in hvFiller:
        match = re.findall(r"^\[(.*?)\]", kvec)
        hvFFirst.append(match[0].split(","))
    eprint("hvFiller ready")

    for kvec in hvLchild:
        match = re.findall(r"^\[(.*?)\]", kvec)
        hvLFirst.append(match[0].split(","))
    eprint("hvLChild ready")

    # Mapping from category & HVec to index
    flat_hvA = [hvec for sublist in hvAFirst for hvec in sublist if hvec not in ["", "Bot", "Top"]]
    flat_hvF = [hvec for sublist in hvFFirst for hvec in sublist if hvec not in ["", "Bot", "Top"]]
    flat_hvL = [hvec for sublist in hvLFirst for hvec in sublist if hvec not in ["", "Bot", "Top"]]
    cat_to_ix = {cat: i for i, cat in enumerate(sorted(set(catAncstr + catLchild)))}
    jdecs_to_ix = {jdecs: i for i, jdecs in enumerate(sorted(set(jDecs)))}
    hvec_to_ix = {hvec: i for i, hvec in enumerate(sorted(set(flat_hvA + flat_hvF + flat_hvL)))}

    cat_a_ix = [cat_to_ix[cat] for cat in catAncstr]
    cat_l_ix = [cat_to_ix[cat] for cat in catLchild]
    jdecs_ix = [jdecs_to_ix[jdecs] for jdecs in jDecs]

    hva_row, hva_col, hva_top, hvf_row, hvf_col, hvf_top, hvl_row, hvl_col, hvl_top = ([] for _ in range(9))

    # KVec indices and "Top" KVec counts
    for i, sublist in enumerate(hvAFirst):
        top_count = 0
        for hvec in sublist:
            if hvec == "Top":
                top_count += 1
            elif hvec == "" or hvec == "Bot":
                continue
            else:
                hva_row.append(i)
                hva_col.append(hvec_to_ix[hvec])
        hva_top.append([top_count])
    hva_mat = csr_matrix((np.ones(len(hva_row), dtype=np.int32), (hva_row, hva_col)),
                         shape=(len(hvAFirst), len(hvec_to_ix)))
    eprint("hva_mat ready")

    for i, sublist in enumerate(hvFFirst):
        top_count = 0
        for hvec in sublist:
            if hvec == "Top":
                top_count += 1
            elif hvec == "" or hvec == "Bot":
                continue
            else:
                hvf_row.append(i)
                hvf_col.append(hvec_to_ix[hvec])
        hvf_top.append([top_count])
    hvf_mat = csr_matrix((np.ones(len(hvf_row), dtype=np.int32), (hvf_row, hvf_col)),
                         shape=(len(hvFFirst), len(hvec_to_ix)))
    eprint("hvf_mat ready")

    for i, sublist in enumerate(hvLFirst):
        top_count = 0
        for hvec in sublist:
            if hvec == "Top":
                top_count += 1
            elif hvec == "" or hvec == "Bot":
                continue
            else:
                hvl_row.append(i)
                hvl_col.append(hvec_to_ix[hvec])
        hvl_top.append([top_count])
    hvl_mat = csr_matrix((np.ones(len(hvl_row), dtype=np.int32), (hvl_row, hvl_col)),
                         shape=(len(hvLFirst), len(hvec_to_ix)))
    eprint("hvl_mat ready")

    eprint("Number of input KVecs: {}".format(len(hvec_to_ix)))
    eprint("Number of output J categories: {}".format(len(jdecs_to_ix)))

    return depth, cat_a_ix, hva_mat, hvf_mat, cat_l_ix, hvl_mat, cat_to_ix, jdecs_ix, jdecs_to_ix, hvec_to_ix, hva_top, hvf_top, hvl_top


def prepare_data_dev(dev_decpars_file, cat_to_ix, jdecs_to_ix, hvec_to_ix):
    with open(dev_decpars_file, "r") as f:
        data = f.readlines()
        data = [line.strip() for line in data]

    depth, catAncstr, hvAncstr, hvFiller, catLchild, hvLchild, jDecs, hvAFirst, hvFFirst, hvLFirst = ([] for _ in
                                                                                                      range(10))

    for line in data:
        d, ca, hva, hvf, cl, hvl, jd = line.split(" ")
        if ca not in cat_to_ix or cl not in cat_to_ix or jd not in jdecs_to_ix:
            continue
        depth.append(int(d))
        catAncstr.append(ca)
        hvAncstr.append(hva)
        hvFiller.append(hvf)
        catLchild.append(cl)
        hvLchild.append(hvl)
        jDecs.append(jd)

    for kvec in hvAncstr:
        match = re.findall(r"^\[(.*?)\]", kvec)
        hvAFirst.append(match[0].split(","))

    for kvec in hvFiller:
        match = re.findall(r"^\[(.*?)\]", kvec)
        hvFFirst.append(match[0].split(","))

    for kvec in hvLchild:
        match = re.findall(r"^\[(.*?)\]", kvec)
        hvLFirst.append(match[0].split(","))

    cat_a_ix = [cat_to_ix[cat] for cat in catAncstr]
    cat_l_ix = [cat_to_ix[cat] for cat in catLchild]
    jdecs_ix = [jdecs_to_ix[jdecs] for jdecs in jDecs]

    hva_row, hva_col, hva_top, hvf_row, hvf_col, hvf_top, hvl_row, hvl_col, hvl_top = ([] for _ in range(9))

    # KVec indices and "Top" KVec counts
    for i, sublist in enumerate(hvAFirst):
        top_count = 0
        for hvec in sublist:
            if hvec == "Top":
                top_count += 1
            elif hvec == "" or hvec == "Bot" or hvec not in hvec_to_ix:
                continue
            else:
                hva_row.append(i)
                hva_col.append(hvec_to_ix[hvec])
        hva_top.append([top_count])
    hva_mat = csr_matrix((np.ones(len(hva_row), dtype=np.int32), (hva_row, hva_col)),
                         shape=(len(hvAFirst), len(hvec_to_ix)))

    for i, sublist in enumerate(hvFFirst):
        top_count = 0
        for hvec in sublist:
            if hvec == "Top":
                top_count += 1
            elif hvec == "" or hvec == "Bot" or hvec not in hvec_to_ix:
                continue
            else:
                hvf_row.append(i)
                hvf_col.append(hvec_to_ix[hvec])
        hvf_top.append([top_count])
    hvf_mat = csr_matrix((np.ones(len(hvf_row), dtype=np.int32), (hvf_row, hvf_col)),
                         shape=(len(hvFFirst), len(hvec_to_ix)))

    for i, sublist in enumerate(hvLFirst):
        top_count = 0
        for hvec in sublist:
            if hvec == "Top":
                top_count += 1
            elif hvec == "" or hvec == "Bot" or hvec not in hvec_to_ix:
                continue
            else:
                hvl_row.append(i)
                hvl_col.append(hvec_to_ix[hvec])
        hvl_top.append([top_count])
    hvl_mat = csr_matrix((np.ones(len(hvl_row), dtype=np.int32), (hvl_row, hvl_col)),
                         shape=(len(hvLFirst), len(hvec_to_ix)))

    return depth, cat_a_ix, hva_mat, hvf_mat, cat_l_ix, hvl_mat, jdecs_ix, hva_top, hvf_top, hvl_top


class JModel(nn.Module):
    def __init__(self, cat_vocab_size, hvec_vocab_size, syn_size, sem_size, hidden_dim, output_dim, dropout_prob):
        super(JModel, self).__init__()
        self.hvec_vocab_size = hvec_vocab_size
        self.syn_size = syn_size
        self.sem_size = sem_size
        self.cat_embeds = nn.Embedding(cat_vocab_size, syn_size)
        self.hvec_embeds = nn.Embedding(hvec_vocab_size, sem_size)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        self.fc1 = nn.Linear(7 + 2 * syn_size + 3 * sem_size, self.hidden_dim, bias=True)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.relu = F.relu
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim, bias=True)

    def forward(self, d_onehot, cat_a_ix, hva_mat, hvf_mat, cat_l_ix, hvl_mat, hva_top, hvf_top, hvl_top, use_gpu,
                ablate_syn, ablate_sem):
        hva_top = torch.FloatTensor(hva_top)
        hvf_top = torch.FloatTensor(hvf_top)
        hvl_top = torch.FloatTensor(hvl_top)

        if ablate_syn:
            cat_a_embed = torch.zeros([len(cat_a_ix), self.syn_size], dtype=torch.float)
            cat_l_embed = torch.zeros([len(cat_l_ix), self.syn_size], dtype=torch.float)

        else:
            cat_a_embed = self.cat_embeds(cat_a_ix)
            cat_l_embed = self.cat_embeds(cat_l_ix)

        if use_gpu >= 0:
            hva_top = hva_top.to("cuda")
            hvf_top = hvf_top.to("cuda")
            hvl_top = hvl_top.to("cuda")
            cat_a_embed = cat_a_embed.to("cuda")
            cat_l_embed = cat_l_embed.to("cuda")

        if ablate_sem:
            hva_embed = torch.zeros([hva_top.shape[0], self.sem_size], dtype=torch.float) + hva_top
            hvf_embed = torch.zeros([hvf_top.shape[0], self.sem_size], dtype=torch.float) + hvf_top
            hvl_embed = torch.zeros([hvl_top.shape[0], self.sem_size], dtype=torch.float) + hvl_top

            if use_gpu >= 0:
                hva_embed = hva_embed.to("cuda")
                hvf_embed = hvf_embed.to("cuda")
                hvl_embed = hvl_embed.to("cuda")

        else:
            hva_mat = hva_mat.tocoo()
            hva_mat = torch.sparse.FloatTensor(torch.LongTensor([hva_mat.row.tolist(), hva_mat.col.tolist()]),
                                               torch.FloatTensor(hva_mat.data.astype(np.float32)),
                                               torch.Size(hva_mat.shape))
            hvf_mat = hvf_mat.tocoo()
            hvf_mat = torch.sparse.FloatTensor(torch.LongTensor([hvf_mat.row.tolist(), hvf_mat.col.tolist()]),
                                               torch.FloatTensor(hvf_mat.data.astype(np.float32)),
                                               torch.Size(hvf_mat.shape))
            hvl_mat = hvl_mat.tocoo()
            hvl_mat = torch.sparse.FloatTensor(torch.LongTensor([hvl_mat.row.tolist(), hvl_mat.col.tolist()]),
                                               torch.FloatTensor(hvl_mat.data.astype(np.float32)),
                                               torch.Size(hvl_mat.shape))
            if use_gpu >= 0:
                hva_mat = hva_mat.to("cuda")
                hvf_mat = hvf_mat.to("cuda")
                hvl_mat = hvl_mat.to("cuda")

            hva_embed = torch.sparse.mm(hva_mat, self.hvec_embeds.weight) + hva_top
            hvf_embed = torch.sparse.mm(hvf_mat, self.hvec_embeds.weight) + hvf_top
            hvl_embed = torch.sparse.mm(hvl_mat, self.hvec_embeds.weight) + hvl_top

        x = torch.cat((cat_a_embed, hva_embed, hvf_embed, cat_l_embed, hvl_embed, d_onehot), 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(use_dev, dev_decpars_file, use_gpu, syn_size, sem_size, hidden_dim, dropout_prob, num_epochs, batch_size,
          learning_rate, weight_decay, l2_reg, ablate_syn, ablate_sem):
    depth, cat_a_ix, hva_mat, hvf_mat, cat_l_ix, hvl_mat, cat_to_ix, jdecs_ix, jdecs_to_ix, hvec_to_ix, hva_top, hvf_top, hvl_top = prepare_data()
    depth = F.one_hot(torch.LongTensor(depth), 7).float()
    cat_a_ix = torch.LongTensor(cat_a_ix)
    cat_l_ix = torch.LongTensor(cat_l_ix)
    target = torch.LongTensor(jdecs_ix)
    model = JModel(len(cat_to_ix), len(hvec_to_ix), syn_size, sem_size, hidden_dim, len(jdecs_to_ix), dropout_prob)

    if use_gpu >= 0:
        depth = depth.to("cuda")
        cat_a_ix = cat_a_ix.to("cuda")
        cat_l_ix = cat_l_ix.to("cuda")
        target = target.to("cuda")
        model = model.cuda()

    if use_dev >= 0:
        dev_depth, dev_cat_a_ix, dev_hva_mat, dev_hvf_mat, dev_cat_l_ix, dev_hvl_mat, dev_jdecs_ix, dev_hva_top, dev_hvf_top, dev_hvl_top = prepare_data_dev(
            dev_decpars_file, cat_to_ix, jdecs_to_ix, hvec_to_ix)
        dev_depth = F.one_hot(torch.LongTensor(dev_depth), 7).float()
        dev_cat_a_ix = torch.LongTensor(dev_cat_a_ix)
        dev_cat_l_ix = torch.LongTensor(dev_cat_l_ix)
        dev_target = torch.LongTensor(dev_jdecs_ix)

        if use_gpu >= 0:
            dev_depth = dev_depth.to("cuda")
            dev_cat_a_ix = dev_cat_a_ix.to("cuda")
            dev_cat_l_ix = dev_cat_l_ix.to("cuda")
            dev_target = dev_target.to("cuda")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.NLLLoss()

    # training loop
    eprint("Start JModel training...")
    epoch = 0

    while True:
        c0 = time.time()
        model.train()
        epoch += 1
        permutation = torch.randperm(len(depth))
        total_train_correct = 0
        total_train_loss = 0
        total_dev_loss = 0

        for i in range(0, len(depth), batch_size):
            indices = permutation[i:i + batch_size]
            batch_d, batch_a, batch_l, batch_target = depth[indices], cat_a_ix[indices], cat_l_ix[indices], target[
                indices]
            batch_hva_mat, batch_hvf_mat, batch_hvl_mat = hva_mat[np.array(indices), :], hvf_mat[np.array(indices),
                                                                                         :], hvl_mat[np.array(indices),
                                                                                             :]
            batch_hva_top, batch_hvf_top, batch_hvl_top = [hva_top[i] for i in indices], [hvf_top[i] for i in
                                                                                          indices], [hvl_top[i] for i in
                                                                                                     indices]

            if use_gpu >= 0:
                l2_loss = torch.cuda.FloatTensor([0])
            else:
                l2_loss = torch.FloatTensor([0])
            for param in model.parameters():
                l2_loss += torch.mean(param.pow(2))

            output = model(batch_d, batch_a, batch_hva_mat, batch_hvf_mat, batch_l, batch_hvl_mat, batch_hva_top,
                           batch_hvf_top, batch_hvl_top, use_gpu, ablate_syn, ablate_sem)
            _, jdec = torch.max(output.data, 1)
            train_correct = (jdec == batch_target).sum().item()
            total_train_correct += train_correct
            nll_loss = criterion(output, batch_target)
            loss = nll_loss + l2_reg * l2_loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if use_dev >= 0:
            with torch.no_grad():
                dev_pred = model(dev_depth, dev_cat_a_ix, dev_hva_mat, dev_hvf_mat, dev_cat_l_ix, dev_hvl_mat,
                                 dev_hva_top, dev_hvf_top, dev_hvl_top, use_gpu, ablate_syn, ablate_sem)
                _, dev_jdec = torch.max(dev_pred.data, 1)
                dev_correct = (dev_jdec == dev_target).sum().item()
                dev_loss = criterion(dev_pred, dev_target)
                total_dev_loss += dev_loss.item()
                dev_acc = 100 * (dev_correct / len(dev_depth))
        else:
            dev_acc = 0

        eprint("Epoch {:04d} | AvgTrainLoss {:.4f} | TrainAcc {:.4f} | DevLoss {:.4f} | DevAcc {:.4f} | Time {:.4f}".
               format(epoch, total_train_loss / (len(depth) // batch_size), 100 * (total_train_correct / len(depth)),
                      total_dev_loss, dev_acc, time.time() - c0))

        if epoch == num_epochs:
            break

    return model, cat_to_ix, jdecs_to_ix, hvec_to_ix


def main(config):
    j_config = config["JModel"]
    torch.manual_seed(j_config.getint("Seed"))
    model, cat_to_ix, jdecs_to_ix, hvec_to_ix = train(j_config.getint("Dev"), j_config.get("DevFile"),
                                                      j_config.getint("GPU"), j_config.getint("SynSize"),
                                                      j_config.getint("SemSize"), j_config.getint("HiddenSize"),
                                                      j_config.getfloat("DropoutProb"), j_config.getint("NEpochs"),
                                                      j_config.getint("BatchSize"), j_config.getfloat("LearningRate"),
                                                      j_config.getfloat("WeightDecay"), j_config.getfloat("L2Reg"),
                                                      j_config.getboolean("AblateSyn"), j_config.getboolean("AblateSem"))

    if j_config.getint("GPU") >= 0:
        cat_embeds = list(model.parameters())[0].data.cpu().numpy()
        hvec_embeds = list(model.parameters())[1].data.cpu().numpy()
        first_weights = list(model.parameters())[2].data.cpu().numpy()
        first_biases = list(model.parameters())[3].data.cpu().numpy()
        second_weights = list(model.parameters())[4].data.cpu().numpy()
        second_biases = list(model.parameters())[5].data.cpu().numpy()
    else:
        cat_embeds = list(model.parameters())[0].data.numpy()
        hvec_embeds = list(model.parameters())[1].data.numpy()
        first_weights = list(model.parameters())[2].data.numpy()
        first_biases = list(model.parameters())[3].data.numpy()
        second_weights = list(model.parameters())[4].data.numpy()
        second_biases = list(model.parameters())[5].data.numpy()

    eprint(first_weights.shape, second_weights.shape)
    print("J F " + ",".join(map(str, first_weights.flatten('F').tolist())))
    print("J f " + ",".join(map(str, first_biases.flatten('F').tolist())))
    print("J S " + ",".join(map(str, second_weights.flatten('F').tolist())))
    print("J s " + ",".join(map(str, second_biases.flatten('F').tolist())))
    if not j_config.getboolean("AblateSyn"):
        for cat, ix in sorted(cat_to_ix.items()):
            print("C " + str(cat) + " [" + ",".join(map(str, cat_embeds[ix])) + "]")
    if not j_config.getboolean("AblateSem"):
        for hvec, ix in sorted(hvec_to_ix.items()):
            print("K " + str(hvec) + " [" + ",".join(map(str, hvec_embeds[ix])) + "]")
    for jdec, ix in sorted(jdecs_to_ix.items()):
        print("j " + str(ix) + " " + str(jdec))


if __name__ == "__main__":
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(sys.argv[1])
    for section in config:
        eprint(section, dict(config[section]))
    main(config)
