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
    depth, catBase, hvBase, hvFiller, fDecs, hvBFirst, hvFFirst, hvAnte, hvAFirst, nullA = ([] for _ in range(10))

    for line in data:
        d, cb, hvb, hvf, hva, nulla, fd = line.split(" ")
        depth.append(int(d))
        catBase.append(cb)
        hvBase.append(hvb)
        hvFiller.append(hvf)
        hvAnte.append(hva)
        nullA.append(int(nulla))
        fDecs.append(fd)
    eprint("Linesplit complete")

    # Extract first KVec from sparse HVec
    for kvec in hvBase:
        match = re.findall(r"^\[(.*?)\]", kvec)
        hvBFirst.append(match[0].split(","))
    eprint("hvBase ready")

    for kvec in hvFiller:
        match = re.findall(r"^\[(.*?)\]", kvec)
        hvFFirst.append(match[0].split(","))
    eprint("hvFiller ready")

    for kvec in hvAnte:
        match = re.findall(r"^\[(.*?)\]", kvec)
        hvAFirst.append(match[0].split(","))
    eprint("hvAnte ready")

    # Mapping from category & HVec to index
    flat_hvB = [hvec for sublist in hvBFirst for hvec in sublist if hvec not in ["", "Bot", "Top"]]
    flat_hvF = [hvec for sublist in hvFFirst for hvec in sublist if hvec not in ["", "Bot", "Top"]]
    flat_hvA = [hvec for sublist in hvAFirst for hvec in sublist if hvec not in ["", "Bot", "Top"]]
    catb_to_ix = {cat: i for i, cat in enumerate(sorted(set(catBase)))}
    fdecs_to_ix = {fdecs: i for i, fdecs in enumerate(sorted(set(fDecs)))}
    hvecb_to_ix = {hvec: i for i, hvec in enumerate(sorted(set(flat_hvB)))}
    hvecf_to_ix = {hvec: i for i, hvec in enumerate(sorted(set(flat_hvF)))}
    hveca_to_ix = {hvec: i for i, hvec in enumerate(sorted(set(flat_hvA)))}

    cat_b_ix = [catb_to_ix[cat] for cat in catBase]
    fdecs_ix = [fdecs_to_ix[fdecs] for fdecs in fDecs]

    hvb_row, hvb_col, hvb_top, hvf_row, hvf_col, hvf_top, hva_row, hva_col, hva_top = ([] for _ in range(9))

    # KVec index sparse matrix and "Top" KVec counts
    for i, sublist in enumerate(hvBFirst):
        top_count = 0
        for hvec in sublist:
            if hvec == "Top":
                top_count += 1
            elif hvec == "" or hvec == "Bot":
                continue
            else:
                hvb_row.append(i)
                hvb_col.append(hvecb_to_ix[hvec])
        hvb_top.append([top_count])
    hvb_mat = csr_matrix((np.ones(len(hvb_row), dtype=np.int32), (hvb_row, hvb_col)),
                         shape=(len(hvBFirst), len(hvecb_to_ix)))
    eprint("hvb_mat ready")

    for i, sublist in enumerate(hvFFirst):
        top_count = 0
        for hvec in sublist:
            if hvec == "Top":
                top_count += 1
            elif hvec == "" or hvec == "Bot":
                continue
            else:
                hvf_row.append(i)
                hvf_col.append(hvecf_to_ix[hvec])
        hvf_top.append([top_count])
    hvf_mat = csr_matrix((np.ones(len(hvf_row), dtype=np.int32), (hvf_row, hvf_col)),
                         shape=(len(hvFFirst), len(hvecf_to_ix)))
    eprint("hvf_mat ready")

    for i, sublist in enumerate(hvAFirst):
        top_count = 0
        for hvec in sublist:
            if hvec == "Top":
                top_count += 1
            elif hvec == "" or hvec == "Bot":
                continue
            else:
                hva_row.append(i)
                hva_col.append(hveca_to_ix[hvec])
        hva_top.append([top_count])
    hva_mat = csr_matrix((np.ones(len(hva_row), dtype=np.int32), (hva_row, hva_col)),
                         shape=(len(hvAFirst), len(hveca_to_ix)))
    eprint("hva_mat ready")

    eprint("Number of input base CVecs: {}".format(len(catb_to_ix)))
    eprint("Number of input base KVecs: {}".format(len(hvecb_to_ix)))
    eprint("Number of input filler KVecs: {}".format(len(hvecf_to_ix)))
    eprint("Number of input antecedent KVecs: {}".format(len(hveca_to_ix)))
    eprint("Number of output F categories: {}".format(len(fdecs_to_ix)))

    return depth, cat_b_ix, hvb_mat, hvf_mat, catb_to_ix, fdecs_ix, fdecs_to_ix, hvecb_to_ix, hvecf_to_ix, hveca_to_ix, hvb_top, hvf_top, hva_mat, hva_top, nullA


def prepare_data_dev(dev_decpars_file, catb_to_ix, fdecs_to_ix, hvecb_to_ix, hvecf_to_ix, hveca_to_ix):
    with open(dev_decpars_file, "r") as f:
        data = f.readlines()
        data = [line.strip() for line in data]

    depth, catBase, hvBase, hvFiller, fDecs, hvBFirst, hvFFirst, hvAnte, hvAFirst, nullA = ([] for _ in range(10))

    for line in data:
        d, cb, hvb, hvf, hva, nulla, fd = line.split(" ")
        if cb not in catb_to_ix or fd not in fdecs_to_ix:
            continue
        depth.append(int(d))
        catBase.append(cb)
        hvBase.append(hvb)
        hvFiller.append(hvf)
        hvAnte.append(hva)
        nullA.append(int(nulla))
        fDecs.append(fd)

    # Extract first KVec from sparse HVec
    for kvec in hvBase:
        match = re.findall(r"^\[(.*?)\]", kvec)
        hvBFirst.append(match[0].split(","))

    for kvec in hvFiller:
        match = re.findall(r"^\[(.*?)\]", kvec)
        hvFFirst.append(match[0].split(","))

    for kvec in hvAnte:
        match = re.findall(r"^\[(.*?)\]", kvec)
        hvAFirst.append(match[0].split(","))

    cat_b_ix = [catb_to_ix[cat] for cat in catBase]
    fdecs_ix = [fdecs_to_ix[fdecs] for fdecs in fDecs]

    hvb_row, hvb_col, hvb_top, hvf_row, hvf_col, hvf_top, hva_row, hva_col, hva_top = ([] for _ in range(9))

    # KVec indices and "Top" KVec counts
    for i, sublist in enumerate(hvBFirst):
        top_count = 0
        for hvec in sublist:
            if hvec == "Top":
                top_count += 1
            elif hvec == "" or hvec == "Bot" or hvec not in hvecb_to_ix:
                continue
            else:
                hvb_row.append(i)
                hvb_col.append(hvecb_to_ix[hvec])
        hvb_top.append([top_count])
    hvb_mat = csr_matrix((np.ones(len(hvb_row), dtype=np.int32), (hvb_row, hvb_col)),
                         shape=(len(hvBFirst), len(hvecb_to_ix)))

    for i, sublist in enumerate(hvFFirst):
        top_count = 0
        for hvec in sublist:
            if hvec == "Top":
                top_count += 1
            elif hvec == "" or hvec == "Bot" or hvec not in hvecf_to_ix:
                continue
            else:
                hvf_row.append(i)
                hvf_col.append(hvecf_to_ix[hvec])
        hvf_top.append([top_count])
    hvf_mat = csr_matrix((np.ones(len(hvf_row), dtype=np.int32), (hvf_row, hvf_col)),
                         shape=(len(hvFFirst), len(hvecf_to_ix)))

    for i, sublist in enumerate(hvAFirst):
        top_count = 0
        for hvec in sublist:
            if hvec == "Top":
                top_count += 1
            elif hvec == "" or hvec == "Bot" or hvec not in hveca_to_ix:
                continue
            else:
                hva_row.append(i)
                hva_col.append(hveca_to_ix[hvec])
        hva_top.append([top_count])
    hva_mat = csr_matrix((np.ones(len(hva_row), dtype=np.int32), (hva_row, hva_col)),
                         shape=(len(hvAFirst), len(hveca_to_ix)))

    return depth, cat_b_ix, hvb_mat, hvf_mat, fdecs_ix, hvb_top, hvf_top, hva_mat, hva_top, nullA 


class FModel(nn.Module):
    def __init__(self, catb_vocab_size, hvecb_vocab_size, hvecf_vocab_size, hveca_vocab_size, syn_size, sem_size, ant_size, hidden_dim, output_dim, dropout_prob):
        super(FModel, self).__init__()
        self.syn_size = syn_size
        self.sem_size = sem_size
        self.catb_embeds = nn.Embedding(catb_vocab_size, syn_size)
        self.hvecb_embeds = nn.Embedding(hvecb_vocab_size, sem_size)
        self.hvecf_embeds = nn.Embedding(hvecf_vocab_size, sem_size)
        self.hveca_embeds = nn.Embedding(hveca_vocab_size, ant_size)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        self.fc1 = nn.Linear(8 + syn_size + 2 * sem_size + ant_size, self.hidden_dim, bias=True)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.relu = F.relu
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim, bias=True)

    def forward(self, d_onehot, cat_b_ix, hvb_mat, hvf_mat, hvb_top, hvf_top, hva_mat, hva_top, nullA, use_gpu, ablate_syn, ablate_sem):
        hvb_top = torch.FloatTensor(hvb_top)
        hvf_top = torch.FloatTensor(hvf_top)
        hva_top = torch.FloatTensor(hva_top)
        nullA   = torch.FloatTensor(nullA)

        if ablate_syn:
            cat_b_embed = torch.zeros([len(cat_b_ix), self.syn_size], dtype=torch.float)

        else:
            cat_b_embed = self.catb_embeds(cat_b_ix)

        if use_gpu >= 0:
            hvb_top = hvb_top.to("cuda")
            hvf_top = hvf_top.to("cuda")
            hva_top = hva_top.to("cuda")
            cat_b_embed = cat_b_embed.to("cuda")

        if ablate_sem:
            hvb_embed = torch.zeros([hvb_top.shape[0], self.sem_size], dtype=torch.float) + hvb_top
            hvf_embed = torch.zeros([hvb_top.shape[0], self.sem_size], dtype=torch.float) + hvf_top
            hva_embed = torch.zeros([hva_top.shape[0], self.ant_size], dtype=torch.float) + hva_top

            if use_gpu >= 0:
                hvb_embed = hvb_embed.to("cuda")
                hvf_embed = hvf_embed.to("cuda")
                hva_embed = hva_embed.to("cuda")

        else:
            hvb_mat = hvb_mat.tocoo()
            hvb_mat = torch.sparse.FloatTensor(torch.LongTensor([hvb_mat.row.tolist(), hvb_mat.col.tolist()]),
                                               torch.FloatTensor(hvb_mat.data.astype(np.float32)),
                                               torch.Size(hvb_mat.shape))
            hvf_mat = hvf_mat.tocoo()
            hvf_mat = torch.sparse.FloatTensor(torch.LongTensor([hvf_mat.row.tolist(), hvf_mat.col.tolist()]),
                                               torch.FloatTensor(hvf_mat.data.astype(np.float32)),
                                               torch.Size(hvf_mat.shape))
            hva_mat = hva_mat.tocoo()
            hva_mat = torch.sparse.FloatTensor(torch.LongTensor([hva_mat.row.tolist(), hva_mat.col.tolist()]),
                                               torch.FloatTensor(hva_mat.data.astype(np.float32)),
                                               torch.Size(hva_mat.shape))

            if use_gpu >= 0:
                hvb_mat = hvb_mat.to("cuda")
                hvf_mat = hvf_mat.to("cuda")
                hva_mat = hva_mat.to("cuda")

            hvb_embed = torch.sparse.mm(hvb_mat, self.hvecb_embeds.weight) + hvb_top
            hvf_embed = torch.sparse.mm(hvf_mat, self.hvecf_embeds.weight) + hvf_top
            hva_embed = torch.sparse.mm(hva_mat, self.hveca_embeds.weight) + hva_top

        x = torch.cat((cat_b_embed, hvb_embed, hvf_embed, hva_embed, nullA.unsqueeze(dim=1), d_onehot), 1)  
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(use_dev, dev_decpars_file, use_gpu, syn_size, sem_size, ant_size, hidden_dim, dropout_prob, num_epochs, batch_size, learning_rate,
          weight_decay, l2_reg, ablate_syn, ablate_sem):
    depth, cat_b_ix, hvb_mat, hvf_mat, catb_to_ix, fdecs_ix, fdecs_to_ix, hvecb_to_ix, hvecf_to_ix, hveca_to_ix, hvb_top, hvf_top, hva_mat, hva_top, nullA = prepare_data()
    depth = F.one_hot(torch.LongTensor(depth), 7).float()
    cat_b_ix = torch.LongTensor(cat_b_ix)
    target = torch.LongTensor(fdecs_ix)
    model = FModel(len(catb_to_ix), len(hvecb_to_ix), len(hvecf_to_ix), len(hveca_to_ix), syn_size, sem_size, ant_size, hidden_dim, len(fdecs_to_ix), dropout_prob)
    nulla = torch.FloatTensor(nullA)

    if use_gpu >= 0:
        depth = depth.to("cuda")
        cat_b_ix = cat_b_ix.to("cuda")
        target = target.to("cuda")
        nulla = nulla.to("cuda")
        model = model.cuda()

    if use_dev >= 0:
        dev_depth, dev_cat_b_ix, dev_hvb_mat, dev_hvf_mat, dev_fdecs_ix, dev_hvb_top, dev_hvf_top, dev_hva_mat, dev_hva_top, dev_nullA = prepare_data_dev(
            dev_decpars_file, catb_to_ix, fdecs_to_ix, hvecb_to_ix, hvecf_to_ix, hveca_to_ix)
#def prepare_data_dev(dev_decpars_file, catb_to_ix, fdecs_to_ix, hvecb_to_ix, hvecf_to_ix, hveca_to_ix):
        dev_depth = F.one_hot(torch.LongTensor(dev_depth), 7).float()
        dev_cat_b_ix = torch.LongTensor(dev_cat_b_ix)
        dev_target = torch.LongTensor(dev_fdecs_ix)

        if use_gpu >= 0:
            dev_depth = dev_depth.to("cuda")
            dev_cat_b_ix = dev_cat_b_ix.to("cuda")
            dev_target = dev_target.to("cuda")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.NLLLoss()

    # training loop
    eprint("Start FModel training...")
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
            batch_d, batch_c, batch_target, batch_nulla = depth[indices], cat_b_ix[indices], target[indices], nulla[indices]
            batch_hvb_mat, batch_hvf_mat, batch_hva_mat = hvb_mat[np.array(indices), :], hvf_mat[np.array(indices), :], hva_mat[np.array(indices), :]
            batch_hvb_top, batch_hvf_top, batch_hva_top = [hvb_top[i] for i in indices], [hvf_top[i] for i in indices], [hva_top[i] for i in indices]

            if use_gpu >= 0:
                l2_loss = torch.cuda.FloatTensor([0])
            else:
                l2_loss = torch.FloatTensor([0])
            for param in model.parameters():
                l2_loss += torch.mean(param.pow(2))

            output = model(batch_d, batch_c, batch_hvb_mat, batch_hvf_mat, batch_hvb_top, batch_hvf_top, batch_hva_mat, batch_hva_top, batch_nulla, use_gpu, ablate_syn, ablate_sem)
            _, fdec = torch.max(output.data, 1)
            train_correct = (fdec == batch_target).sum().item()
            total_train_correct += train_correct
            nll_loss = criterion(output, batch_target)
            loss = nll_loss + l2_reg * l2_loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if use_dev >= 0:
            with torch.no_grad():
                model.eval()
                dev_pred = model(dev_depth, dev_cat_b_ix, dev_hvb_mat, dev_hvf_mat, dev_hvb_top, dev_hvf_top, dev_hva_mat, dev_hva_top, dev_nullA, use_gpu,
                                 ablate_syn, ablate_sem)
                _, dev_fdec = torch.max(dev_pred.data, 1)
                dev_correct = (dev_fdec == dev_target).sum().item()
                dev_loss = criterion(dev_pred, dev_target)
                total_dev_loss += dev_loss.item()
                dev_acc = 100 * (dev_correct / len(dev_depth))
        else:
            dev_acc = 0

        eprint("Epoch {:04d} | AvgTrainLoss {:.4f} | TrainAcc {:.4f} | DevLoss {:.4f} | DevAcc {:.4f} | Time {:.4f}".
               format(epoch, total_train_loss / ((len(depth) // batch_size) + 1), 100 * (total_train_correct / len(depth)),
                      total_dev_loss, dev_acc, time.time() - c0))

        if epoch == num_epochs:
            break

    return model, catb_to_ix, fdecs_to_ix, hvecb_to_ix, hvecf_to_ix, hveca_to_ix


def main(config):
    f_config = config["FModel"]
    torch.manual_seed(f_config.getint("Seed"))
    model, catb_to_ix, fdecs_to_ix, hvecb_to_ix, hvecf_to_ix, hveca_to_ix = train(f_config.getint("Dev"), f_config.get("DevFile"),
                                                      f_config.getint("GPU"),
                                                      f_config.getint("SynSize"), f_config.getint("SemSize"),
                                                      f_config.getint("AntSize"), f_config.getint("HiddenSize"), 
                                                      f_config.getfloat("DropoutProb"),
                                                      f_config.getint("NEpochs"), f_config.getint("BatchSize"),
                                                      f_config.getfloat("LearningRate"),
                                                      f_config.getfloat("WeightDecay"), f_config.getfloat("L2Reg"),
                                                      f_config.getboolean("AblateSyn"),
                                                      f_config.getboolean("AblateSem"))

    model.eval()

    if f_config.getint("GPU") >= 0:
        catb_embeds = model.state_dict()["catb_embeds.weight"].data.cpu().numpy()
        hvecb_embeds = model.state_dict()["hvecb_embeds.weight"].data.cpu().numpy()
        hvecf_embeds = model.state_dict()["hvecf_embeds.weight"].data.cpu().numpy()
        hveca_embeds = model.state_dict()["hveca_embeds.weight"].data.cpu().numpy()
        first_weights = model.state_dict()["fc1.weight"].data.cpu().numpy()
        first_biases = model.state_dict()["fc1.bias"].data.cpu().numpy()
        second_weights = model.state_dict()["fc2.weight"].data.cpu().numpy()
        second_biases = model.state_dict()["fc2.bias"].data.cpu().numpy()
    else:
        catb_embeds = model.state_dict()["catb_embeds.weight"].data.numpy()
        hvecb_embeds = model.state_dict()["hvecb_embeds.weight"].data.numpy()
        hvecf_embeds = model.state_dict()["hvecf_embeds.weight"].data.numpy()
        hveca_embeds = model.state_dict()["hveca_embeds.weight"].data.numpy()
        first_weights = model.state_dict()["fc1.weight"].data.numpy()
        first_biases = model.state_dict()["fc1.bias"].data.numpy()
        second_weights = model.state_dict()["fc2.weight"].data.numpy()
        second_biases = model.state_dict()["fc2.bias"].data.numpy()

    eprint(first_weights.shape, second_weights.shape)
    print("F F " + ",".join(map(str, first_weights.flatten('F').tolist())))
    print("F f " + ",".join(map(str, first_biases.flatten('F').tolist())))
    print("F S " + ",".join(map(str, second_weights.flatten('F').tolist())))
    print("F s " + ",".join(map(str, second_biases.flatten('F').tolist())))

    if not f_config.getboolean("AblateSyn"):
        for cat, ix in sorted(catb_to_ix.items()):
            print("C B " + str(cat) + " " + ",".join(map(str, catb_embeds[ix])))
    if not f_config.getboolean("AblateSem"):
        for hvec, ix in sorted(hvecb_to_ix.items()):
            print("K B " + str(hvec) + " " + ",".join(map(str, hvecb_embeds[ix])))
        for hvec, ix in sorted(hvecf_to_ix.items()):
            print("K F " + str(hvec) + " " + ",".join(map(str, hvecf_embeds[ix])))
        for hvec, ix in sorted(hveca_to_ix.items()):
            print("K A " + str(hvec) + " " + ",".join(map(str, hveca_embeds[ix])))
    for fdec, ix in sorted(fdecs_to_ix.items()):
        print("f " + str(ix) + " " + str(fdec))


if __name__ == "__main__":
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(sys.argv[1])
    for section in config:
        eprint(section, dict(config[section]))
    main(config)
