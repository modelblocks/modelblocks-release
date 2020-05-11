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
    cat_to_ix = {cat: i for i, cat in enumerate(sorted(set(catBase)))}
    fdecs_to_ix = {fdecs: i for i, fdecs in enumerate(sorted(set(fDecs)))}
    hvec_to_ix = {hvec: i for i, hvec in enumerate(sorted(set(flat_hvB + flat_hvF + flat_hvA)))}

    cat_b_ix = [cat_to_ix[cat] for cat in catBase]
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
                hvb_col.append(hvec_to_ix[hvec])
        hvb_top.append([top_count])
    hvb_mat = csr_matrix((np.ones(len(hvb_row), dtype=np.int32), (hvb_row, hvb_col)),
                         shape=(len(hvBFirst), len(hvec_to_ix)))
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
                hvf_col.append(hvec_to_ix[hvec])
        hvf_top.append([top_count])
    hvf_mat = csr_matrix((np.ones(len(hvf_row), dtype=np.int32), (hvf_row, hvf_col)),
                         shape=(len(hvFFirst), len(hvec_to_ix)))
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
                hva_col.append(hvec_to_ix[hvec])
        hva_top.append([top_count])
    hva_mat = csr_matrix((np.ones(len(hva_row), dtype=np.int32), (hva_row, hva_col)),
                         shape=(len(hvAFirst), len(hvec_to_ix)))
    eprint("hva_mat ready")
    eprint("Number of input KVecs: {}".format(len(hvec_to_ix)))
    eprint("Number of output F categories: {}".format(len(fdecs_to_ix)))

    return depth, cat_b_ix, hvb_mat, hvf_mat, cat_to_ix, fdecs_ix, fdecs_to_ix, hvec_to_ix, hvb_top, hvf_top, hva_mat, hva_top, nullA 


def prepare_data_dev(dev_decpars_file, cat_to_ix, fdecs_to_ix, hvec_to_ix):
    #TODO add n model deps later
    with open(dev_decpars_file, "r") as f:
        data = f.readlines()
        data = [line.strip() for line in data]

    depth, catBase, hvBase, hvFiller, fDecs, hvBFirst, hvFFirst = ([] for _ in range(7))

    for line in data:
        d, cb, hvb, hvf, fd = line.split(" ")
        if cb not in cat_to_ix or fd not in fdecs_to_ix:
            continue
        depth.append(int(d))
        catBase.append(cb)
        hvBase.append(hvb)
        hvFiller.append(hvf)
        fDecs.append(fd)

    for kvec in hvBase:
        match = re.findall(r"^\[(.*?)\]", kvec)
        hvBFirst.append(match[0].split(","))

    for kvec in hvFiller:
        match = re.findall(r"^\[(.*?)\]", kvec)
        hvFFirst.append(match[0].split(","))

    cat_b_ix = [cat_to_ix[cat] for cat in catBase]
    fdecs_ix = [fdecs_to_ix[fdecs] for fdecs in fDecs]

    hvb_row, hvb_col, hvf_row, hvf_col, hvb_top, hvf_top = ([] for _ in range(6))

    # KVec indices and "Top" KVec counts
    for i, sublist in enumerate(hvBFirst):
        top_count = 0
        for hvec in sublist:
            if hvec == "Top":
                top_count += 1
            elif hvec == "" or hvec == "Bot":
                continue
            else:
                hvb_row.append(i)
                hvb_col.append(hvec_to_ix[hvec])
        hvb_top.append([top_count])
    hvb_mat = csr_matrix((np.ones(len(hvb_row), dtype=np.int32), (hvb_row, hvb_col)),
                         shape=(len(hvBFirst), len(hvec_to_ix)))

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

    return depth, cat_b_ix, hvb_mat, hvf_mat, fdecs_ix, hvb_top, hvf_top


class FModel(nn.Module):
    def __init__(self, cat_vocab_size, hvec_vocab_size, syn_size, sem_size, hidden_dim, output_dim):
        super(FModel, self).__init__()
        self.hvec_vocab_size = hvec_vocab_size
        self.sem_size = sem_size
        self.cat_embeds = nn.Embedding(cat_vocab_size, syn_size)
        self.hvec_embeds = nn.Embedding(hvec_vocab_size, sem_size)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(8 + syn_size + 3 * sem_size, self.hidden_dim, bias=True)
        self.relu = F.relu
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim, bias=True)

    def forward(self, d_onehot, cat_b_ix, hvb_mat, hvf_mat, hvb_top, hvf_top, hva_mat, hva_top, nullA, use_gpu, ablate_sem):
        cat_b_embed = self.cat_embeds(cat_b_ix)
        hvb_top = torch.FloatTensor(hvb_top)
        hvf_top = torch.FloatTensor(hvf_top)
        hva_top = torch.FloatTensor(hva_top)

        if use_gpu >= 0:
            cat_b_embed = cat_b_embed.to("cuda")
            hvb_mat = hvb_mat.to("cuda")
            hvf_mat = hvf_mat.to("cuda")
            hvb_top = hvb_top.to("cuda")
            hvf_top = hvf_top.to("cuda")
            hva_top = hva_top.to("cuda")
            hva_mat = hva_mat.to("cuda")

        if ablate_sem:
            hvb_embed = torch.zeros([hvb_top.shape[0], self.sem_size], dtype=torch.float) + hvb_top
            hvf_embed = torch.zeros([hvb_top.shape[0], self.sem_size], dtype=torch.float) + hvf_top
            hva_embed = torch.zeros([hva_top.shape[0], self.sem_size], dtype=torch.float) + hva_top

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
            hvb_embed = torch.sparse.mm(hvb_mat, self.hvec_embeds.weight) + hvb_top
            hvf_embed = torch.sparse.mm(hvf_mat, self.hvec_embeds.weight) + hvf_top
            hva_embed = torch.sparse.mm(hva_mat, self.hvec_embeds.weight) + hva_top

        x = torch.cat((cat_b_embed, hvb_embed, hvf_embed, hva_embed, nullA.unsqueeze(dim=1), d_onehot), 1)  
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(use_dev, dev_decpars_file, use_gpu, syn_size, sem_size, hidden_dim, num_epochs, batch_size, learning_rate,
          weight_decay, l2_reg, ablate_sem):
    depth, cat_b_ix, hvb_mat, hvf_mat, cat_to_ix, fdecs_ix, fdecs_to_ix, hvec_to_ix, hvb_top, hvf_top, hva_mat, hva_top, nullA = prepare_data()
    depth = F.one_hot(torch.LongTensor(depth), 7).float()
    cat_b_ix = torch.LongTensor(cat_b_ix)
    target = torch.LongTensor(fdecs_ix)
    model = FModel(len(cat_to_ix), len(hvec_to_ix), syn_size, sem_size, hidden_dim, len(fdecs_to_ix))
    nulla = torch.FloatTensor(nullA)

    if use_gpu >= 0:
        depth = depth.to("cuda")
        cat_b_ix = cat_b_ix.to("cuda")
        target = target.to("cuda")
        nulla = nulla.to("cuda")
        model = model.cuda()

    if use_dev >= 0:
        dev_depth, dev_cat_b_mat, dev_hvb_mat, dev_hvf_mat, dev_fdecs_ix, dev_hvb_top, dev_hvf_top = prepare_data_dev(
            dev_decpars_file, cat_to_ix, fdecs_to_ix, hvec_to_ix)
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
            #batch_hvb_mat, batch_hvf_mat = hvb_mat[np.array(indices), :], hvf_mat[np.array(indices), :]
            batch_hvb_mat, batch_hvf_mat, batch_hva_mat = hvb_mat[np.array(indices), :], hvf_mat[np.array(indices), :], hva_mat[np.array(indices), :]
            #batch_hvb_top, batch_hvf_top = [hvb_top[i] for i in indices], [hvf_top[i] for i in indices]
            batch_hvb_top, batch_hvf_top, batch_hva_top = [hvb_top[i] for i in indices], [hvf_top[i] for i in indices], [hva_top[i] for i in indices]

            if use_gpu >= 0:
                l2_loss = torch.cuda.FloatTensor([0])
            else:
                l2_loss = torch.FloatTensor([0])
            for param in model.parameters():
                l2_loss += torch.mean(param.pow(2))

            output = model(batch_d, batch_c, batch_hvb_mat, batch_hvf_mat, batch_hvb_top, batch_hvf_top, batch_hva_mat, batch_hva_top, batch_nulla, use_gpu,
                           ablate_sem)
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
            #TODO dev support for n dependencies
            with torch.no_grad():
                dev_pred = model(dev_depth, dev_cat_b_ix, dev_hvb_mat, dev_hvf_mat, dev_hvb_top, dev_hvf_top, use_gpu,
                                 ablate_sem)
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

    return model, cat_to_ix, fdecs_to_ix, hvec_to_ix


def main(config):
    f_config = config["FModel"]
    model, cat_to_ix, fdecs_to_ix, hvec_to_ix = train(f_config.getint("Dev"), f_config.get("DevFile"),
                                                      f_config.getint("GPU"),
                                                      f_config.getint("SynSize"), f_config.getint("SemSize"),
                                                      f_config.getint("HiddenSize"),
                                                      f_config.getint("NEpochs"), f_config.getint("BatchSize"),
                                                      f_config.getfloat("LearningRate"),
                                                      f_config.getfloat("WeightDecay"), f_config.getfloat("L2Reg"),
                                                      f_config.getboolean("AblateSem"))

    if f_config.getint("GPU") >= 0:
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
    print("F F " + ",".join(map(str, first_weights.flatten('F').tolist())))
    print("F f " + ",".join(map(str, first_biases.flatten('F').tolist())))
    print("F S " + ",".join(map(str, second_weights.flatten('F').tolist())))
    print("F s " + ",".join(map(str, second_biases.flatten('F').tolist())))
    for cat, ix in sorted(cat_to_ix.items()):
        print("C " + str(cat) + " [" + ",".join(map(str, cat_embeds[ix])) + "]")
    if not f_config.getboolean("AblateSem"):
        for hvec, ix in sorted(hvec_to_ix.items()):
            print("K " + str(hvec) + " [" + ",".join(map(str, hvec_embeds[ix])) + "]")
    for fdec, ix in sorted(fdecs_to_ix.items()):
        print("f " + str(ix) + " " + str(fdec))


if __name__ == "__main__":
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(sys.argv[1])
    for section in config:
        eprint(section, dict(config[section]))
    main(config)
