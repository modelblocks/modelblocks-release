import sys, configparser, torch, re, os
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def prepare_data():
    data = [line.strip() for line in sys.stdin]
    depth, catBase, hvBase, hvFiller, fDecs, hvBFirst, hvFFirst = ([] for _ in range(7))

    for line in data:
        d, cb, hvb, hvf, fd = line.split(" ")
        depth.append(int(d))
        catBase.append(cb)
        hvBase.append(hvb)
        hvFiller.append(hvf)
        fDecs.append(fd)
    eprint("Linesplit complete")

    # Extract first KVec from dense HVec
    for hvec in hvBase:
        match = re.findall(r"^\[(.*?)\]", hvec)
        hvBFirst.append([float(i) for i in match[0].split(",")])
    eprint("hvBase ready")

    for hvec in hvFiller:
        match = re.findall(r"^\[(.*?)\]", hvec)
        hvFFirst.append([float(i) for i in match[0].split(",")])
    eprint("hvFiller ready")

    cat_to_ix = {cat: i for i, cat in enumerate(sorted(set(catBase)))}
    fdecs_to_ix = {fdecs: i for i, fdecs in enumerate(sorted(set(fDecs)))}

    cat_b_ix = [cat_to_ix[cat] for cat in catBase]
    fdecs_ix = [fdecs_to_ix[fdecs] for fdecs in fDecs]

    eprint("Number of output F categories: {}".format(str(len(fdecs_to_ix))))

    return depth, cat_b_ix, hvBFirst, hvFFirst, cat_to_ix, fdecs_ix, fdecs_to_ix


def prepare_data_dev(dev_decpars_file, cat_to_ix, fdecs_to_ix):
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

    for hvec in hvBase:
        match = re.findall(r"^\[(.*?)\]", hvec)
        hvBFirst.append([float(i) for i in match[0].split(",")])

    for hvec in hvFiller:
        match = re.findall(r"^\[(.*?)\]", hvec)
        hvFFirst.append([float(i) for i in match[0].split(",")])

    cat_b_ix = [cat_to_ix[cat] for cat in catBase]
    fdecs_ix = [fdecs_to_ix[fdecs] for fdecs in fDecs]

    return depth, cat_b_ix, hvBFirst, hvFFirst, fdecs_ix


class FModel(nn.Module):
    def __init__(self, cat_vocab_size, syn_size, sem_size, hidden_dim, output_dim):
        super(FModel, self).__init__()
        self.cat_embeds = nn.Embedding(cat_vocab_size, syn_size)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(7+syn_size+2*sem_size, self.hidden_dim, bias=True)
        self.relu = F.relu
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim, bias=True)

    def forward(self, d_onehot, cat_b_ix, hvb, hvf, use_gpu):
        cat_b_embed = self.cat_embeds(cat_b_ix)
        if use_gpu >= 0:
            cat_b_embed = cat_b_embed.to("cuda")
        x = torch.cat((cat_b_embed, hvb, hvf, d_onehot), 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(use_dev, dev_decpars_file, use_gpu, syn_size, sem_size, hidden_dim, num_epochs, batch_size, learning_rate, l2_reg):
    depth, cat_b_ix, hvb, hvf, cat_to_ix, fdecs_ix, fdecs_to_ix = prepare_data()
    depth = F.one_hot(torch.LongTensor(depth), 7).float()
    cat_b_ix = torch.LongTensor(cat_b_ix)
    hvb = torch.FloatTensor(hvb)
    hvf = torch.FloatTensor(hvf)
    target = torch.LongTensor(fdecs_ix)
    model = FModel(len(cat_to_ix), syn_size, sem_size, hidden_dim, len(fdecs_to_ix))

    if use_gpu >= 0:
        depth = depth.to("cuda")
        cat_b_ix = cat_b_ix.to("cuda")
        hvb = hvb.to("cuda")
        hvf = hvf.to("cuda")
        target = target.to("cuda")
        model = model.cuda()

    if use_dev >= 0:
        dev_depth, dev_cat_b_ix, dev_hvb, dev_hvf, dev_fdecs_ix = prepare_data_dev(dev_decpars_file, cat_to_ix, fdecs_to_ix)
        dev_depth = F.one_hot(torch.LongTensor(dev_depth), 7).float()
        dev_cat_b_ix = torch.LongTensor(dev_cat_b_ix)
        dev_hvb = torch.FloatTensor(dev_hvb)
        dev_hvf = torch.FloatTensor(dev_hvf)
        dev_target = torch.LongTensor(dev_fdecs_ix)

        if use_gpu >= 0:
            dev_depth = dev_depth.to("cuda")
            dev_cat_b_ix = dev_cat_b_ix.to("cuda")
            dev_hvb = dev_hvb.to("cuda")
            dev_hvf = dev_hvf.to("cuda")
            dev_target = dev_target.to("cuda")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    criterion = nn.NLLLoss()

    # training loop
    eprint("Start FModel training loop...")
    epoch = 0

    while True:
        model.train()
        epoch += 1
        permutation = torch.randperm(len(depth))
        total_train_correct = 0
        total_train_loss = 0
        total_dev_loss = 0

        for i in range(0, len(depth), batch_size):
            indices = permutation[i:i+batch_size]
            batch_d, batch_c, batch_hvb, batch_hvf, batch_target = depth[indices], cat_b_ix[indices], hvb[indices], hvf[indices], target[indices]
            output = model(batch_d, batch_c, batch_hvb, batch_hvf, use_gpu)
            _, fdec = torch.max(output.data, 1)
            train_correct = (fdec == batch_target).sum().item()
            total_train_correct += train_correct
            loss = criterion(output, batch_target)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if use_dev >= 0:
            with torch.no_grad():
                dev_pred = model(dev_depth, dev_cat_b_ix, dev_hvb, dev_hvf, use_gpu)
                _, dev_fdec = torch.max(dev_pred.data, 1)
                dev_correct = (dev_fdec == dev_target).sum().item()
                dev_loss = criterion(dev_pred, dev_target)
                total_dev_loss += dev_loss.item()
                dev_acc = 100*(dev_correct/len(dev_depth))
        else:
            dev_acc = 0

        eprint("Epoch {:04d} | AvgTrainLoss {:.4f} | TrainAcc {:.4f} | DevLoss {:.4f} | DevAcc {:.4f}".
               format(epoch, total_train_loss/(len(depth)//batch_size), 100*(total_train_correct/len(depth)), total_dev_loss, dev_acc))

        if epoch == num_epochs:
            break

    return model, cat_to_ix, fdecs_to_ix


def main(config):
    f_config = config["FModel"]
    model, cat_to_ix, fdecs_to_ix = train(f_config.getint("Dev"), f_config.get("DevFile"), f_config.getint("GPU"),
                                          f_config.getint("SynSize"), f_config.getint("SemSize"), f_config.getint("HiddenSize"),
                                          f_config.getint("NEpochs"), f_config.getint("BatchSize"),
                                          f_config.getfloat("LearningRate"), f_config.getfloat("Regularization"))

    if f_config.getint("GPU") >= 0:
        cat_embeds = list(model.parameters())[0].data.cpu().numpy()
        first_weights = list(model.parameters())[1].data.cpu().numpy()
        first_biases = list(model.parameters())[2].data.cpu().numpy()
        second_weights = list(model.parameters())[3].data.cpu().numpy()
        second_biases = list(model.parameters())[4].data.cpu().numpy()
    else:
        cat_embeds = list(model.parameters())[0].data.numpy()
        first_weights = list(model.parameters())[1].data.numpy()
        first_biases = list(model.parameters())[2].data.numpy()
        second_weights = list(model.parameters())[3].data.numpy()
        second_biases = list(model.parameters())[4].data.numpy()

    eprint(first_weights.shape, second_weights.shape)
    print("F F " + ",".join(map(str, first_weights.flatten('F').tolist())))
    print("F f " + ",".join(map(str, first_biases.flatten('F').tolist())))
    print("F S " + ",".join(map(str, second_weights.flatten('F').tolist())))
    print("F s " + ",".join(map(str, second_biases.flatten('F').tolist())))
    for cat, ix in sorted(cat_to_ix.items()):
        print("C " + str(cat) + " [" + ",".join(map(str, cat_embeds[ix])) + "]")
    for fdec, ix in sorted(fdecs_to_ix.items()):
        print("f " + str(ix) + " " + str(fdec))


if __name__ == "__main__":
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(sys.argv[1])
    for section in config:
        eprint(section, dict(config[section]))
    main(config)
