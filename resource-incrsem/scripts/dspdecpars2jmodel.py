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
    depth, catAncstr, hvAncstr, hvFiller, catLchild, hvLchild, jDecs, hvAFirst, hvFFirst, hvLFirst = ([] for _ in range(10))

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

    # Extract first KVec from dense HVec
    for hvec in hvAncstr:
        match = re.findall(r"^\[(.*?)\]", hvec)
        hvAFirst.append([float(i) for i in match[0].split(",")])
    eprint("hvAncstr ready")

    for hvec in hvFiller:
        match = re.findall(r"^\[(.*?)\]", hvec)
        hvFFirst.append([float(i) for i in match[0].split(",")])
    eprint("hvFiller ready")

    for hvec in hvLchild:
        match = re.findall(r"^\[(.*?)\]", hvec)
        hvLFirst.append([float(i) for i in match[0].split(",")])
    eprint("hvLChild ready")

    cat_to_ix = {cat: i for i, cat in enumerate(sorted(set(catAncstr+catLchild)))}
    jdecs_to_ix = {jdecs: i for i, jdecs in enumerate(sorted(set(jDecs)))}

    cat_a_ix = [cat_to_ix[cat] for cat in catAncstr]
    cat_l_ix = [cat_to_ix[cat] for cat in catLchild]
    jdecs_ix = [jdecs_to_ix[jdecs] for jdecs in jDecs]

    eprint("Number of output J categories: {}".format(str(len(jdecs_to_ix))))

    return depth, cat_a_ix, hvAFirst, hvFFirst, cat_l_ix, hvLFirst, cat_to_ix, jdecs_ix, jdecs_to_ix


def prepare_data_dev(dev_decpars_file, cat_to_ix, jdecs_to_ix):
    with open(dev_decpars_file, "r") as f:
        data = f.readlines()
        data = [line.strip() for line in data]

    depth, catAncstr, hvAncstr, hvFiller, catLchild, hvLchild, jDecs, hvAFirst, hvFFirst, hvLFirst = ([] for _ in range(10))

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

    for hvec in hvAncstr:
        match = re.findall(r"^\[(.*?)\]", hvec)
        hvAFirst.append([float(i) for i in match[0].split(",")])

    for hvec in hvFiller:
        match = re.findall(r"^\[(.*?)\]", hvec)
        hvFFirst.append([float(i) for i in match[0].split(",")])

    for hvec in hvLchild:
        match = re.findall(r"^\[(.*?)\]", hvec)
        hvLFirst.append([float(i) for i in match[0].split(",")])

    cat_a_ix = [cat_to_ix[cat] for cat in catAncstr]
    cat_l_ix = [cat_to_ix[cat] for cat in catLchild]
    jdecs_ix = [jdecs_to_ix[jdecs] for jdecs in jDecs]

    return depth, cat_a_ix, hvAFirst, hvFFirst, cat_l_ix, hvLFirst, jdecs_ix


class JModel(nn.Module):
    def __init__(self, cat_vocab_size, syn_size, sem_size, hidden_dim, output_dim):
        super(JModel, self).__init__()
        self.cat_embeds = nn.Embedding(cat_vocab_size, syn_size)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(7+2*syn_size+3*sem_size, self.hidden_dim, bias=True)
        self.relu = F.relu
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim, bias=True)

    def forward(self, d_onehot, cat_a_ix, hva, hvf, cat_l_ix, hvl, use_gpu):
        cat_a_embed = self.cat_embeds(cat_a_ix)
        cat_l_embed = self.cat_embeds(cat_l_ix)
        if use_gpu >= 0:
            cat_a_embed = cat_a_embed.to("cuda")
            cat_l_embed = cat_l_embed.to("cuda")
        x = torch.cat((cat_a_embed, hva, hvf, cat_l_embed, hvl, d_onehot), 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(use_dev, dev_decpars_file, use_gpu, syn_size, sem_size, hidden_dim, num_epochs, batch_size, learning_rate, l2_reg):
    depth, cat_a_ix, hva, hvf, cat_l_ix, hvl, cat_to_ix, jdecs_ix, jdecs_to_ix = prepare_data()
    depth = F.one_hot(torch.LongTensor(depth), 7).float()
    cat_a_ix = torch.LongTensor(cat_a_ix)
    hva = torch.FloatTensor(hva)
    hvf = torch.FloatTensor(hvf)
    cat_l_ix = torch.LongTensor(cat_l_ix)
    hvl = torch.FloatTensor(hvl)
    target = torch.LongTensor(jdecs_ix)
    model = JModel(len(cat_to_ix), syn_size, sem_size, hidden_dim, len(jdecs_to_ix))

    if use_gpu >= 0:
        depth = depth.to("cuda")
        cat_a_ix = cat_a_ix.to("cuda")
        hva = hva.to("cuda")
        hvf = hvf.to("cuda")
        cat_l_ix = cat_l_ix.to("cuda")
        hvl = hvl.to("cuda")
        target = target.to("cuda")
        model = model.cuda()

    if use_dev >= 0:
        dev_depth, dev_cat_a_ix, dev_hva, dev_hvf, dev_cat_l_ix, dev_hvl, dev_jdecs_ix = prepare_data_dev(dev_decpars_file, cat_to_ix, jdecs_to_ix)
        dev_depth = F.one_hot(torch.LongTensor(dev_depth), 7).float()
        dev_cat_a_ix = torch.LongTensor(dev_cat_a_ix)
        dev_hva = torch.FloatTensor(dev_hva)
        dev_hvf = torch.FloatTensor(dev_hvf)
        dev_cat_l_ix = torch.LongTensor(dev_cat_l_ix)
        dev_hvl = torch.FloatTensor(dev_hvl)
        dev_target = torch.LongTensor(dev_jdecs_ix)

        if use_gpu >= 0:
            dev_depth = dev_depth.to("cuda")
            dev_cat_a_ix = dev_cat_a_ix.to("cuda")
            dev_hva = dev_hva.to("cuda")
            dev_hvf = dev_hvf.to("cuda")
            dev_cat_l_ix = dev_cat_l_ix.to("cuda")
            dev_hvl = dev_hvl.to("cuda")
            dev_target = dev_target.to("cuda")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    criterion = nn.NLLLoss()

    # training loop
    eprint("Start JModel training loop...")
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
            batch_d, batch_a, batch_hva, batch_hvf, batch_l, batch_hvl, batch_target = depth[indices], cat_a_ix[indices], hva[indices], hvf[indices], cat_l_ix[indices], hvl[indices], target[indices]
            output = model(batch_d, batch_a, batch_hva, batch_hvf, batch_l, batch_hvl, use_gpu)
            _, jdec = torch.max(output.data, 1)
            train_correct = (jdec == batch_target).sum().item()
            total_train_correct += train_correct
            loss = criterion(output, batch_target)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if use_dev >= 0:
            with torch.no_grad():
                dev_pred = model(dev_depth, dev_cat_a_ix, dev_hva, dev_hvf, dev_cat_l_ix, dev_hvl, use_gpu)
                _, dev_fdec = torch.max(dev_pred.data, 1)
                dev_correct = (dev_fdec == dev_target).sum().item()
                dev_loss = criterion(dev_pred, dev_target)
                total_dev_loss += dev_loss.item()
                dev_acc = 100 * (dev_correct / len(dev_depth))
        else:
            dev_acc = 0

        eprint("Epoch {:04d} | AvgTrainLoss {:.4f} | TrainAcc {:.4f} | DevLoss {:.4f} | DevAcc {:.4f}".
               format(epoch, total_train_loss/(len(depth)//batch_size), 100*(total_train_correct/len(depth)), total_dev_loss, dev_acc))

        if epoch == num_epochs:
            break

    return model, cat_to_ix, jdecs_to_ix


def main(config):
    j_config = config["JModel"]
    model, cat_to_ix, jdecs_to_ix = train(j_config.getint("Dev"), j_config.get("DevFile"), j_config.getint("GPU"),
                                          j_config.getint("SynSize"), j_config.getint("SemSize"), j_config.getint("HiddenSize"),
                                          j_config.getint("NEpochs"), j_config.getint("BatchSize"),
                                          j_config.getfloat("LearningRate"), j_config.getfloat("Regularization"))

    if j_config.getint("GPU") >= 0:
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
    print("J F " + ",".join(map(str, first_weights.flatten('F').tolist())))
    print("J f " + ",".join(map(str, first_biases.flatten('F').tolist())))
    print("J S " + ",".join(map(str, second_weights.flatten('F').tolist())))
    print("J s " + ",".join(map(str, second_biases.flatten('F').tolist())))
    for cat, ix in sorted(cat_to_ix.items()):
        print("C " + str(cat) + " [" + ",".join(map(str, cat_embeds[ix])) + "]")
    for jdec, ix in sorted(jdecs_to_ix.items()):
        print("j " + str(ix) + " " + str(jdec))


if __name__ == "__main__":
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(sys.argv[1])
    for section in config:
        eprint(section, dict(config[section]))
    main(config)
