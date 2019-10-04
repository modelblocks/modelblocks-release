import sys, re, os
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def prepare_data(decpars_file, use_dev):
    with open(decpars_file, "r") as f:
        data = f.readlines()
        data = [line.strip() for line in data]

    depth, catAncstr, hvAncstr, hvFiller, catLchild, hvLchild, jDecs, hvAFirst, hvFFirst, hvLFirst = ([] for i in range(10))

    for line in data:
        d, ca, hva, hvf, cl, hvl, jd = line.split(" ")
        depth.append(int(d))
        catAncstr.append(ca)
        hvAncstr.append(hva)
        hvFiller.append(hvf)
        catLchild.append(cl)
        hvLchild.append(hvl)
        jDecs.append(jd)
    eprint("linesplit complete")

    for kvec in hvAncstr:
        match = re.findall(r"^\[(.*?)\]", kvec)
        hvAFirst.append([float(i) for i in match[0].split(",")])
    eprint("hvAncstr ready")

    for kvec in hvFiller:
        match = re.findall(r"^\[(.*?)\]", kvec)
        hvFFirst.append([float(i) for i in match[0].split(",")])
    eprint("hvFiller ready")

    for kvec in hvLchild:
        match = re.findall(r"^\[(.*?)\]", kvec)
        hvLFirst.append([float(i) for i in match[0].split(",")])
    eprint("hvLChild ready")

    cat_to_ix = {cat: i for i, cat in enumerate(sorted(set(catAncstr+catLchild)))}
    jdecs_to_ix = {jdecs: i for i, jdecs in enumerate(sorted(set(jDecs)))}

    if use_dev >= 0:
        cat_to_ix["UNK"] = len(cat_to_ix)
        jdecs_to_ix["UNK"] = len(jdecs_to_ix)

    cat_a_ix, cat_l_ix, jdecs_ix = ([] for i in range(3))

    for cat in catAncstr:
        cat_a_ix.append(cat_to_ix[cat])

    for cat in catLchild:
        cat_l_ix.append(cat_to_ix[cat])

    for jdecs in jDecs:
        jdecs_ix.append(jdecs_to_ix[jdecs])

    ix_to_cat = {k: v for v, k in cat_to_ix.items()}
    ix_to_jdecs = {k: v for v, k in jdecs_to_ix.items()}

    eprint("number of output J categories: " + str(len(ix_to_jdecs)))

    return depth, cat_a_ix, hvAFirst, hvFFirst, cat_l_ix, hvLFirst, ix_to_cat, jdecs_ix, ix_to_jdecs


def prepare_data_dev(dev_decpars_file, ix_to_cat, ix_to_jdecs):
    with open(dev_decpars_file, "r") as f:
        data = f.readlines()
        data = [line.strip() for line in data]

    depth, catAncstr, hvAncstr, hvFiller, catLchild, hvLchild, jDecs, hvAFirst, hvFFirst, hvLFirst = ([] for i in range(10))

    for line in data:
        d, ca, hva, hvf, cl, hvl, jd = line.split(" ")
        depth.append(int(d))
        catAncstr.append(ca)
        hvAncstr.append(hva)
        hvFiller.append(hvf)
        catLchild.append(cl)
        hvLchild.append(hvl)
        jDecs.append(jd)

    for kvec in hvAncstr:
        match = re.findall(r"^\[(.*?)\]", kvec)
        hvAFirst.append([float(i) for i in match[0].split(",")])

    for kvec in hvFiller:
        match = re.findall(r"^\[(.*?)\]", kvec)
        hvFFirst.append([float(i) for i in match[0].split(",")])

    for kvec in hvLchild:
        match = re.findall(r"^\[(.*?)\]", kvec)
        hvLFirst.append([float(i) for i in match[0].split(",")])

    cat_to_ix = {k: v for v, k in ix_to_cat.items()}
    jdecs_to_ix = {k: v for v, k in ix_to_jdecs.items()}
    cat_a_ix, cat_l_ix, jdecs_ix = ([] for i in range(3))

    for cat in catAncstr:
        cat_a_ix.append(cat_to_ix.get(cat, cat_to_ix["UNK"]))

    for cat in catLchild:
        cat_l_ix.append(cat_to_ix.get(cat, cat_to_ix["UNK"]))

    for jdecs in jDecs:
        jdecs_ix.append(jdecs_to_ix.get(jdecs, jdecs_to_ix["UNK"]))

    return depth, cat_a_ix, hvAFirst, hvFFirst, cat_l_ix, hvLFirst, jdecs_ix


# train models
class JModel(nn.Module):
    def __init__(self, cat_vocab_size, syn_size, sem_size, hidden_dim, output_dim):
        super(JModel, self).__init__()
        self.cat_embeds = nn.Embedding(cat_vocab_size, syn_size)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(7+2*syn_size+3*sem_size, self.hidden_dim, bias=False)
        self.relu = F.relu
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim, bias=False)

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

# continue here
def train(decpars_file, use_dev, dev_decpars_file, use_gpu, syn_size, sem_size, epochs, batch_size, learning_rate):
    depth, cat_a_ix, hva, hvf, cat_l_ix, hvl, ix_to_cat, jdecs_ix, ix_to_jdecs = prepare_data(decpars_file, use_dev)
    depth = F.one_hot(torch.LongTensor(depth), 7).float()
    cat_a_ix = torch.LongTensor(cat_a_ix)
    hva = torch.FloatTensor(hva)
    hvf = torch.FloatTensor(hvf)
    cat_l_ix = torch.LongTensor(cat_l_ix)
    hvl = torch.FloatTensor(hvl)
    target = torch.LongTensor(jdecs_ix)
    model = JModel(len(ix_to_cat), syn_size, sem_size, 7+2*syn_size+3*sem_size, len(ix_to_jdecs))

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
        dev_depth, dev_cat_a_ix, dev_hva, dev_hvf, dev_cat_l_ix, dev_hvl, dev_jdecs_ix = prepare_data_dev(dev_decpars_file, ix_to_cat, ix_to_jdecs)
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

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(epochs):
        permutation = torch.randperm(len(depth))
        trng_loss = 0
        cum_dev_loss = 0

        for i in range(0, len(depth), batch_size):
            indices = permutation[i:i+batch_size]
            batch_d, batch_a, batch_hva, batch_hvf, batch_l, batch_hvl, batch_target = depth[indices], cat_a_ix[indices], hva[indices], hvf[indices], cat_l_ix[indices], hvl[indices], target[indices]
            output = model(batch_d, batch_a, batch_hva, batch_hvf, batch_l, batch_hvl, use_gpu)
            loss = criterion(output, batch_target)
            trng_loss = loss.item()
           # eprint('Epoch [%d/%d], Batch [%d/%d], Loss: %.4f' % (
           # epoch + 1, epochs, i // batch_size + 1, len(depth) // batch_size + 1, loss.item()))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if use_dev >= 0:
            with torch.no_grad():
                dev_pred = model(dev_depth, dev_cat_a_ix, dev_hva, dev_hvf, dev_cat_l_ix, dev_hvl, use_gpu)
                dev_loss = criterion(dev_pred, dev_target)
                cum_dev_loss += dev_loss.item()

        eprint("Epoch:", epoch+1, "Training loss:", trng_loss, "Dev loss:", cum_dev_loss)

    return model, ix_to_cat, ix_to_jdecs


def main(args):
    model, ix_to_cat, ix_to_jdecs = train(args.decpars_file, args.use_dev, args.dev_decpars_file, args.use_gpu, args.syn_size, args.sem_size, args.epochs, args.batch_size, args.learning_rate)

    if args.use_gpu >= 0:
        cat_embeds = list(model.parameters())[0].data.cpu().numpy()
        first_weights = list(model.parameters())[1].data.cpu().numpy()
        second_weights = list(model.parameters())[2].data.cpu().numpy()
    else:
        cat_embeds = list(model.parameters())[0].data.numpy()
        first_weights = list(model.parameters())[1].data.numpy()
        second_weights = list(model.parameters())[2].data.numpy()

    eprint(first_weights.shape, second_weights.shape)
    print("J F " + ",".join(map(str, first_weights.flatten().tolist())))
    print("J S " + ",".join(map(str, second_weights.flatten('F').tolist())))
    for key in ix_to_cat:
        print("C " + str(ix_to_cat[key]) + " [" + ",".join(map(str, cat_embeds[key])) + "]")
    for key in ix_to_jdecs:
        print("j " + str(key) + " " + str(ix_to_jdecs[key]) + "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='J model trainer')
    parser.add_argument("-train", "--decpars-file", type=str, required=True, help="training data")
    parser.add_argument("--use-dev", type=int, default=-1, help="use validation")
    parser.add_argument("-dev", "--dev-decpars-file", type=str, help="validation data")
    parser.add_argument("--use-gpu", type=int, default=1, help="use gpu")
    parser.add_argument("--syn-size", type=int, default=10, help="syntax size")
    parser.add_argument("--sem-size", type=int, default=20, help="semantics size")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="mini-batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="learning rate")
    args = parser.parse_args()
    eprint(args)
    main(args)
