import sys, configparser, torch, re, os
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def getFirstVec(vecs):
    match = re.findall(r"^\[(.*?)\]", vecs)
    return [float(i) for i in match[0].split(",")]

def prepare_data(devfile=None, cat_to_ix=None):

    if devfile != None: #prepare dev data from file using existing category map
        catBases, catAntes, hvBases, hvAntes, wordDists, sqWordDists, corefOns, labels = ([] for _ in range(8))
        with open(devfile, "r") as f:
            dev_data = f.readlines()
            dev_data = [line.strip() for line in dev_data]
    
        for line in dev_data:
            catBase, catAnte, hvBase, hvAnte, wordDist, sqWordDist, corefOn, label = line.split(" ")
            if catBase not in cat_to_ix or catAnte not in cat_to_ix:
                continue
            catBases.append(catBase)
            catAntes.append(catAnte)
            hvBases.append(getFirstVec(hvBase))
            hvAntes.append(getFirstVec(hvAnte))
            wordDists.append(int(wordDist))
            sqWordDists.append(int(sqWordDist))
            corefOns.append(int(corefOn))
            labels.append(int(label))

        cat_base_ixs = [cat_to_ix[cat] for cat in catBases]
        cat_ante_ixs = [cat_to_ix[cat] for cat in catAntes]

        return cat_to_ix, cat_base_ixs, cat_ante_ixs, hvBases, hvAntes, wordDists, sqWordDists, corefOns, labels

    else: #prepare training data
        #cat_to_ix, cat_base_ixs, cat_ante_ixs, hvBases, hvAntes, wordDists, sqWordDists, corefOns, labels = prepare_data()
        #init empty lists for each field, nest to represent entire dataset 
        #each data point is a pair of base/antecedent features, including syncat, hvec, and 3 additional features, and one binary class label. 3 addtional features are word dist, sqdist, corefon/off.
        data = [line.strip() for line in sys.stdin]
        catBases, catAntes, hvBases, hvAntes, wordDists, sqWordDists, corefOns, labels = ([] for _ in range(8))

        for line in data:
            #populate data into correct list
            catBase, catAnte, hvBase, hvAnte, wordDist, sqWordDist, corefOn, label = line.split(" ")
            catBases.append(catBase)
            catAntes.append(catAnte)
            hvBases.append(getFirstVec(hvBase))
            hvAntes.append(getFirstVec(hvAnte))
            wordDists.append(int(wordDist))
            sqWordDists.append(int(sqWordDist))
            corefOns.append(int(corefOn))
            labels.append(int(label))

        #prepare maps from variables to integers
        allCats = set(catBases).union(set(catAntes))
        cat_to_ix = {cat: i for i, cat in enumerate(sorted(allCats))}
        cat_base_ixs = [cat_to_ix[cat] for cat in catBases]
        cat_ante_ixs = [cat_to_ix[cat] for cat in catAntes]

        return cat_to_ix, cat_base_ixs, cat_ante_ixs, hvBases, hvAntes, wordDists, sqWordDists, corefOns, labels

class NModel(nn.Module):
    def __init__(self, cat_vocab_size, syn_size, sem_size, hidden_dim, output_dim):
        super(NModel, self).__init__()
        self.cat_embeds = nn.Embedding(cat_vocab_size, syn_size)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(2*syn_size+2*sem_size+3, self.hidden_dim, bias=True)
        self.relu = F.relu
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim, bias=True)

    def forward(self, cat_base_ixs, cat_ante_ixs, hvbases, hvantes, worddists, sqworddists, corefons, use_gpu):
        cat_base_embed = self.cat_embeds(cat_base_ixs)
        cat_ante_embed = self.cat_embeds(cat_ante_ixs)
        if use_gpu >= 0:
            cat_base_embed = cat_base_embed.to("cuda")
            cat_ante_embed = cat_ante_embed.to("cuda")
        x = torch.cat((cat_base_embed, cat_ante_embed, hvbases, hvantes, worddists.unsqueeze(dim=1), sqworddists.unsqueeze(dim=1), corefons.unsqueeze(dim=1)), 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(use_dev, dev_decpars_file, use_gpu, syn_size, sem_size, hidden_dim, num_epochs, batch_size, learning_rate, l2_reg, useClassFreqWeighting):
    cat_to_ix, cat_base_ixs, cat_ante_ixs, hvBases, hvAntes, wordDists, sqWordDists, corefOns, labels = prepare_data()
    cat_base_ixs = torch.LongTensor(cat_base_ixs)
    cat_ante_ixs = torch.LongTensor(cat_ante_ixs)
    hvBases = torch.FloatTensor(hvBases)
    hvAntes = torch.FloatTensor(hvAntes)
    wordDists = torch.LongTensor(wordDists)
    sqWordDists = torch.LongTensor(sqWordDists)
    corefOns = torch.LongTensor(corefOns)
    target = torch.LongTensor(labels)
    outputdim = len(set(target.tolist())) 
    assert outputdim == 2
    model = NModel(len(cat_to_ix), syn_size, sem_size, hidden_dim, outputdim) #output_dim is last param

    if use_gpu >= 0:
        cat_base_ixs = cat_base_ixs.to("cuda")
        cat_ante_ixs = cat_ante_ixs.to("cuda")
        hvBases = hvBases.to("cuda")
        hvAntes = hvAntes.to("cuda")
        wordDists = wordDists.to("cuda")
        sqWordDists = sqWordDists.to("cuda")
        corefOns = corefOns.to("cuda")
        target = target.to("cuda")
        model = model.cuda()

    if use_dev >= 0:
        #generate data from training maps and dev data file
        dev_cat_to_ix, dev_cat_base_ixs, dev_cat_ante_ixs, dev_hvBases, dev_hvAntes, dev_wordDists, dev_sqWordDists, dev_corefOns, dev_labels = prepare_data_dev(dev_decpars_file, cat_to_ix)
        dev_cat_base_ixs = torch.LongTensor(dev_cat_base_ixs)
        dev_cat_ante_ixs = torch.LongTensor(dev_cat_ante_ixs)
        dev_hvBases = torch.FloatTensor(dev_hvBases)
        dev_hvAntes = torch.FloatTensor(dev_hvAntes)
        dev_wordDists = torch.LongTensor(dev_wordDists)
        dev_sqWordDists = torch.LongTensor(dev_sqWordDists)
        dev_corefOns = torch.LongTensor(dev_corefOns)
        dev_target = torch.LongTensor(dev_labels)

        if use_gpu >= 0:
            dev_cat_base_ixs = dev_cat_base_ixs.to("cuda")
            dev_cat_ante_ixs = dev_cat_ante_ixs.to("cuda")
            dev_hvBases = dev_hvBases.to("cuda")
            dev_hvAntes = dev_hvAntes.to("cuda")
            dev_wordDists = dev_wordDists.to("cuda")
            dev_sqWordDists = dev_sqWordDists.to("cuda")
            dev_corefOns = dev_corefOns.to("cuda")
            dev_target = dev_target.to("cuda")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    #criterion = nn.NLLLoss()

    eprint("target shape: {}".format(str(target.shape)))

    total = target.shape[0]
    num_neg = total - torch.sum(target, dim=0) 
    wv = [num_neg/total, 1-num_neg/total] if useClassFreqWeighting else None
    eprint("num_neg: {}".format(num_neg))
    eprint("total: {}".format(total))
    #criterion = nn.NLLLoss(weights=wv) 
    criterion = nn.NLLLoss() 

    # training loop
    eprint("Start NModel training loop...")
    epoch = 0

    while True:
        model.train()
        epoch += 1
        permutation = torch.randperm(len(target))
        total_train_correct = 0
        total_train_loss = 0
        total_dev_loss = 0

        for i in range(0, len(target), batch_size):
            indices = permutation[i:i+batch_size]
            #batch_d, batch_c, batch_hvb, batch_hvf, batch_target = depth[indices], cat_b_ix[indices], hvb[indices], hvf[indices], target[indices]
            batch_catbase, batch_catante, batch_hvbase, batch_hvante, batch_worddist, batch_sqworddist, batch_corefon, batch_target = cat_base_ixs[indices], cat_ante_ixs[indices], hvBases[indices], hvAntes[indices], wordDists[indices], sqWordDists[indices], corefOns[indices], target[indices]
            output = model(batch_catbase, batch_catante, batch_hvbase, batch_hvante, batch_worddist.float(), batch_sqworddist.float(), batch_corefon.float(), use_gpu)
            _, ndec = torch.max(output.data, 1)
            train_correct = (ndec == batch_target).sum().item()
            total_train_correct += train_correct
            loss = criterion(output, batch_target)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if use_dev >= 0:
            with torch.no_grad():
                dev_pred = model(dev_cat_base_ixs, dev_cat_ante_ixs, dev_hvBases, dev_hvAntes, dev_wordDists.float(), dev_sqWordDists.float(), dev_corefOns.float(), use_gpu)
                _, dev_ndec = torch.max(dev_pred.data, 1)
                dev_correct = (dev_ndec == dev_target).sum().item()
                dev_loss = criterion(dev_pred, dev_target)
                total_dev_loss += dev_loss.item()
                dev_acc = 100*(dev_correct/len(dev_depth))
        else:
            dev_acc = 0

        eprint("Epoch {:04d} | AvgTrainLoss {:.4f} | TrainAcc {:.4f} | DevLoss {:.4f} | DevAcc {:.4f}".
               format(epoch, total_train_loss/(len(target)//batch_size), 100*(total_train_correct/len(target)), total_dev_loss, dev_acc))

        if epoch == num_epochs:
            break

    return model, cat_to_ix


def main(config):
    n_config = config["NModel"]
    model, cat_to_ix = train(n_config.getint("Dev"), n_config.get("DevFile"), 
                             n_config.getint("GPU"), n_config.getint("SynSize"), 
                             n_config.getint("SemSize"), n_config.getint("HiddenSize"),
                             n_config.getint("NEpochs"), n_config.getint("BatchSize"),
                             n_config.getfloat("LearningRate"), n_config.getfloat("Regularization"),
                             n_config.getint("UseClassFreqWeighting"))

    #get trained weights, biases
    if n_config.getint("GPU") >= 0:
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

    #output trained weights, biases
    eprint(first_weights.shape, second_weights.shape)
    print("N F " + ",".join(map(str, first_weights.flatten('F').tolist())))
    print("N f " + ",".join(map(str, first_biases.flatten('F').tolist())))
    print("N S " + ",".join(map(str, second_weights.flatten('F').tolist())))
    print("N s " + ",".join(map(str, second_biases.flatten('F').tolist())))
    for cat, ix in sorted(cat_to_ix.items()):
        print("C " + str(cat) + " [" + ",".join(map(str, cat_embeds[ix])) + "]")


if __name__ == "__main__":
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(sys.argv[1])
    for section in config:
        eprint(section, dict(config[section]))
    main(config)
