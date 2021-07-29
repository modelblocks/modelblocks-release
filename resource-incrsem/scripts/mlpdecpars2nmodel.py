import sys, configparser, torch, re, os, time
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.sparse import csr_matrix
import pdb
np.set_printoptions(threshold=sys.maxsize)

BASEKVOCABSIZE = 0
ANTEKVOCABSIZE = 0

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def ensure_binary(data):
    '''confirms two labels are present.  if not, adds 1 additional training example with label flipped'''
    labels = set()
    for line in data:
        try:
            _, _, _, _, _, _, _, _, label = line.split(" ")
        except:
            eprint("WARNING: mlpdecpars spec not observed: {}".format(line))
            continue
        labels.add(label)
    if len(labels) == 1:
        eprint("only one label found, adding fake training example...")
        to_flip = data[0]
        if to_flip[-1] == "1":
            new_example = to_flip[:-1]+"0"
        elif to_flip[-1] == "0":
            new_example = to_flip[:-1]+"1"
        else:
            eprint("WARNING: data format not supported!")
        eprint("adding: {}".format(new_example))
        data.insert(0,new_example) #prepend new training example to data
    return data


def prepare_data():
    data = [line.strip() for line in sys.stdin]
    data = ensure_binary(data)

    catBases, catAntes, hvBases, hvAntes, hvBaseFirsts, hvAnteFirsts, wordDists, sqWordDists, corefOns, labels = ([] for _ in range(10))
    #depth, catBase, hvBase, hvFiller, fDecs, hvBFirst, hvFFirst = ([] for _ in range(8))
    for line in data:
        try:
            catBase, catAnte, hvBase, hvAnte, wordDist, sqWordDist, corefOn, _, label = line.split(" ")
        except:
            eprint("unspec line: {}".format(line))
            continue
            #raise Exception("out of spec line in input")
        #d, cb, hvb, hvf, fd = line.split(" ")
        #depth.append(int(d))
        #catBase.append(cb)
        #hvBase.append(hvb)
        #hvFiller.append(hvf)
        #fDecs.append(fd)
        catBases.append(catBase)
        catAntes.append(catAnte)
        hvBases.append(hvBase)
        hvAntes.append(hvAnte)
        wordDists.append(int(wordDist))
        sqWordDists.append(int(sqWordDist))
        corefOns.append(int(corefOn))
        labels.append(int(label))

    eprint("Linesplit complete")
    # Extract first KVec from sparse HVec
    for hvec in hvBases:
        match = re.findall(r"^\[(.*?)\]", hvec)
        hvBaseFirsts.append(match[0].split(","))
    eprint("hvBaseFirsts ready")
    global BASEKVOCABSIZE 
    BASEKVOCABSIZE = len(hvBaseFirsts)

    for hvec in hvAntes:
        match = re.findall(r"^\[(.*?)\]", hvec)
        hvAnteFirsts.append(match[0].split(","))
    eprint("hvAnteFirsts ready")
    global ANTEKVOCABSIZE
    ANTEKVOCABSIZE = len(hvAnteFirsts)

    # Mapping from category & HVec to index
    flat_hvB = [hvec for sublist in hvBaseFirsts for hvec in sublist if hvec not in ["", "Bot", "Top"]]
    flat_hvA = [hvec for sublist in hvAnteFirsts for hvec in sublist if hvec not in ["", "Bot", "Top"]]
    allCats = set(catBases).union(set(catAntes))
    cat_to_ix = {cat: i for i, cat in enumerate(sorted(set(allCats)))}
    
    #fdecs_to_ix = {fdecs: i for i, fdecs in enumerate(sorted(set(fDecs)))}
    hvec_to_ix = {hvec: i for i, hvec in enumerate(sorted(set(flat_hvB + flat_hvA)))}

    cat_base_ixs = [cat_to_ix[cat] for cat in catBases]
    cat_ante_ixs = [cat_to_ix[cat] for cat in catAntes]
    #fdecs_ix = [fdecs_to_ix[fdecs] for fdecs in fDecs]

    hvb_row, hvb_col, hvb_top, hva_row, hva_col, hva_top = ([] for _ in range(6))

    # KVec index sparse matrix and "Top" KVec counts
    for i, sublist in enumerate(hvBaseFirsts):
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
                         shape=(len(hvBaseFirsts), len(hvec_to_ix)))
    eprint("hvb_mat ready")

    for i, sublist in enumerate(hvAnteFirsts):
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
                         shape=(len(hvAnteFirsts), len(hvec_to_ix)))
    eprint("hva_mat ready")

    eprint("Number of input KVecs: {}".format(len(hvec_to_ix)))
    #eprint("Number of output F categories: {}".format(len(fdecs_to_ix)))

    return cat_to_ix, cat_base_ixs, cat_ante_ixs, hvb_mat, hva_mat, hvec_to_ix, hvb_top, hva_top, wordDists, sqWordDists, corefOns, labels 
    #return depth, cat_b_ix, hvb_mat, hvf_mat, cat_to_ix, fdecs_ix, fdecs_to_ix, hvec_to_ix, hvb_top, hvf_top


def prepare_data_dev(dev_decpars_file, cat_to_ix, hvec_to_ix):
    with open(dev_decpars_file, "r") as f:
        data = f.readlines()
        data = [line.strip() for line in data]

    catBases, catAntes, hvBases, hvAntes, hvBaseFirsts, hvAnteFirsts, wordDists, sqWordDists, corefOns, labels = ([] for _ in range(10))
    eprint("finished reading dev data. beginning processing...")
    for line in data:
        catBase, catAnte, hvBase, hvAnte, wordDist, sqWordDist, corefOn, _, label = line.split(" ")
        if catBase not in cat_to_ix or catAnte not in cat_to_ix:
            continue
        catBases.append(catBase)
        catAntes.append(catAnte)
        hvBases.append(hvBase)
        hvAntes.append(hvAnte)
        wordDists.append(int(wordDist))
        sqWordDists.append(int(sqWordDist))
        corefOns.append(int(corefOn))
        labels.append(int(label))

    for kvec in hvBases:
        match = re.findall(r"^\[(.*?)\]", kvec)
        hvBaseFirsts.append(match[0].split(","))

    for kvec in hvAntes:
        match = re.findall(r"^\[(.*?)\]", kvec)
        hvAnteFirsts.append(match[0].split(","))

    cat_b_ix = [cat_to_ix[cat] for cat in catBases]
    cat_a_ix = [cat_to_ix[cat] for cat in catAntes]

    hvb_row, hvb_col, hva_row, hva_col, hvb_top, hva_top = ([] for _ in range(6))

    # KVec indices and "Top" KVec counts
    for i, sublist in enumerate(hvBaseFirsts):
        top_count = 0
        for hvec in sublist:
            if hvec == "Top":
                top_count += 1
            elif hvec == "" or hvec == "Bot" or hvec not in hvec_to_ix:
                continue
            else:
                hvb_row.append(i)
                hvb_col.append(hvec_to_ix[hvec]) 
        hvb_top.append([top_count])
    hvb_mat = csr_matrix((np.ones(len(hvb_row), dtype=np.int32), (hvb_row, hvb_col)),
                         shape=(len(hvBaseFirsts), len(hvec_to_ix)))

    for i, sublist in enumerate(hvAnteFirsts):
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
                         shape=(len(hvAnteFirsts), len(hvec_to_ix)))

    return cat_b_ix, cat_a_ix, hvb_mat, hva_mat, hvb_top, hva_top, wordDists, sqWordDists, corefOns, labels 


class NModel(nn.Module):

    def __init__(self, cat_vocab_size, hvec_vocab_size, syn_size, sem_size, hidden_dim, output_dim, dropout_prob):
        super(NModel, self).__init__()
        self.hvec_vocab_size = hvec_vocab_size
        self.sem_size = sem_size
        self.cat_embeds = nn.Embedding(cat_vocab_size, syn_size)
        self.hvec_embeds = nn.Embedding(hvec_vocab_size, sem_size)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        self.fc1 = nn.Linear(2*syn_size+2*sem_size+3, self.hidden_dim, bias=True)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.relu = F.relu
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim, bias=True)

    def forward(self, cat_base_ixs, cat_ante_ixs, hvb_mat, hva_mat, hvb_top, hva_top, worddists, sqworddists, corefons, use_gpu, ablate_sem):
    #def forward(self, d_onehot, cat_b_ix, hvb_mat, hvf_mat, hvb_top, hvf_top, use_gpu, ablate_sem):
        cat_base_embed = self.cat_embeds(cat_base_ixs)
        cat_ante_embed = self.cat_embeds(cat_ante_ixs)
        hvb_top = torch.FloatTensor(hvb_top)
        hva_top = torch.FloatTensor(hva_top)

        if use_gpu > 0:
            cat_base_embed = cat_base_embed.to("cuda")
            cat_ante_embed = cat_ante_embed.to("cuda")
            hvb_top = hvb_top.to("cuda")
            hva_top = hva_top.to("cuda")

        if ablate_sem:
            hvb_embed = torch.zeros([hvb_top.shape[0], self.sem_size], dtype=torch.float) + hvb_top
            hva_embed = torch.zeros([hva_top.shape[0], self.sem_size], dtype=torch.float) + hvf_top

        else:
            hvb_mat = hvb_mat.tocoo()
            hvb_mat = torch.sparse.FloatTensor(torch.LongTensor([hvb_mat.row.tolist(), hvb_mat.col.tolist()]),
                                               torch.FloatTensor(hvb_mat.data.astype(np.float32)),
                                               torch.Size(hvb_mat.shape))
            hva_mat = hva_mat.tocoo()
            hva_mat = torch.sparse.FloatTensor(torch.LongTensor([hva_mat.row.tolist(), hva_mat.col.tolist()]),
                                               torch.FloatTensor(hva_mat.data.astype(np.float32)),
                                               torch.Size(hva_mat.shape))
            if use_gpu > 0:
                hvb_mat = hvb_mat.to("cuda")
                hva_mat = hva_mat.to("cuda")

            hvb_embed = torch.sparse.mm(hvb_mat, self.hvec_embeds.weight) + hvb_top
            hva_embed = torch.sparse.mm(hva_mat, self.hvec_embeds.weight) + hva_top

            if use_gpu > 0:
                hvb_embed = hvb_embed.to("cuda")
                hva_embed = hva_embed.to("cuda")

        x = torch.cat((cat_base_embed, cat_ante_embed, hvb_embed, hva_embed, worddists.unsqueeze(dim=1), sqworddists.unsqueeze(dim=1), corefons.unsqueeze(dim=1)), 1)

        np.set_printoptions(threshold=sys.maxsize)
        torch.set_printoptions(threshold=sys.maxsize)
        #eprint(x)
        #eprint(hvb_mat[0,:])
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

def print_examples(cat_to_ix, cat_base_ixs, cat_ante_ixs, hvb_mat, hva_mat, hvec_to_ix, hvb_top, hva_top, wordDists, sqWordDists, corefOns, labels):
   '''WARNING: don't try to do this for a big dataset - it will run out of memory trying to change sparse to full matrix'''
   ix_to_cat = {v: k for k, v in cat_to_ix.items()} 
   ix_to_hvec = {v: k for k, v in hvec_to_ix.items()} 
   ex_idxs = range(len(cat_base_ixs))
   eprint("example data:")
   for ex_idx in ex_idxs:
       eprint("base_ix: {} base_cat: {} ante_ix: {} ante_cat: {} worddist: {} sqworddist: {} corefon: {} label: {}".format(cat_base_ixs[ex_idx],ix_to_cat[cat_base_ixs[ex_idx]], cat_ante_ixs[ex_idx], ix_to_cat[cat_ante_ixs[ex_idx]], wordDists[ex_idx], sqWordDists[ex_idx], corefOns[ex_idx], labels[ex_idx]))
       eprint("  base hvecs: {}".format(",".join([ ix_to_hvec[i] for i,val in enumerate(hvb_mat.toarray()[ex_idx]) if val > 0 ])))
       eprint("  antecedent hvecs: {}".format(",".join([ ix_to_hvec[i] for i,val in enumerate(hva_mat.toarray()[ex_idx]) if val > 0 ])))
       #eprint("hvec b: {} hvec a: {}".format([ix_to_hvec[x] for x in hvb_mat[ex_idx] if x != 0], [ix_to_hvec[x] for x in hva_mat[ex_idx] if x != 0]))

def train(use_dev, dev_decpars_file, use_gpu, syn_size, sem_size, hidden_dim, dropout_prob,
          num_epochs, batch_size, learning_rate, weight_decay, l2_reg, 
          ablate_sem, useClassFreqWeighting):
    #depth, cat_b_ix, hvb_mat, hvf_mat, cat_to_ix, fdecs_ix, fdecs_to_ix, hvec_to_ix, hvb_top, hvf_top = prepare_data()
    cat_to_ix, cat_base_ixs, cat_ante_ixs, hvb_mat, hva_mat, hvec_to_ix, hvb_top, hva_top, wordDists, sqWordDists, corefOns, labels = prepare_data() 
    #cat_to_ix, cat_base_ixs, cat_ante_ixs, hvb_mat, hva_mat, hvec_to_ix, hvb_top, hva_top, wordDists, sqWordDists, corefOns, labels = prepare_data() 
    #print_examples( cat_to_ix, cat_base_ixs, cat_ante_ixs, hvb_mat, hva_mat, hvec_to_ix, hvb_top, hva_top, wordDists, sqWordDists, corefOns, labels)
    
    #depth = F.one_hot(torch.LongTensor(depth), 7).float()
    #cat_b_ix = torch.LongTensor(cat_b_ix)
    #target = torch.LongTensor(fdecs_ix)
    #model = FModel(len(cat_to_ix), len(hvec_to_ix), syn_size, sem_size, hidden_dim, len(fdecs_to_ix))
    cat_base_ixs = torch.LongTensor(cat_base_ixs)
    cat_ante_ixs = torch.LongTensor(cat_ante_ixs)
    #hvBases = torch.FloatTensor(hvBases)
    #hvAntes = torch.FloatTensor(hvAntes)
    wordDists = torch.LongTensor(wordDists)
    sqWordDists = torch.LongTensor(sqWordDists)
    corefOns = torch.LongTensor(corefOns)
    target = torch.LongTensor(labels)
    outputdim = len(set(target.tolist())) 
    assert outputdim == 2
    model = NModel(len(cat_to_ix), len(hvec_to_ix), syn_size, sem_size, hidden_dim, outputdim, dropout_prob) 

    if use_gpu > 0:
        #depth = depth.to("cuda")
        cat_base_ixs = cat_base_ixs.to("cuda")
        cat_ante_ixs = cat_ante_ixs.to("cuda")
        #hvBases = hvBases.to("cuda")
        #hvAntes = hvAntes.to("cuda")
        wordDists = wordDists.to("cuda")
        sqWordDists = sqWordDists.to("cuda")
        corefOns = corefOns.to("cuda")
        target = target.to("cuda")
        model = model.cuda()
        #cat_base_ixs = cat_base_ixs.to("cuda")
        #target = target.to("cuda")
        #model = model.cuda()

    if use_dev > 0:
        #dev_depth, dev_cat_b_mat, dev_hvb_mat, dev_hvf_mat, dev_fdecs_ix, dev_hvb_top, dev_hvf_top = prepare_data_dev(
        #    dev_decpars_file, cat_to_ix, fdecs_to_ix, hvec_to_ix)
        dev_cat_b_ix, dev_cat_a_ix, dev_hvb_mat, dev_hva_mat, dev_hvb_top, dev_hva_top, dev_worddists, dev_sqworddists, dev_corefons, dev_labels = prepare_data_dev(dev_decpars_file, cat_to_ix, hvec_to_ix)
        #dev_depth = F.one_hot(torch.LongTensor(dev_depth), 7).float()
        dev_cat_b_ix = torch.LongTensor(dev_cat_b_ix) #TODO fix dev_cat_b_ix not assigned yet
        dev_cat_a_ix = torch.LongTensor(dev_cat_a_ix)
        dev_worddists = torch.LongTensor(dev_worddists)
        dev_sqworddists = torch.LongTensor(dev_sqworddists)
        dev_corefons = torch.LongTensor(dev_corefons)
        dev_target = torch.LongTensor(dev_labels)

        if use_gpu > 0:
            dev_cat_b_ix = dev_cat_b_ix.to("cuda")
            dev_cat_a_ix = dev_cat_a_ix.to("cuda")
            dev_worddists = dev_worddists.to("cuda")
            dev_sqworddists = dev_sqworddists.to("cuda")
            dev_corefons = dev_corefons.to("cuda")
            dev_target = dev_target.to("cuda")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #TODO implement useClassFreqWeighting
    criterion = nn.NLLLoss()

    # training loop
    eprint("Start NModel training...")
    epoch = 0

    while True:
        c0 = time.time()
        model.train()
        epoch += 1
        permutation = torch.randperm(len(target))
        total_train_correct = 0
        total_train_loss = 0
        total_dev_loss = 0

        for i in range(0, len(target), batch_size):
            indices = permutation[i:i + batch_size]
            #indices = range(i,i+batch_size)
            #eprint("using indices: {} to {}".format(i,i+batch_size))

            batch_catbase, batch_catante, batch_worddist, batch_sqworddist, batch_corefon, batch_target = cat_base_ixs[indices], cat_ante_ixs[indices], wordDists[indices], sqWordDists[indices], corefOns[indices], target[indices]
            #batch_d, batch_c, batch_target = depth[indices], cat_b_ix[indices], target[indices]
            batch_hvb_mat, batch_hva_mat = hvb_mat[np.array(indices), :], hva_mat[np.array(indices), :]
            batch_hvb_top, batch_hva_top = [hvb_top[i] for i in indices], [hva_top[i] for i in indices]
            if use_gpu > 0:
                l2_loss = torch.cuda.FloatTensor([0])
            else:
                l2_loss = torch.FloatTensor([0])
            for param in model.parameters():
                if torch.numel(param) == 0:
                    continue
                l2_loss += torch.mean(param.pow(2))

            #output = model(batch_d, batch_c, batch_hvb_mat, batch_hvf_mat, batch_hvb_top, batch_hvf_top, use_gpu,
            output = model(batch_catbase, batch_catante, batch_hvb_mat, 
                           batch_hva_mat, batch_hvb_top, batch_hva_top, 
                           batch_worddist.float(), batch_sqworddist.float(), 
                           batch_corefon.float(), use_gpu, ablate_sem)
            _, ndec = torch.max(output.data, 1)
            train_correct = (ndec == batch_target).sum().item()
            total_train_correct += train_correct
            nll_loss = criterion(output, batch_target)
            loss = nll_loss + l2_reg * l2_loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if use_dev > 0:
            with torch.no_grad():
                #dev_pred = model(dev_depth, dev_cat_b_ix, dev_hvb_mat, dev_hvf_mat, dev_hvb_top, dev_hvf_top, use_gpu,
                dev_pred = model(dev_cat_b_ix, dev_cat_a_ix, dev_hvb_mat, 
                                 dev_hva_mat, dev_hvb_top, dev_hva_top, 
                                 dev_worddists.float(), dev_sqworddists.float(),
                                 dev_corefons.float(), use_gpu, ablate_sem)
                _, dev_ndec = torch.max(dev_pred.data, 1)
                dev_correct = (dev_ndec == dev_target).sum().item()
                dev_loss = criterion(dev_pred, dev_target)
                total_dev_loss += dev_loss.item()
                dev_acc = 100 * (dev_correct / len(dev_target))
        else:
            dev_acc = 0

        eprint("Epoch {:04d} | AvgTrainLoss {:.4f} | TrainAcc {:.4f} | DevLoss {:.4f} | DevAcc {:.4f} | Time {:.4f}".
               format(epoch, total_train_loss / ((len(target) // batch_size) + 1), 100 * (total_train_correct / len(target)),
                      total_dev_loss, dev_acc, time.time() - c0))

        if epoch == num_epochs:
            break

    #return model, cat_to_ix, fdecs_to_ix, hvec_to_ix
    #print batch cat and hvembed and dist feats

    return model, cat_to_ix, hvec_to_ix


def main(config):
    n_config = config["NModel"]
    model, cat_to_ix, hvec_to_ix = train(n_config.getint("Dev"), 
                                   n_config.get("DevFile"), 
                                   n_config.getint("GPU"), 
                                   n_config.getint("SynSize"), 
                                   n_config.getint("SemSize"), 
                                   n_config.getint("HiddenSize"), 
                                   n_config.getfloat("DropoutProb"),
                                   n_config.getint("NEpochs"), 
                                   n_config.getint("BatchSize"), 
                                   n_config.getfloat("LearningRate"), 
                                   n_config.getfloat("WeightDecay"), 
                                   n_config.getfloat("L2Reg"), 
                                   n_config.getboolean("AblateSem"),
                                   n_config.getboolean("UseClassFreqWeighting")
)

    if n_config.getint("GPU") >= 0:
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
    print("N F " + ",".join(map(str, first_weights.flatten('F').tolist())))
    print("N f " + ",".join(map(str, first_biases.flatten('F').tolist())))
    print("N S " + ",".join(map(str, second_weights.flatten('F').tolist())))
    print("N s " + ",".join(map(str, second_biases.flatten('F').tolist())))
    for cat, ix in sorted(cat_to_ix.items()):
        print("C " + str(cat) + " " + ",".join(map(str, cat_embeds[ix])))
    if not n_config.getboolean("AblateSem"):
        for hvec, ix in sorted(hvec_to_ix.items()):
            print("K " + str(hvec) + " " + ",".join(map(str, hvec_embeds[ix])))
    #for fdec, ix in sorted(fdecs_to_ix.items()):
    #    print("f " + str(ix) + " " + str(fdec))

    '''
    #run an arbitrary forward pass on trained model
    model.eval()
    #data = torch.randn(1, 3, 24, 24) # Load your data here, this is just dummy data
    #data=np.array([[10,45],[11,12],[4,1]])
    data = np.array([[0,0]])
    rows = data[:,0]
    cols = data[:,1]
    
    #hva_mat = csr_matrix((np.ones(len(hva_row), dtype=np.int32), (hva_row, hva_col)), shape=(len(hvAnteFirsts), len(hvec_to_ix)))
    eprint("length of hvec_to_ix: "+ str(len(hvec_to_ix)))
    #eprint("basekvocabsize: {}".format(BASEKVOCABSIZE))
    #eprint("antekvocabsize: {}".format(ANTEKVOCABSIZE))
    eprint(data)
    eprint(rows)
    eprint(cols)
    #emptycsrbase = csr_matrix((np.zeros(len(rows),dtype=np.int32), (rows,cols)), shape=(BASEKVOCABSIZE, len(hvec_to_ix))) #batch x kvocab
    #emptycsrante = csr_matrix((np.zeros(len(rows),dtype=np.int32), (rows,cols)), shape=(ANTEKVOCABSIZE, len(hvec_to_ix))) #batch x kvocab
    emptycsr = csr_matrix((np.zeros(len(rows),dtype=np.int32), (rows,cols)), shape=(1, len(hvec_to_ix))) #batch x kvocab
    #output = model([cat_to_ix["T"]],[cat_to_ix["T"]], emptycsrbase, emptycsrante, [0], [0], [0], [0], [0], False,False)
    zero = torch.LongTensor([0])
    zerofloat = torch.FloatTensor([0])
    output = model(torch.LongTensor([cat_to_ix["V-aN"]]),torch.LongTensor([cat_to_ix["T"]]), emptycsr, emptycsr, zerofloat, zerofloat, zerofloat, zerofloat, zerofloat, -1, False)
    #output = model(torch.LongTensor([cat_to_ix["T"]]),torch.LongTensor([cat_to_ix["T"]]), emptycsr, emptycsr, zerofloat, zerofloat, zerofloat, zerofloat, zerofloat, -1, False)
    #def forward(self, cat_base_ixs, cat_ante_ixs, hvb_mat, hva_mat, hvb_top, hva_top, worddists, sqworddists, corefons, use_gpu, ablate_sem):
    #output = model(data)
    prediction = torch.argmax(output)
    eprint("V-aN,T,bot,bot,0,0,0 output on trained model: ")
    #eprint("T,T,bot,bot,0,0,0 output on trained model: ")
    eprint(output)
    eprint("prediction: ")
    eprint(prediction)
    '''

if __name__ == "__main__":
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(sys.argv[1])
    for section in config:
        eprint(section, dict(config[section]))
    main(config)
