import sys, configparser, torch, re, os, time, random, math
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F
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


def prepare_data():
    data = [line.strip() for line in sys.stdin]
    depth, fdec, jdec, ex, op1, op2, catl, cata, catb = ([] for _ in range(9))

    for line in data:
        d, f, j, e, o1, o2, cl, ca, _, cb = line.split(" ")
        depth.append(int(d))
        fdec.append(int(f))
        jdec.append(int(j))
        ex.append(e)
        op1.append(o1)
        op2.append(o2)
        catl.append(cl)
        cata.append(ca)
        catb.append(cb)
    eprint("Training file processing complete")

    # Mapping from operators & categories to index
    ex_to_ix = {ex: i for i, ex in enumerate(sorted(set(ex)))}
    op1_to_ix = {op: i for i, op in enumerate(sorted(set(op1)))}
    op2_to_ix = {op: i for i, op in enumerate(sorted(set(op2)))}
    catl_to_ix = {cat: i for i, cat in enumerate(sorted(set(catl)))}
    cata_to_ix = {cat: i for i, cat in enumerate(sorted(set(cata)))}
    catb_to_ix = {cat: i for i, cat in enumerate(sorted(set(catb)))}

    eprint("Number of in-place operators: {}".format(len(ex_to_ix)))
    eprint("Number of left-child operators: {}".format(len(op1_to_ix)))
    eprint("Number of right-child operators: {}".format(len(op2_to_ix)))
    eprint("Number of input left-child CVecs: {}".format(len(catl_to_ix)))
    eprint("Number of input ancestor CVecs: {}".format(len(cata_to_ix)))
    eprint("Number of output base CVecs: {}".format(len(catb_to_ix)))

    return (depth, fdec, jdec, ex, op1, op2, catl, cata, catb, ex_to_ix, op1_to_ix, op2_to_ix, catl_to_ix, cata_to_ix, catb_to_ix)


def prepare_data_dev(dev_decpars_file, ex_to_ix, op1_to_ix, op2_to_ix, catl_to_ix, cata_to_ix, catb_to_ix):
    with open(dev_decpars_file, "r") as f:
        data = f.readlines()
        data = [line.strip() for line in data]

    depth, fdec, jdec, ex, op1, op2, catl, cata, catb = ([] for _ in range(9))
    for line in data:
        d, f, j, e, o1, o2, cl, ca, _, cb = line.split(" ")
        if e not in ex_to_ix or o1 not in op1_to_ix or o2 not in op2_to_ix or cl not in catl_to_ix or \
                ca not in cata_to_ix or cb not in catb_to_ix:
            eprint("Unseen operator or syntactic category in dev file ({}, {}, {}, {}, {}, {})!".format(e, o1, o2, cl, ca, cb))
            continue
        else:
            depth.append(int(d))
            fdec.append(int(f))
            jdec.append(int(j))
            ex.append(e)
            op1.append(o1)
            op2.append(o2)
            catl.append(cl)
            cata.append(ca)
            catb.append(cb)
    eprint("Dev file processing complete")

    return (depth, fdec, jdec, ex, op1, op2, catl, cata, catb)


def create_one_batch(training_data, ex_to_ix, op1_to_ix, op2_to_ix, catl_to_ix, cata_to_ix, catb_to_ix):
    depth, fdec, jdec, ex, op1, op2, catl, cata, catb = training_data
    depth_ix = F.one_hot(torch.LongTensor(depth), 7).float()
    fdec_ix = torch.FloatTensor(fdec).view(-1, 1)
    jdec_ix = torch.FloatTensor(jdec).view(-1, 1)
    ex_ix = [ex_to_ix[e] for e in ex]
    ex_ix = torch.LongTensor(ex_ix)
    op1_ix = [op1_to_ix[o] for o in op1]
    op1_ix = torch.LongTensor(op1_ix)
    op2_ix = [op2_to_ix[o] for o in op2]
    op2_ix = torch.LongTensor(op2_ix)
    catl_ix = [catl_to_ix[c] for c in catl]
    catl_ix = torch.LongTensor(catl_ix)
    cata_ix = [cata_to_ix[c] for c in cata]
    cata_ix = torch.LongTensor(cata_ix)
    catb_ix = [catb_to_ix[c] for c in catb]
    catb_ix = torch.LongTensor(catb_ix)

    return depth_ix, fdec_ix, jdec_ix, ex_ix, op1_ix, op2_ix, catl_ix, cata_ix, catb_ix


class BModel(nn.Module):
    def __init__(self,
        device,
        ex_vocab_size,
        op1_vocab_size,
        op2_vocab_size,
        op_dim,
        catl_vocab_size,
        cata_vocab_size,
        syn_dim,
        ff_input_dropout,
        ff_hidden_dim,
        ff_hidden_dropout,
        ff_output_dim):

        super(BModel, self).__init__()
        self.device = device
        self.ex_embeds = nn.Embedding(ex_vocab_size, op_dim)
        self.op1_embeds = nn.Embedding(op1_vocab_size, op_dim)
        self.op2_embeds = nn.Embedding(op2_vocab_size, op_dim)
        self.catl_embeds = nn.Embedding(catl_vocab_size, syn_dim)
        self.cata_embeds = nn.Embedding(cata_vocab_size, syn_dim)
        self.dropout_1 = nn.Dropout(ff_input_dropout)
        self.fc1 = nn.Linear(7 + 2 + 3*op_dim + 2*syn_dim, ff_hidden_dim, bias=True)
        self.dropout_2 = nn.Dropout(ff_hidden_dropout)
        self.relu = F.relu
        self.fc2 = nn.Linear(ff_hidden_dim, ff_output_dim, bias=True)

    def forward(self,
        d_onehot,
        fdec_ix,
        jdec_ix,
        ex_ix,
        op1_ix,
        op2_ix,
        catl_ix,
        cata_ix):

        ex_embed = self.ex_embeds(ex_ix)
        op1_embed = self.op1_embeds(op1_ix)
        op2_embed = self.op2_embeds(op2_ix)
        catl_embed = self.catl_embeds(catl_ix)
        cata_embed = self.cata_embeds(cata_ix)

        x = torch.cat((fdec_ix, jdec_ix, ex_embed, op1_embed, op2_embed, catl_embed, cata_embed, d_onehot), 1)
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

    depth, fdec, jdec, ex, op1, op2, catl, cata, catb, ex_to_ix, op1_to_ix, op2_to_ix, catl_to_ix, cata_to_ix, catb_to_ix = prepare_data()

    if config.getboolean("Dev"):
        d_depth, d_fdec, d_jdec, d_ex, d_op1, d_op2, d_catl, d_cata, d_catb = prepare_data_dev(config.get("DevFile"), ex_to_ix, op1_to_ix, op2_to_ix, catl_to_ix, cata_to_ix, catb_to_ix)

    model = BModel(device,
                   len(ex_to_ix),
                   len(op1_to_ix),
                   len(op2_to_ix),
                   config.getint("OpSize"),
                   len(catl_to_ix),
                   len(cata_to_ix),
                   config.getint("SynSize"),
                   config.getfloat("FFInputDropout"),
                   config.getint("HiddenSize"),
                   config.getfloat("FFHiddenDropout"),
                   len(catb_to_ix))

    eprint(str(model))
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    eprint("Topmost model has {} parameters".format(num_params))

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.getfloat("LearningRate"), weight_decay=config.getfloat("WeightDecay"))
    criterion = nn.NLLLoss()

    # training loop
    eprint("Start BModel training...")
    start_time = time.time()

    for epoch in range(config.getint("NEpochs")):
        vars = list(zip(depth, fdec, jdec, ex, op1, op2, catl, cata, catb))
        random.shuffle(vars)
        depth, fdec, jdec, ex, op1, op2, catl, cata, catb = zip(*vars)
        model.train()
        epoch_data_points = 0
        epoch_correct = 0
        epoch_loss = 0
        dev_data_points = 100
        dev_correct = 0
        dev_loss = 0

        for j in range(0, len(depth), config.getint("BatchSize")):
            b_depth = depth[j:j+config.getint("BatchSize")]
            b_fdec = fdec[j:j+config.getint("BatchSize")]
            b_jdec = jdec[j:j+config.getint("BatchSize")]
            b_ex = ex[j:j+config.getint("BatchSize")]
            b_op1 = op1[j:j+config.getint("BatchSize")]
            b_op2 = op2[j:j+config.getint("BatchSize")]
            b_catl = catl[j:j+config.getint("BatchSize")]
            b_cata = cata[j:j+config.getint("BatchSize")]
            b_catb = catb[j:j+config.getint("BatchSize")]
            depth_ix, fdec_ix, jdec_ix, ex_ix, op1_ix, op2_ix, catl_ix, cata_ix, catb_ix = \
            create_one_batch((b_depth, b_fdec, b_jdec, b_ex, b_op1, b_op2, b_catl, b_cata, b_catb),
                             ex_to_ix, op1_to_ix, op2_to_ix, catl_to_ix, cata_to_ix, catb_to_ix)

            # maybe move these to create_one_batch function
            depth_ix = depth_ix.to(device)
            fdec_ix = fdec_ix.to(device)
            jdec_ix = jdec_ix.to(device)
            ex_ix = ex_ix.to(device)
            op1_ix = op1_ix.to(device)
            op2_ix = op2_ix.to(device)
            catl_ix = catl_ix.to(device)
            cata_ix = cata_ix.to(device)
            catb_ix = catb_ix.to(device)

            optimizer.zero_grad()
            if config.getfloat("L2Reg") > 0:
                l2_loss = torch.cuda.FloatTensor([0]) if device == "cuda" else torch.FloatTensor([0])
                for param in model.parameters():
                    l2_loss += torch.mean(param.pow(2))
            else:
                l2_loss = 0

            output = model(depth_ix, fdec_ix, jdec_ix, ex_ix, op1_ix, op2_ix, catl_ix, cata_ix)
            _, pred_catb = torch.max(output.data, 1)

            epoch_data_points += len(pred_catb)
            batch_correct = (pred_catb == catb_ix).sum().item()
            epoch_correct += batch_correct
            nll_loss = criterion(output, catb_ix)
            loss = nll_loss + config.getfloat("L2Reg") * l2_loss
            epoch_loss += loss.item() * len(pred_catb)
            loss.backward()
            optimizer.step()

        if config.getboolean("Dev"):
            eprint("Entering Dev...")
            model.eval()
            dev_data_points = 0
            dev_correct = 0
            dev_loss = 0
            # TODO: check whether F/J lstm models were doing dev incorrectly
            for j in range(0, len(d_depth), config.getint("DevBatchSize")):
                b_depth = d_depth[j:j+config.getint("DevBatchSize")]
                b_fdec = d_fdec[j:j+config.getint("DevBatchSize")]
                b_jdec = d_jdec[j:j+config.getint("DevBatchSize")]
                b_ex = d_ex[j:j+config.getint("DevBatchSize")]
                b_op1 = d_op1[j:j+config.getint("DevBatchSize")]
                b_op2 = d_op2[j:j+config.getint("DevBatchSize")]
                b_catl = d_catl[j:j+config.getint("DevBatchSize")]
                b_cata = d_cata[j:j+config.getint("DevBatchSize")]
                b_catb = d_catb[j:j+config.getint("DevBatchSize")]

                depth_ix, fdec_ix, jdec_ix, ex_ix, op1_ix, op2_ix, catl_ix, cata_ix, catb_ix = create_one_batch(
                    (b_depth, b_fdec, b_jdec, b_ex, b_op1, b_op2, b_catl, b_cata, b_catb), ex_to_ix, op1_to_ix,
                    op2_to_ix, catl_to_ix, cata_to_ix, catb_to_ix)
                depth_ix = depth_ix.to(device)
                fdec_ix = fdec_ix.to(device)
                jdec_ix = jdec_ix.to(device)
                ex_ix = ex_ix.to(device)
                op1_ix = op1_ix.to(device)
                op2_ix = op2_ix.to(device)
                catl_ix = catl_ix.to(device)
                cata_ix = cata_ix.to(device)
                catb_ix = catb_ix.to(device)

                output = model(depth_ix, fdec_ix, jdec_ix, ex_ix, op1_ix, op2_ix, catl_ix, cata_ix)
                _, pred_catb = torch.max(output.data, 1)

                dev_data_points += len(pred_catb)
                batch_correct = (pred_catb == catb_ix).sum().item()
                dev_correct += batch_correct

                loss = criterion(output, catb_ix)
                dev_loss += loss.item() * len(pred_catb)

        eprint("Epoch {:04d} | AvgTrainLoss {:.4f} | TrainAcc {:.4f} | DevLoss {:.4f} | DevAcc {:.4f} | Time {:.4f}".format(
                epoch+1, epoch_loss/epoch_data_points, 100*(epoch_correct/epoch_data_points),
                dev_loss/dev_data_points, 100*(dev_correct/dev_data_points), time.time()-start_time))
        start_time = time.time()

    return model, ex_to_ix, op1_to_ix, op2_to_ix, catl_to_ix, cata_to_ix, catb_to_ix


def main(config):
    b_config = config["BModel"]
    model, ex_to_ix, op1_to_ix, op2_to_ix, catl_to_ix, cata_to_ix, catb_to_ix = train(b_config)
    # for param in model.state_dict():
    #     eprint(param)

    model.eval()
    if b_config.getboolean("GPU"):
        ex_embeds = model.state_dict()["ex_embeds.weight"].data.cpu().numpy()
        op1_embeds = model.state_dict()["op1_embeds.weight"].data.cpu().numpy()
        op2_embeds = model.state_dict()["op2_embeds.weight"].data.cpu().numpy()
        catl_embeds = model.state_dict()["catl_embeds.weight"].data.cpu().numpy()
        cata_embeds = model.state_dict()["cata_embeds.weight"].data.cpu().numpy()
        first_weights = model.state_dict()["fc1.weight"].data.cpu().numpy()
        first_biases = model.state_dict()["fc1.bias"].data.cpu().numpy()
        second_weights = model.state_dict()["fc2.weight"].data.cpu().numpy()
        second_biases = model.state_dict()["fc2.bias"].data.cpu().numpy()
    else:
        ex_embeds = model.state_dict()["ex_embeds.weight"].data.numpy()
        op1_embeds = model.state_dict()["op1_embeds.weight"].data.numpy()
        op2_embeds = model.state_dict()["op2_embeds.weight"].data.numpy()
        catl_embeds = model.state_dict()["catl_embeds.weight"].data.numpy()
        cata_embeds = model.state_dict()["cata_embeds.weight"].data.numpy()
        first_weights = model.state_dict()["fc1.weight"].data.numpy()
        first_biases = model.state_dict()["fc1.bias"].data.numpy()
        second_weights = model.state_dict()["fc2.weight"].data.numpy()
        second_biases = model.state_dict()["fc2.bias"].data.numpy()

    # Final classifier parameters
    print("B F " + ",".join(map(str, first_weights.flatten("F").tolist())))
    print("B f " + ",".join(map(str, first_biases.flatten("F").tolist())))
    print("B S " + ",".join(map(str, second_weights.flatten("F").tolist())))
    print("B s " + ",".join(map(str, second_biases.flatten("F").tolist())))

    # Embedding parameters (operators, syncat)
    for ex, ix in sorted(ex_to_ix.items()):
        print("E E " + str(ex) + " [" + ",".join(map(str, ex_embeds[ix])) + "]")

    for op, ix in sorted(op1_to_ix.items()):
        print("O 1 " + str(op) + " [" + ",".join(map(str, op1_embeds[ix])) + "]")

    for op, ix in sorted(op2_to_ix.items()):
        print("O 2 " + str(op) + " [" + ",".join(map(str, op2_embeds[ix])) + "]")

    for cat, ix in sorted(catl_to_ix.items()):
        print("C L " + str(cat) + " [" + ",".join(map(str, catl_embeds[ix])) + "]")

    for cat, ix in sorted(cata_to_ix.items()):
        print("C A " + str(cat) + " [" + ",".join(map(str, cata_embeds[ix])) + "]")

    # B decisions
    for cat, ix in sorted(catb_to_ix.items()):
        print("b " + str(ix) + " " + str(cat))


if __name__ == "__main__":
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(sys.argv[1])
    for section in config:
        eprint(section, dict(config[section]))
    main(config)
