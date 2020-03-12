"""
Trainer for inducing predicate vectors and cued association transition model
Some code is adapted from the DGL implementation of RGCN link prediction:
https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""

import sys, configparser, torch, utils
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models import BaseRGCN, RGCNBlockLayer, OModel
torch.set_printoptions(profile="full")


class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(num_nodes, h_dim)

    def forward(self, g):
        node_id = g.ndata['id'].squeeze()
        # indexes embeddings by node_id
        g.ndata['h'] = self.embedding(node_id)


class RGCN(BaseRGCN):
    # overriding None with EmbeddingLayer
    def build_input_layer(self):
        return EmbeddingLayer(self.num_nodes, self.h_dim)

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None
        return RGCNBlockLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases, activation=act, self_loop=self.self_loop, dropout=self.dropout)


class WordPredict(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels, num_preds, num_bases=-1, num_hidden_layers=1, dropout=0, self_loop=False, use_cuda=False, reg_param=0):
        super(WordPredict, self).__init__()
        # declare vt vectors for all nodes or just predicates
        self.target_emb = nn.Parameter(torch.Tensor(num_preds, h_dim))
        # self.target_emb = nn.Parameter(torch.Tensor(in_dim, h_dim))
        nn.init.xavier_uniform_(self.target_emb, gain=nn.init.calculate_gain('relu'))
        self.rgcn = RGCN(in_dim, h_dim, h_dim, num_rels, num_bases, num_hidden_layers, dropout, self_loop, use_cuda)
        self.reg_param = reg_param

    def get_norm_score(self, embedding, pred_graph_nodes):
        rgcn_emb = embedding[pred_graph_nodes]
        score = torch.matmul(rgcn_emb, torch.t(self.target_emb))
        norm_score = F.log_softmax(score, dim=1)
        return norm_score

    def forward(self, g):
        return self.rgcn.forward(g)

    def my_nll_loss(self, g, pred_graph_nodes, gold_pred_index):
        embedding = self.forward(g)
        score = self.get_norm_score(embedding, pred_graph_nodes)
        _, prediction = torch.max(score, 1)
        predict_loss = F.nll_loss(score, gold_pred_index)
        target_reg_loss = torch.mean(self.target_emb.pow(2))
        num_pred_nodes = len(pred_graph_nodes)
        correct = (prediction == gold_pred_index).sum().item()
        return predict_loss + self.reg_param * target_reg_loss, correct, num_pred_nodes


def main(graph_file, config):
    rgcn_config = config["RGCN"]

    utils.eprint("Preprocessing {}".format(graph_file))
    labeled_cues, entity_dict, relation_dict, sentence_dict, num_preds = utils.discgraphs_to_cues(graph_file)
    utils.eprint("Preprocessing finished")

    sent_edges, sent_unique_preds = utils.cues_to_edge_ids(labeled_cues, entity_dict, relation_dict, sentence_dict)
    num_nodes, num_rels, num_sents = len(entity_dict), len(relation_dict), len(sent_edges)
    num_edges = sum([len(sentence) for sentence in sent_edges])

    utils.eprint("# entities: {}".format(num_nodes))
    utils.eprint("# predicates: {}".format(num_preds))
    utils.eprint("# relations: {}".format(num_rels))
    utils.eprint("# sentences: {}".format(num_sents))
    utils.eprint("# edges: {}".format(num_edges))

    # check cuda
    use_cuda = rgcn_config.getint("GPU") >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(rgcn_config.getint("GPU"))

    # create model
    model = WordPredict(num_nodes, rgcn_config.getint("VecSize"), num_rels, num_preds, num_bases=rgcn_config.getint("NBases"), num_hidden_layers=rgcn_config.getint("NLayers"), dropout=rgcn_config.getfloat("Dropout"), self_loop=rgcn_config.getboolean("SelfLoop"), use_cuda=use_cuda, reg_param=rgcn_config.getfloat("Regularization"))

    for parameter in model.parameters():
        utils.eprint(parameter.size())

    if use_cuda:
        model.cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=rgcn_config.getfloat("LearningRate"))

    # training loop
    utils.eprint("Start RGCN training loop...")
    epoch = 0

    # model loading
    if rgcn_config.get("ModelPath") != "":
        checkpoint = torch.load(rgcn_config.get("ModelPath"))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        utils.eprint("Loading model {} from Epoch {}".format(rgcn_config.get("ModelPath"), epoch))

    while True:
        model.train()
        epoch += 1
        total_train_nodes = 0
        total_train_correct = 0
        total_train_loss = 0

        # sample by sentence number
        permutation = torch.randperm(num_sents)
        for i in range(0, num_sents, rgcn_config.getint("BatchSize")):
            indices = permutation[i: i+rgcn_config.getint("BatchSize")]
            sampled_edges = [np.array(sent_edges[index]) for index in indices]
            sampled_preds = [sent_unique_preds[index] for index in indices]
            flat_sampled_preds = [pred for index in indices for pred in sent_unique_preds[index]]
            bg, pred_graph_nodes = utils.generate_batched_graph(sampled_edges, sampled_preds, use_cuda)
            flat_sampled_preds = torch.LongTensor(flat_sampled_preds)

            if use_cuda:
                flat_sampled_preds = flat_sampled_preds.cuda()

            loss, correct, num_pred_nodes = model.my_nll_loss(bg, pred_graph_nodes, flat_sampled_preds)
            total_train_correct += correct
            total_train_nodes += num_pred_nodes
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), rgcn_config.getfloat("GradNorm")) # clip gradients
            optimizer.step()

            utils.eprint("Epoch {:04d} | Batch {:04d}/{:04d} | Loss {:.4f} | TrainAcc {:.4f}".format(epoch, i//rgcn_config.getint("BatchSize")+1, num_sents//rgcn_config.getint("BatchSize")+1, loss.item(), total_train_correct/total_train_nodes))
            optimizer.zero_grad()

        if epoch % rgcn_config.getint("EvaluateEvery") == 0:
        # within-category and between-category similarity
            if use_cuda:
                model.cpu()  # test on CPU

            model.eval()
            between, within = utils.group_similarity(model, entity_dict)
            utils.eprint("Epoch {:04d} | Between {:.4f} | Within {:.4f} | W-B {:.4f} | TrainAcc {:.4f} | AvgTrainLoss {:.4f}".format(epoch, between, within, within-between, total_train_correct/total_train_nodes, total_train_loss/(num_sents//rgcn_config.getint("BatchSize"))))
            utils.eprint("=" * 60)

            if use_cuda:
                model.cuda()

        if epoch % rgcn_config.getint("SaveModelEvery") == 0:
            if use_cuda:
                model.cpu()
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "loss": loss}, "{}_{}dim_{}epochs_{}lr_{}sentbatches_rgcnmodel.pt".format(graph_file, rgcn_config.get("VecSize"), epoch, rgcn_config.get("LearningRate"), rgcn_config.get("BatchSize")))
            if use_cuda:
                model.cuda()

        if epoch == rgcn_config.getint("NEpochs"):
            break

    utils.eprint("RGCN training finished")
    utils.eprint("Preprocessing for OModel trainer")
    all_edges = [np.array(sent) for sent in sent_edges]
    fg, src_ids, dst_ids = utils.generate_full_graph(all_edges, relation_dict)
    utils.eprint("Preprocessing finished")

    # calculate h_vectors on CPU due to memory concerns
    if use_cuda:
        model.cpu()

    with torch.no_grad():
        fg_embedding = model(fg)

    # training loop
    omodel_config = config["OModel"]
    utils.eprint("Start OModel training loop...")
    first_weights = {}
    first_biases = {}
    second_weights = {}
    second_biases = {}

    for rel in sorted(relation_dict):
        utils.eprint("Training model {}".format(rel))
        src, dst = fg_embedding[src_ids[rel]], fg_embedding[dst_ids[rel]]
        omodel = OModel(rgcn_config.getint("VecSize"))
        optimizer = torch.optim.Adam(omodel.parameters(), omodel_config.getfloat("LearningRate"))
        criterion = nn.MSELoss(reduction="mean")
        current_loss = 1000
        epoch = 1

        while True:
            prediction = omodel(src)
            loss = criterion(prediction, dst)
            loss.backward()
            optimizer.step()
            utils.eprint('Relation {} | Epoch {:04d} | Loss {:.4f}'.format(rel, epoch, loss.item()))
            optimizer.zero_grad()
            epoch += 1
            if loss.item() < current_loss:
                if current_loss - loss.item() < omodel_config.getfloat("StopCriterion"):
                    utils.eprint("Stopping criterion met")
                    break
                else:
                    current_loss = loss.item()

        first_weights[rel] = list(omodel.parameters())[0].data.numpy()
        first_biases[rel] = list(omodel.parameters())[1].data.numpy()
        second_weights[rel] = list(omodel.parameters())[2].data.numpy()
        second_biases[rel] = list(omodel.parameters())[3].data.numpy()

    id_to_entity = {v: k for k, v in entity_dict.items()}

    for k in range(num_preds):
        print("E " + id_to_entity[k] + " [" + ",".join(map(str, model.target_emb[k].data.numpy())) + "]")

    for rel in relation_dict:
        print("O " + str(rel) + " F " + ",".join(map(str, first_weights[rel].flatten().tolist())))
        print("O " + str(rel) + " f " + ",".join(map(str, first_biases[rel].flatten().tolist())))
        print("O " + str(rel) + " S " + ",".join(map(str, second_weights[rel].flatten().tolist())))
        print("O " + str(rel) + " s " + ",".join(map(str, second_biases[rel].flatten().tolist())))

    # temporary hack
    for rel in [4, 5, 6, 7, 8]:
        print("O " + str(rel) + " F " + ",".join(map(str, first_weights["3"].flatten().tolist())))
        print("O " + str(rel) + " f " + ",".join(map(str, first_biases["3"].flatten().tolist())))
        print("O " + str(rel) + " S " + ",".join(map(str, second_weights["3"].flatten().tolist())))
        print("O " + str(rel) + " s " + ",".join(map(str, second_biases["3"].flatten().tolist())))

    for rel in [-4, -5, -6, -7, -8]:
        print("O " + str(rel) + " F " + ",".join(map(str, first_weights["-3"].flatten().tolist())))
        print("O " + str(rel) + " f " + ",".join(map(str, first_biases["-3"].flatten().tolist())))
        print("O " + str(rel) + " S " + ",".join(map(str, second_weights["-3"].flatten().tolist())))
        print("O " + str(rel) + " s " + ",".join(map(str, second_biases["-3"].flatten().tolist())))


if __name__ == '__main__':
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(sys.argv[2])
    for section in config:
        utils.eprint(section, dict(config[section]))
    main(sys.argv[1], config)
