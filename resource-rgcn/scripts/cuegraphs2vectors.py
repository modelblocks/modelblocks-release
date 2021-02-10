"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/MichSchli/RelationPrediction

Difference compared to MichSchli/RelationPrediction
* report raw metrics instead of filtered metrics
"""

import sys
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import knowledge_graph as knwlgrh

from layers import RGCNBlockLayer as RGCNLayer
from model import BaseRGCN

import utils
import os

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, h_dim)

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
        return RGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases, activation=act, self_loop=True, dropout=self.dropout)

class LinkPredict(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels, rel_type, num_bases=-1, num_hidden_layers=1, dropout=0, use_cuda=False, reg_param=0):
        super(LinkPredict, self).__init__()
        self.rgcn = RGCN(in_dim, h_dim, h_dim, num_rels, num_bases, num_hidden_layers, dropout, use_cuda)
        self.reg_param = reg_param
        self.rel_type = rel_type
        if rel_type == "vector":
            self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        elif rel_type == "matrix":
            self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim, h_dim))
        nn.init.xavier_uniform_(self.w_relation, gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        # DistMult
        s = embedding[triplets[:,0]]
        r = self.w_relation[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    # training a diagonal matrix for each relation
    def get_pred_vector(self, embedding, triplets):
        s = embedding[triplets[:,0]]
        r = self.w_relation[triplets[:,1]]
        o = embedding[triplets[:,2]]
        pred = s*r
        return pred, o

    # training a full matrix for each relation
    def get_pred_matrix(self, embedding, triplets):
        s = embedding[triplets[:,0]]
        s = torch.unsqueeze(s, 1)
        r = self.w_relation[triplets[:,1]]
        o = embedding[triplets[:,2]]
        pred = torch.matmul(s,r)
        pred = torch.squeeze(pred)
        return pred, o

    def forward(self, g):
        return self.rgcn.forward(g)

    def evaluate(self, g):
        # get embedding and relation weight without grad
        embedding = self.forward(g)
        return embedding, self.w_relation

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def bce_loss(self, g, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        embedding = self.forward(g)
        if self.rel_type == "matrix":
            raise TypeError("BCE loss requires vector relationships")
        score = self.calc_score(embedding, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embedding)
        return predict_loss + self.reg_param * reg_loss

    def mse_loss(self, g, triplets):
        embedding = self.forward(g)
        if self.rel_type == "vector":
            pred, gold = self.get_pred_vector(embedding, triplets)
        elif self.rel_type == "matrix":
            pred, gold = self.get_pred_matrix(embedding, triplets)
        else:
            raise NameError("learned relation not defined")
        criterion = nn.MSELoss(reduction="sum")
        predict_loss = criterion(pred, gold)
        reg_loss = self.regularization_loss(embedding)
        return predict_loss + self.reg_param * reg_loss

    def cos_loss(self, g, triplets, labels):
        embedding = self.forward(g)
        if self.rel_type == "vector":
            pred, gold = self.get_pred_vector(embedding, triplets)
        elif self.rel_type == "matrix":
            pred, gold = self.get_pred_matrix(embedding, triplets)
        else:
            raise NameError("learned relation not defined")
        criterion = nn.CosineEmbeddingLoss(margin=0.0, reduction="sum")
        predict_loss = criterion(pred, gold, labels)
        reg_loss = self.regularization_loss(embedding)
        return predict_loss + self.reg_param * reg_loss

def main(args):
    # load graph data
    if args.sampling == "neighborhood":
        data = knwlgrh.load_link(args.dataset)
        train_data = data.train
    elif args.sampling == "batch":
        data = knwlgrh.load_sent_link(args.dataset)
        train_data = data.sent_train
        edge_train_data = data.edge_train
    else:
        raise NameError("sampling method not defined")
    num_nodes = data.num_nodes
    # np.array (n, 3) for neighborhood, nested triplet list for batch

    num_rels = data.num_rels

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    # create model
    model = LinkPredict(num_nodes, args.n_hidden, num_rels, rel_type=args.rel_type, num_bases=args.n_bases,
        num_hidden_layers=args.n_layers, dropout=args.dropout, use_cuda=use_cuda, reg_param=args.regularization)

    for foo in model.parameters():
        print(foo.size())

    # build test graph
    if args.sampling == "batch":
        test_graph, test_rel, test_norm = utils.build_test_graph(num_nodes, num_rels, np.array(edge_train_data))

    elif args.sampling == "neighborhood":
        test_graph, test_rel, test_norm = utils.build_test_graph(num_nodes, num_rels, train_data)

    test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    test_rel = torch.from_numpy(test_rel)
    test_norm = torch.from_numpy(test_norm).view(-1, 1)
    test_graph.ndata.update({'id': test_node_id, 'norm': test_norm})
    # edges have unique IDs according to order of population
    # labels each edge with the relation type
    test_graph.edata['type'] = test_rel

    if use_cuda:
        model.cuda()

    # build adj list and calculate degrees for sampling
    if args.sampling == "neighborhood":
        adj_list, degrees = utils.get_adj_and_degrees(num_nodes, train_data)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # training loop
    print("start training...")
    epoch = 0
    while True:
        model.train()
        epoch += 1

        if args.sampling == "neighborhood":
            # perform edge neighborhood sampling to generate training graph and data
            # return g, uniq_v, rel, norm, samples, labels
            g, node_id, edge_type, node_norm, data, labels = utils.generate_sampled_graph_and_labels(
                train_data, args.graph_batch_size, args.graph_split_size,
                num_rels, adj_list, degrees, args.negative_sample, args.loss)
            # only half of the sampled data is used to create g
            # but "data" is all of the sampled data
            print("Done edge sampling")

            # set node/edge feature
            node_id = torch.from_numpy(node_id).view(-1, 1)
            edge_type = torch.from_numpy(edge_type)
            node_norm = torch.from_numpy(node_norm).view(-1, 1)
            data, labels = torch.from_numpy(data), torch.from_numpy(labels)
            deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)
            if use_cuda:
                node_id, deg = node_id.cuda(), deg.cuda()
                edge_type, node_norm = edge_type.cuda(), node_norm.cuda()
                data, labels = data.cuda(), labels.cuda()

            # assigns node IDs
            g.ndata.update({'id': node_id, 'norm': node_norm})
            # assigns node types
            g.edata['type'] = edge_type

            # perform propagation using half the sampled data only (g)
            # but evaluate the learned embedding using all of the sampled data (data)
            if args.loss == "bce":
                loss = model.bce_loss(g, data, labels)
            elif args.loss == "cos":
                loss = model.cos_loss(g, data, labels)
            else:
                raise NameError("loss function not specified")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
            optimizer.step()

            eprint("Epoch {:04d} | Loss {:.4f}".format(epoch, loss.item()))

            optimizer.zero_grad()

        elif args.sampling == "batch":
            # sample by sentence number and stack
            permutation = torch.randperm(len(train_data))
            for i in range(0, len(train_data), args.graph_batch_size):
                indices = permutation[i:i+args.graph_batch_size]
                # break
                # for index in indices:
                #     print(train_data[index])
                #     break
                sentences = [train_data[index] for index in indices]
                array = np.concatenate(sentences)
                # array = train_data[indices]
                g, node_id, edge_type, node_norm, data, labels = utils.generate_graph_and_labels(array, args.graph_split_size,
                num_rels, args.negative_sample, args.loss)
                # only half of the sampled data is used to create g
                # but "data" is all of the sampled data

                # set node/edge feature
                node_id = torch.from_numpy(node_id).view(-1, 1)
                edge_type = torch.from_numpy(edge_type)
                node_norm = torch.from_numpy(node_norm).view(-1, 1)
                data, labels = torch.from_numpy(data), torch.from_numpy(labels)
                deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)
                if use_cuda:
                    node_id, deg = node_id.cuda(), deg.cuda()
                    edge_type, node_norm = edge_type.cuda(), node_norm.cuda()
                    data, labels = data.cuda(), labels.cuda()

                # assigns node IDs
                g.ndata.update({'id': node_id, 'norm': node_norm})
                # assigns node types
                g.edata['type'] = edge_type

                # perform propagation using half the sampled data only (g)
                # but evaluate the learned embedding using all of the sampled data (data)
                if args.loss == "bce":
                    loss = model.bce_loss(g, data, labels)
                elif args.loss == "cos":
                    loss = model.cos_loss(g, data, labels)
                else:
                    raise NameError("loss function not specified")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm) # clip gradients
                optimizer.step()
                eprint("Epoch {:04d} | Batch {:04d}/{:04d} | Loss {:.4f}".format(epoch, i//args.graph_batch_size+1, len(train_data)//args.graph_batch_size+1, loss.item()))
                eprint("="*40)
                optimizer.zero_grad()

        else:
            raise NameError("sampling method not specified")

        if epoch % args.save_every == 0:
            model.eval()
            node_vec, rel_vec = model.evaluate(test_graph)
            torch.save(node_vec, args.dataset + "_" + args.rel_type + "rel_" + args.loss + "loss_" + str(args.n_hidden) + "dim_" + str(args.negative_sample) + "samples_" + str(epoch) + "epochs_nodes" + ".pt")
            torch.save(rel_vec, args.dataset + "_" + args.rel_type + "rel_" + args.loss + "loss_" + str(args.n_hidden) + "dim_" + str(args.negative_sample) + "samples_" + str(epoch) + "epochs_rels" + ".pt")

        if epoch == args.n_epochs:
            break

    print("Training finished")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    # default dropout rate = 0.2
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=20,
                        help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    # if using blocklayer, n-hidden needs to be divisible by n-bases
    parser.add_argument("--n-bases", type=int, default=1,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--n-epochs", type=int, default=1000,
                        help="number of minimum training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="dataset to use")
    parser.add_argument("-r", "--rel-type", type=str, required=True,
                        help="type of relation to learn, 'vector' (diagonal matrix) or 'matrix'")
    parser.add_argument("-l", "--loss", type=str, required=True,
                        help="loss function to use, 'bce' or 'cos'")
    parser.add_argument("-s", "--sampling", type=str, required=True,
                        help="sampling method, 'neighborhood' or 'batch'")
    parser.add_argument("--eval-batch-size", type=int, default=1000,
                        help="batch size when evaluating")
    parser.add_argument("--regularization", type=float, default=0.01,
                        help="regularization weight")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")
    parser.add_argument("--graph-batch-size", type=int, default=1024,
                        help="number of edges to sample in each iteration for neighborhood sampling, and number of sentences to sample for batch sampling")
    parser.add_argument("--graph-split-size", type=float, default=0.5,
                        help="portion of edges used as positive sample")
    parser.add_argument("--negative-sample", type=int, default=2,
                        help="number of negative samples per positive sample")
    parser.add_argument("--save-every", type=int, default=1000,
                        help="save node/relationship representations every n epochs")
    args = parser.parse_args()
    print(args)
    main(args)
