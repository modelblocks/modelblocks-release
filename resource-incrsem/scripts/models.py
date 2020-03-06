"""
Models for inducing predicate vectors and cued association transition model
Some code is adapted from the DGL implementation of RGCN link prediction:
https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""

import torch, dgl
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


#######################################################################
#
# RGCN layer and model
#
#######################################################################


class BaseRGCN(nn.Module):

    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, use_cuda=False):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.self_loop = self_loop
        self.use_cuda = use_cuda

        # create RGCN layers
        self.build_model()

        # create initial features
        self.features = self.create_features()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    # initialize feature for each node
    def create_features(self):
        return None

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g):
        if self.features is not None:
            g.ndata['id'] = self.features
        for layer in self.layers:
            layer(g)
        return g.ndata.pop('h')


class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, activation=None, self_loop=False, dropout=0.0):
        super(RGCNLayer, self).__init__()
        self.activation = activation
        self.self_loop = self_loop
        self.bias_term = nn.Parameter(torch.Tensor(num_rels, out_feat))
        nn.init.xavier_uniform_(self.bias_term, gain=nn.init.calculate_gain('relu'))
        self.gate_weight = nn.Parameter(torch.Tensor(num_rels, out_feat, 1))
        nn.init.xavier_uniform_(self.gate_weight, gain=nn.init.calculate_gain('relu'))
        self.gate_bias = nn.Parameter(torch.Tensor(num_rels, 1))
        nn.init.xavier_uniform_(self.gate_bias, gain=nn.init.calculate_gain('relu'))

        # weight for self-loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

        # dropout on self-loop
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    # define how propagation is done in subclass
    def propagate(self, g):
        raise NotImplementedError

    def forward(self, g):
        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            if self.dropout is not None:
                # dropout randomly zeroes out elements at a given probability
                loop_message = self.dropout(loop_message)

        g.set_n_initializer(dgl.init.zero_initializer)
        self.propagate(g)

        # apply self-loop and activation
        node_repr = g.ndata['h']
        if self.self_loop:
            node_repr = node_repr + loop_message
        if self.activation:
            node_repr = self.activation(node_repr)

        g.ndata['h'] = node_repr


class RGCNBlockLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, activation=None, self_loop=False, dropout=0.0):
        super(RGCNBlockLayer, self).__init__(in_feat, out_feat, num_rels, activation, self_loop=self_loop, dropout=dropout)
        self.num_rels = num_rels
        self.num_bases = num_bases
        assert self.num_bases > 0
        self.out_feat = out_feat
        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases
        # assumes in_feat and out_feat are both divisible by num_bases
        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases * self.submat_in * self.submat_out))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    # linear transform and gating on source node representations
    def msg_func(self, edges):
        weight = self.weight.index_select(0, edges.data['type']).view(-1, self.submat_in, self.submat_out)
        node = edges.src['h'].view(-1, 1, self.submat_in)
        full_node = edges.src['h'].view(-1, 1, 20)
        bias = self.bias_term.index_select(0, edges.data['type'])
        gate_weight = self.gate_weight.index_select(0, edges.data['type'])
        gate_bias = self.gate_bias.index_select(0, edges.data['type'])
        gate_score = torch.sigmoid(torch.bmm(full_node, gate_weight).view(-1, 1) + gate_bias)
        msg = gate_score * (torch.bmm(node, weight).view(-1, self.out_feat) + bias)
        return {'msg': msg}

    # sum all incoming messages
    def propagate(self, g):
        g.update_all(self.msg_func, fn.sum(msg='msg', out='h'), self.apply_func)

    def apply_func(self, nodes):
        return {'h': nodes.data['h']}


#######################################################################
#
# OModel model
#
#######################################################################


class OModel(nn.Module):

    def __init__(self, embedding_dim):
        super(OModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.fc1 = nn.Linear(self.embedding_dim, 2*self.embedding_dim, bias=True)
        self.relu = F.relu
        self.fc2 = nn.Linear(2*self.embedding_dim, self.embedding_dim, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
