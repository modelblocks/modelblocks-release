"""
Utility functions for inducing predicate vectors and cued association transition model
Some code is adapted from the DGL implementation of RGCN link prediction:
https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""

import sys, os, re
import numpy as np
import torch
import torch.nn.functional as F
import dgl
import itertools


#######################################################################
#
# Utility function for preprocessing .discgraphs
#
#######################################################################


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def merge_inheritance(inh_chain):
    """
    Takes a nested list of "inheritance" nodes and returns a nested list
    of nodes that eventually inherit from the same node ("inheritance group")
    """
    inh_group = []

    while len(inh_chain) > 0:
        first, *rest = inh_chain
        first = set(first)
        lf = -1
        while len(first) > lf:
            lf = len(first)
            rest2 = []
            for r in rest:
                if len(first.intersection(set(r))) > 0:
                    first |= set(r)
                else:
                    rest2.append(r)
            rest = rest2
        inh_group.append(first)
        inh_chain = rest

    return inh_group


def discgraphs_to_cues(graph_file):
    """
    Main .discgraphs preprocessing code
    Returns nested list of edges grouped by sentence and entity, relationship, sentence dictionaries
    """
    labeled_cues = []
    discourse_num = 0
    sents = []
    entities = []
    num_preds = 0
    relations = []

    with open(graph_file, "r") as discgraphs:
        data = discgraphs.readlines()

    # first pass: store inheritance structure
    for line in data:
        discourse_num += 1
        sentence_cues = []
        inh_chains = []
        inh_dict = {}
        nodes_to_preds = {}

        cuechunks = line.split()
        for chunk in cuechunks:
            cues = chunk.split(",", 2)

            # remove 4+ relationships
            if cues[1] not in ["0", "1", "2", "3", "c", "e", "h", "r"]:
                continue

            if cues[1] in ["c", "e", "h", "r"]:
                inh_chains.append([cues[0], cues[2]])
                continue

            # append sentence ID
            sentence_num = ''.join(filter(lambda i: i.isdigit(), cues[0]))[:-2]
            cues.append(str(discourse_num).zfill(5)+str(sentence_num).zfill(3))

            # # comment in for sentence-specific non-predicate nodes
            # for i in [0, 2]:
            #     if re.match("[0-9]", cues[i]):
            #         cues[i] = str(discourse_num).zfill(5)+str(sentence_num).zfill(3) + cues[i]

            sentence_cues.append(cues)

        inh_groups = merge_inheritance(inh_chains)

        # create dictionary for substituting inherited nodes
        for group in inh_groups:
            sorted_group = sorted(list(group))
            for elem in sorted_group[1:]:
                inh_dict[elem] = sorted_group[0]

        # second pass: substitute inherited nodes, store predicate relationships
        for cues in sentence_cues:
            for i in [0, 2]:
                if cues[i] in inh_dict:
                    cues[i] = inh_dict[cues[i]]

            if cues[1] == "0":
                nodes_to_preds[cues[0]] = cues[2]

        # third pass: replace predicates and remove 0 edges
        final_sentence_cues = []
        for cues in sentence_cues:
            if cues[1] == "0":
                continue

            for i in [0, 2]:
                if cues[i] in nodes_to_preds:
                    cues[i] = nodes_to_preds[cues[i]]

            final_sentence_cues.append(cues)

        labeled_cues.extend(final_sentence_cues)

    # final pass: create dictionaries based on final cues
    for cue in labeled_cues:
        entities.extend([cue[0], cue[2]])
        relations.append(cue[1])
        sents.append(cue[3])
        relations.append("-" + str(cue[1]))

    entity_dict = {k: v for v, k in enumerate(sorted(set(entities), key=lambda e: (re.match("[0-9]", e) is not None, e)))}
    relation_dict = {k: v for v, k in enumerate(sorted(set(relations)))}
    sentence_dict = {k: v for v, k in enumerate(sorted(set(sents)))}

    # count number of predicates
    for k in entity_dict:
        if not re.match("[0-9]", k):
            num_preds += 1

    return labeled_cues, entity_dict, relation_dict, sentence_dict, num_preds


#######################################################################
#
# Utility function for building training and testing graphs
#
#######################################################################


def cues_to_edge_ids(labeled_cues, entity_dict, relation_dict, sentence_dict):
    """
    Returns two nested lists of edges and unique predicates (mapped to node/relation IDs) grouped by sentence
    """
    # returns [[subID, relID, objID], [subID, relID, objID], ...]
    sent_edges = [[] for _ in range(len(sentence_dict))]
    sent_all_preds = [[] for _ in range(len(sentence_dict))]
    sent_unique_preds = []

    for cues in labeled_cues:
        s = entity_dict[cues[0]]
        r = relation_dict[cues[1]]
        inv_r = relation_dict["-"+cues[1]]
        o = entity_dict[cues[2]]
        sent_edges[sentence_dict[cues[3]]].append([s, r, o])
        # append inverse relationship
        sent_edges[sentence_dict[cues[3]]].append([o, inv_r, s])
        if not re.match("[0-9]", cues[0]):
            sent_all_preds[sentence_dict[cues[3]]].append(s)

    for preds in sent_all_preds:
        sent_unique_preds.append(list(sorted(set(preds))))

    return sent_edges, sent_unique_preds


def generate_batched_graph(array_list, pred_list, use_cuda):
    """
    Takes two nested lists (sentence edges, sentence predicates) of size len(mini_batch) and
    returns a BatchedDGLGraph object and a list of predicate graph nodes
    """
    graphs = []

    for array, preds in zip(array_list, pred_list):
        src, rel, dst = array.transpose()
        uniq_v, edges = np.unique((src, dst), return_inverse=True)
        src, dst = np.reshape(edges, (2, -1))
        g, rel, norm = build_graph_from_triplets(len(uniq_v), (src, rel, dst))
        is_pred = torch.LongTensor([1 if node in preds else 0 for node in uniq_v]).view(-1, 1)
        node_id = torch.from_numpy(uniq_v).view(-1, 1)
        node_norm = torch.from_numpy(norm).view(-1, 1)
        edge_type = torch.from_numpy(rel)

        if use_cuda:
            node_id, edge_type, is_pred, node_norm = node_id.cuda(), edge_type.cuda(), is_pred.cuda(), node_norm.cuda()

        # assigns node IDs
        g.ndata.update({'id': node_id, 'is_pred': is_pred, 'norm': node_norm})
        # assigns node types
        g.edata['type'] = edge_type
        graphs.append(g)

    bg = dgl.batch(graphs)
    pred_graph_nodes = (bg.ndata['is_pred'] == 1.0).squeeze().nonzero().squeeze()
    eprint('Sampled {} nodes and {} edges from {} sentences'.format(bg.number_of_nodes(), bg.number_of_edges(), len(graphs)))

    return bg, pred_graph_nodes


def generate_full_graph(array_list, pred_list, relation_dict):
    """
    Takes a nested list and the relation dictionary and returns a BatchedDGLGraph object
    and relation-specific source and destination ID tensors
    """
    graphs = []
    src_ids = {}
    dst_ids = {}

    for array, preds in zip(array_list, pred_list):
        src, rel, dst = array.transpose()
        uniq_v, edges = np.unique((src, dst), return_inverse=True)
        src, dst = np.reshape(edges, (2, -1))
        g, rel, norm = build_graph_from_triplets(len(uniq_v), (src, rel, dst))
        is_pred = torch.FloatTensor([1 if node in preds else 0 for node in uniq_v]).view(-1, 1)
        node_id = torch.from_numpy(uniq_v).view(-1, 1)
        node_norm = torch.from_numpy(norm).view(-1, 1)
        edge_type = torch.from_numpy(rel)

        # assigns node IDs
        g.ndata.update({'id': node_id, 'is_pred': is_pred, 'norm': node_norm})
        # assigns node types
        g.edata['type'] = edge_type
        graphs.append(g)

    fg = dgl.batch(graphs)

    for k in relation_dict:
        src, dst = fg.find_edges((fg.edata['type'] == relation_dict[k]).squeeze().nonzero().squeeze())
        src_ids[k], dst_ids[k] = src, dst

    return fg, src_ids, dst_ids


def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    return norm


def build_graph_from_triplets(num_nodes, triplets):
    """
    Create a DGL graph with edge types and normalization factor
    """
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    src, rel, dst = triplets
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)

    return g, rel, norm


def group_similarity(model, entity_dict):
    """
    Takes RGCN model and node entity dictionary as input and calculates the between- and
    within-category cosine similarities of some N-aD predicates
    """
    with torch.no_grad():
        embedding = model.target_emb

    currency = ["N-aD:dollar", "N-aD:euro", "N-aD:pound", "N-aD:lira", "N-aD:franc", "N-aD:yen"]
    time = ["N-aD:day", "N-aD:month", "N-aD:year", "N-aD:january", "N-aD:july", "N-aD:december"]
    business = ["N-aD:co.", "N-aD:corp.", "N-aD:inc.", "N-aD:l.p.", "N-aD:ltd.", "N-aD:business"]
    jobs = ["N-aD:president", "N-aD:officer", "N-aD:ceo", "N-aD:director", "N-aD:minister", "N-aD:chairman"]
    countries = ["N-aD:u.s.", "N-aD:china", "N-aD:japan", "N-aD:germany", "N-aD:india", "N-aD:u.k."]
    categories = [currency, time, business, jobs, countries]

    bet_group_sim = []
    within_group_sim = []

    for i, j in itertools.combinations(categories, 2):
        for k, l in itertools.product(i, j):
            if k not in entity_dict or l not in entity_dict:
                continue
            bet_group_sim.append(F.cosine_similarity(embedding[entity_dict[k]], embedding[entity_dict[l]], 0).item())

    for i in categories:
        for j, k in itertools.combinations(i, 2):
            if j not in entity_dict or k not in entity_dict:
                continue
            within_group_sim.append(F.cosine_similarity(embedding[entity_dict[j]], embedding[entity_dict[k]], 0).item())

    bet_group_avg = round(sum(bet_group_sim)/len(bet_group_sim), 4)
    within_group_avg = round(sum(within_group_sim)/len(within_group_sim), 4)

    return bet_group_avg, within_group_avg
