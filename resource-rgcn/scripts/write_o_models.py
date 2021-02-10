import os, sys, torch
import numpy as np

def read_dictionary(filename):
    d = {}
    with open(filename, 'r+') as f:
        for line in f:
            line = line.strip().split('\t')
            d[line[1]] = int(line[0])
    return d

def read_triplets(filename):
    with open(filename, 'r+') as f:
        for line in f:
            processed_line = line.strip().split('\t')
            yield processed_line

def read_triplets_as_list(filename, entity_dict, relation_dict):
    l = []
    for triplet in read_triplets(filename):
        s = entity_dict[triplet[0]]
        r = relation_dict[triplet[1]]
        o = entity_dict[triplet[2]]
        l.append([s, r, o])
    return l

def main(path, nodefile, relfile, rel_type):
    nodes = torch.load(nodefile)
    relations = torch.load(relfile)
    entity_path = os.path.join(path, 'entities.dict')
    relation_path = os.path.join(path, 'relations.dict')
    train_path = os.path.join(path, 'train.txt')
    entity_dict = read_dictionary(entity_path)
    relation_dict = read_dictionary(relation_path)
    train_data = np.array(read_triplets_as_list(train_path, entity_dict, relation_dict))
    ix_to_entity = {v: k for k, v in entity_dict.items()}
    terminals = []

    if rel_type == "vector":
        neg_zero_applied = nodes * relations[relation_dict["-0"]]
    elif rel_type == "matrix":
        nodes = torch.unsqueeze(nodes, 1)
        neg_zero_applied = torch.matmul(nodes, relations[relation_dict["-0"]])
        neg_zero_applied = torch.squeeze(neg_zero_applied)
        neg_zero_applied = neg_zero_applied.data.numpy()
    else:
        raise NameError("Relation type not specified")

    relations = relations.data.numpy()

    for i in range(train_data.shape[0]):
        if train_data[i, 1]==relation_dict["-0"]:
            terminals.append(ix_to_entity[train_data[i, 0]])
    terminals = set(terminals)

    with open(nodefile[:-3] + "_emat_omodel.txt", "w") as f:
        for terminal in terminals:
            f.write("E "+str(terminal)+" ["+",".join(map(str, neg_zero_applied[entity_dict[terminal]]))+"]\n")
        for rel in relation_dict:
            if rel == "0" or rel == "-0":
                continue
            if rel_type == "vector":
                f.write("O " + str(rel) + " " + ",".join(map(str, relations[relation_dict[rel]])) + "\n")
            elif rel_type == "matrix":
                rel_mat = relations[relation_dict[rel]]
                rel_mat = np.squeeze(rel_mat)
                # for ensuring python == c++
                # print(rel, np.matmul(np.ones((1,40)), rel_mat))
                # transposed for easier multiplication in dense model code
                rel_mat = np.transpose(rel_mat)
                rel_mat = rel_mat.flatten().tolist()
                f.write("O " + str(rel) + " " + ",".join(map(str, rel_mat)) + "\n")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])