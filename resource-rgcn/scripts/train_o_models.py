import os, sys, re, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

def prepare_data(path, vectors, rel):
    entity_path = os.path.join(path, 'entities.dict')
    relation_path = os.path.join(path, 'relations.dict')
    train_path = os.path.join(path, 'train.txt')
    entity_dict = read_dictionary(entity_path)
    relation_dict = read_dictionary(relation_path)
    train_data = np.array(read_triplets_as_list(train_path, entity_dict, relation_dict))
    rel_nodes = train_data[train_data[:, 1] == relation_dict[rel]]
    rel_subjects = vectors[rel_nodes[:, 0]]
    rel_objects = vectors[rel_nodes[:, 2]]
    return rel_nodes, rel_subjects, rel_objects

# train models
class OModel(nn.Module):

    def __init__(self, embedding_dim):
        super(OModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.fc1 = nn.Linear(self.embedding_dim, 2*self.embedding_dim, bias=False)
        self.relu = F.relu
        self.fc2 = nn.Linear(2*self.embedding_dim, self.embedding_dim, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train(path, vectors, rel):
    _, subjects, objects = prepare_data(path, vectors, rel)
    print("Training model", rel)
    model = OModel(subjects.size()[1])
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss(reduction="mean")

    # for epoch in range(epochs):
    current_loss = 100
    epoch = 1
    while True:
        trng_pred = model(subjects)
        loss = criterion(trng_pred, objects)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print('Relation %s, Epoch %d, Loss: %.4f' % (rel, epoch, loss.item()))
        epoch += 1
        if loss.item() < current_loss:
            if current_loss-loss.item() < 0.0001:
                print("Stopping criterion met")
                break
            else:
                current_loss = loss.item()

    print("Training model", "-"+rel)
    neg_model = OModel(objects.size()[1])
    neg_optimizer = optim.Adam(neg_model.parameters(), lr=0.01)
    current_loss = 100
    epoch = 1
    while True:
        trng_pred = neg_model(objects)
        neg_loss = criterion(trng_pred, subjects)
        neg_loss.backward()
        neg_optimizer.step()
        neg_optimizer.zero_grad()
        print('Relation %s, Epoch %d, Loss: %.4f' % ("-"+rel, epoch, neg_loss.item()))
        epoch += 1
        if neg_loss.item() < current_loss:
            if current_loss - neg_loss.item() < 0.0001:
                print("Stopping criterion met")
                break
            else:
                current_loss = neg_loss.item()

    return model, neg_model, subjects, objects

def main(path, vectorfile):
    vectors = torch.load(vectorfile)
    vectors.requires_grad = False
    entity_path = os.path.join(path, 'entities.dict')
    relation_path = os.path.join(path, 'relations.dict')
    # train_path = os.path.join(path, 'train.txt')
    entity_dict = read_dictionary(entity_path)
    relation_dict = read_dictionary(relation_path)
    # train_data = np.array(read_triplets_as_list(train_path, entity_dict, relation_dict))
    # ix_to_entity = {v: k for k, v in entity_dict.items()}
    # print("Begin training O model -0")
    # neg_zero_model, _, _ = train(path, vectors, "-0")
    terminals = []
    # neg_zero_applied = neg_zero_model(vectors).detach().numpy()

    for i in entity_dict:
        if not re.match("[0-9]", i):
            terminals.append(i)

    # for i in range(train_data.shape[0]):
    #     if train_data[i, 1]==relation_dict["-0"]:
    #         terminals.append(ix_to_entity[train_data[i, 0]])
    # terminals = set(terminals)

    first_weights = {}
    second_weights = {}
    for rel in relation_dict:
        # if rel == "-0" or rel == "0":
        #     continue
        # print("Begin training O model", rel)
        model, neg_model, _, _ = train(path, vectors, rel)
        first_weights[rel] = list(model.parameters())[0].data.numpy()
        second_weights[rel] = list(model.parameters())[1].data.numpy()
        first_weights["-"+rel] = list(neg_model.parameters())[0].data.numpy()
        second_weights["-"+rel] = list(neg_model.parameters())[1].data.numpy()

    with open(vectorfile[:-3] + "_emat_omodel.txt", "w") as f:
        for terminal in terminals:
            # f.write("E "+str(terminal)+" ["+",".join(map(str, neg_zero_applied[entity_dict[terminal]]))+"]\n")
            f.write("E " + str(terminal) + " [" + ",".join(map(str, vectors[entity_dict[terminal]].data.numpy())) + "]\n")
        for rel in first_weights:
            f.write("O "+str(rel)+" F "+",".join(map(str, first_weights[rel].flatten().tolist()))+"\n")
            f.write("O "+str(rel)+" S "+",".join(map(str, second_weights[rel].flatten().tolist()))+"\n")
        # temporary hack
        for rel in [4, 5, 6, 7, 8]:
            f.write("O "+str(rel)+" F "+",".join(map(str, first_weights["3"].flatten().tolist()))+"\n")
            f.write("O "+str(rel)+" S "+",".join(map(str, second_weights["3"].flatten().tolist()))+"\n")
        for rel in [-4, -5, -6, -7, -8]:
            f.write("O "+str(rel)+" F "+",".join(map(str, first_weights["-3"].flatten().tolist()))+"\n")
            f.write("O "+str(rel)+" S "+",".join(map(str, second_weights["-3"].flatten().tolist()))+"\n")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
