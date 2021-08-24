import sys, configparser, torch, re, os, time, random, pickle
from collections import Counter
import torch.nn as nn
import torch.optim as optim

from transformerfmodel import eprint
from train_fmodel import extract_first_kvec, hvecIxReplace
from transformerjmodel import TransformerJModel

# The C++ code expects a certain format for each jdec, so this is
# a fake jdec used for padding
PAD = '0&J&J&J'


# Dummy class for recording information about each J decision
class JInfo:
    pass

def _initialize_jinfo_list(infile):
    # list of lists; each outer list is a sentence
    per_sentence_jinfo = list()
    
    # list of f decisions for the current sentence
    curr_jinfo = list()
    
    is_first_line = True
    for line in infile:
        is_new_sentence, depth, cat_anc, hv_anc, hv_filler, cat_lc, hv_lc, \
            jdec = line.split()
        is_new_sentence = bool(int(is_new_sentence))
        if is_new_sentence:
            if is_first_line:
                is_first_line = False
            else:
                per_sentence_jinfo.append(curr_jinfo)
                curr_jinfo = list()

        jinfo = JInfo()
        jinfo.depth = int(depth)
        # "raw" members are strings/lists of strings that haven't been
        # mapped to IDs yet
        jinfo.raw_cat_anc = cat_anc
        jinfo.raw_hv_anc = extract_first_kvec(hv_anc)
        jinfo.raw_hv_filler = extract_first_kvec(hv_filler)
        jinfo.raw_cat_lc = cat_lc
        jinfo.raw_hv_lc = extract_first_kvec(hv_lc)
        jinfo.raw_jdec = jdec
        curr_jinfo.append(jinfo)

    per_sentence_jinfo.append(curr_jinfo)

    return per_sentence_jinfo


def _map_jinfo_list_to_ix(per_sentence_jinfo, cat_anc_to_ix, hv_anc_to_ix,
    hv_filler_to_ix, cat_lc_to_ix, hv_lc_to_ix, jdecs_to_ix, dev=False):
    '''
    Replaces strings inside JInfo objects with their corresponding indices.
    These modifications are done in place rather than creating new
    Jinfo objects.
    '''

    new_per_sentence_jinfo = list()
    for sentence_jinfo in per_sentence_jinfo:
        new_sentence_jinfo = list()
        for jinfo in sentence_jinfo:
            try:
                jinfo.cat_anc = cat_anc_to_ix[jinfo.raw_cat_anc]
                jinfo.hv_anc = hvecIxReplace(jinfo.raw_hv_anc, hv_anc_to_ix)
                jinfo.hv_filler = hvecIxReplace(jinfo.raw_hv_filler, hv_filler_to_ix)
                jinfo.cat_lc = cat_lc_to_ix[jinfo.raw_cat_lc]
                jinfo.hv_lc = hvecIxReplace(jinfo.raw_hv_lc, hv_lc_to_ix)
                jinfo.jdec = jdecs_to_ix[jinfo.raw_jdec]
            except KeyError:
                # dev may contain jdecs, cats, etc that haven't appeared in
                # training data. Throw out any such data
                if dev: continue
                else: raise
            new_sentence_jinfo.append(jinfo)
        new_per_sentence_jinfo.append(new_sentence_jinfo)
    return new_per_sentence_jinfo


def prepare_data():
    per_sentence_jinfo = _initialize_jinfo_list(sys.stdin)
           
    all_cat_anc = set()
    all_hv_anc = set()
    all_hv_filler = set()
    all_cat_lc = set()
    all_hv_lc = set()
    all_jdecs = set()
    for sentence in per_sentence_jinfo:
        for jinfo in sentence:
            all_cat_anc.add(jinfo.raw_cat_anc)
            all_hv_anc.update(set(jinfo.raw_hv_anc))
            all_hv_filler.update(set(jinfo.raw_hv_filler))
            all_cat_lc.add(jinfo.raw_cat_lc)
            all_hv_lc.update(jinfo.raw_hv_lc)
            all_jdecs.add(jinfo.raw_jdec)

    cat_anc_to_ix = {cat: i for i, cat in enumerate(sorted(all_cat_anc))}
    hv_anc_to_ix = {hvec: i for i, hvec in enumerate(sorted(all_hv_anc))}
    hv_filler_to_ix = {hvec: i for i, hvec in enumerate(sorted(all_hv_filler))}
    cat_lc_to_ix = {cat: i for i, cat in enumerate(sorted(all_cat_lc))}
    hv_lc_to_ix = {hvec: i for i, hvec in enumerate(sorted(all_hv_lc))}
    jdecs_to_ix = {jdec: i for i, jdec in enumerate(sorted(all_jdecs))}
    # ID for padding symbol added at end of sequence
    jdecs_to_ix[PAD] = len(jdecs_to_ix)

    per_sentence_jinfo = _map_jinfo_list_to_ix(
        per_sentence_jinfo, cat_anc_to_ix, hv_anc_to_ix, hv_filler_to_ix,
        cat_lc_to_ix, hv_lc_to_ix, jdecs_to_ix
    )

    return per_sentence_jinfo, cat_anc_to_ix, hv_anc_to_ix, hv_filler_to_ix, \
        cat_lc_to_ix, hv_lc_to_ix, jdecs_to_ix


def prepare_dev_data(dev_decpars, cat_anc_to_ix, hv_anc_to_ix, hv_filler_to_ix,
    cat_lc_to_ix, hv_lc_to_ix, jdecs_to_ix):
    per_sentence_jinfo = _initialize_jinfo_list(dev_decpars)
    per_sentence_jinfo = _map_jinfo_list_to_ix(
        per_sentence_jinfo, cat_anc_to_ix, hv_anc_to_ix, hv_filler_to_ix,
        cat_lc_to_ix, hv_lc_to_ix, jdecs_to_ix, dev=True
    )
    return per_sentence_jinfo


def get_jinfo_seqs(per_sentence_jinfo, window_size):
    jinfo_seqs = list()
    for sentence in per_sentence_jinfo:
        for i in range(0, len(sentence), window_size):
            jinfo_seqs.append(sentence[i:i+window_size])
    return jinfo_seqs


def pad_target_matrix(targets, symbol):
    '''
    Given a list of per-sequence JDec targets and a padding symbol,
    returns an SxN padded matrix of targets.
    S is max sequence length, and N in number of sequences
    '''
    max_seq_length = max(len(t) for t in targets)
    padded_targets = list()
    for t in targets:
        padded_targets.append(t + [symbol]*(max_seq_length - len(t)))
    return torch.transpose(torch.LongTensor(padded_targets), 0, 1)


def train(j_config):
    use_dev = j_config.getboolean('UseDev')
    dev_decpars = j_config.get('DevFile')
    use_gpu = j_config.getboolean('UseGPU')
    window_size = j_config.getint('AttnWindowSize')
    batch_size = j_config.getint('BatchSize')
    epochs = j_config.getint('NEpochs')
    l2_reg = j_config.getfloat('L2Reg')
    learning_rate = j_config.getfloat('LearningRate')
    weight_decay = j_config.getfloat('WeightDecay')

    per_sentence_jinfo, cat_anc_to_ix, hv_anc_to_ix, hv_filler_to_ix, \
        cat_lc_to_ix, hv_lc_to_ix, jdecs_to_ix = prepare_data()

    model = TransformerJModel(
                j_config=j_config,
                cat_anc_vocab_size=len(cat_anc_to_ix),
                hv_anc_vocab_size=len(hv_anc_to_ix),
                hv_filler_vocab_size=len(hv_filler_to_ix),
                cat_lc_vocab_size=len(cat_lc_to_ix),
                hv_lc_vocab_size=len(hv_lc_to_ix),
                output_dim=len(jdecs_to_ix)
    )

    if use_gpu:
        model = model.cuda()

    #criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_seqs = get_jinfo_seqs(per_sentence_jinfo, window_size)

    if use_dev:
        dev_per_sentence_jinfo = prepare_dev_data(
            open(dev_decpars), cat_anc_to_ix, hv_anc_to_ix, hv_filler_to_ix,
            cat_lc_to_ix, hv_lc_to_ix, jdecs_to_ix
        )
        dev_seqs = get_jinfo_seqs(dev_per_sentence_jinfo, window_size)

    # TODO add validation on dev set
    for epoch in range(epochs):
        model.train()
        c0 = time.time()
        random.shuffle(train_seqs)
        total_train_correct = 0
        total_train_loss = 0
        total_train_items = 0

        for j in range(0, len(train_seqs), batch_size):
            if use_gpu:
                l2_loss = torch.cuda.FloatTensor([0])
            else:
                l2_loss = torch.FloatTensor([0])
            for param in model.parameters():
                if torch.numel(param) == 0:
                    continue
                l2_loss += torch.mean(param.pow(2))

            batch = train_seqs[j:j+batch_size]
            target = [[ji.jdec for ji in seq] for seq in batch]
            # L x N
            # Note: padding of the input sequence happens within the
            # forward method of the J model
            target = pad_target_matrix(target, jdecs_to_ix[PAD])
            if use_gpu:
                target = target.to('cuda')

            # output dimension is L x N x E
            # L: window length
            # N: batch size
            # E: output dimensionality (number of classes)
            output = model(batch)

            # jdec dimension is L x N
            _, jdec = torch.max(output.data, 2)

            # count all items where jdec matches target and target isn't padding
            train_correct = ((jdec == target) * (target != jdecs_to_ix[PAD])).sum().item()
            total_train_correct += train_correct
            
            # again, ignore padding
            total_train_items += (target != jdecs_to_ix[PAD]).sum().item()
            #total_train_items += target.numel()

            
            # NLLLoss requires target dimension to be N x L
            # https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
            target = torch.transpose(target, 0, 1) # LxN --> NxL
                
            # NLLLoss requires output dimension to be N x E x L
            output = torch.transpose(output, 0, 2) # LxNxE --> ExNxL
            output = torch.transpose(output, 0, 1) # ExNxL --> NxExL


            # assign PAD class 0 weight so that it doesn't influence loss.
            # equally weight all other classes
            num_classes = output.shape[1]
            # last class is PAD
            assert jdecs_to_ix[PAD] == num_classes - 1
            per_class_weight = 1/(num_classes-1)
            class_weights = torch.FloatTensor([per_class_weight]*(num_classes-1) + [0])

            if use_gpu:
                class_weights = class_weights.to('cuda')

            criterion = nn.NLLLoss(weight=class_weights)
            nll_loss = criterion(output, target)
            loss = nll_loss + l2_reg * l2_loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if use_dev:
            with torch.no_grad():
                model.eval()

                # L x N
                dev_target = [[ji.jdec for ji in seq] for seq in dev_seqs]
                dev_target = pad_target_matrix(dev_target, jdecs_to_ix[PAD])
                if use_gpu:
                    dev_target = dev_target.to('cuda')

                # L x N x E
                dev_output = model(dev_seqs)

                _, dev_jdec = torch.max(dev_output.data, 2)

                dev_correct = ((dev_jdec == dev_target) * (dev_target != jdecs_to_ix[PAD])).sum().item()
                total_dev_items = (dev_target != jdecs_to_ix[PAD]).sum().item()

                # NLLLoss requires target dimension to be N x L
                # https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
                dev_target = torch.transpose(dev_target, 0, 1) # LxN --> NxL
                    
                # NLLLoss requires output dimension to be N x E x L
                dev_output = torch.transpose(dev_output, 0, 2) # LxNxE --> ExNxL
                dev_output = torch.transpose(dev_output, 0, 1) # ExNxL --> NxExL

                # assign PAD class 0 weight so that it doesn't influence loss.
                # equally weight all other classes
                num_classes = dev_output.shape[1]
                # last class is PAD
                assert jdecs_to_ix[PAD] == num_classes - 1
                per_class_weight = 1/(num_classes-1)
                class_weights = torch.FloatTensor([per_class_weight]*(num_classes-1) + [0])
                if use_gpu:
                    class_weights = class_weights.to('cuda')

                criterion = nn.NLLLoss(weight=class_weights)
                dev_loss = criterion(dev_output, dev_target).item()
                dev_acc = 100 * (dev_correct / total_dev_items)
        else:
            dev_acc = 0
            dev_loss = 0

        eprint('Epoch {:04d} | AvgTrainLoss {:.4f} | TrainAcc {:.4f} | DevLoss {:.4f} | DevAcc {:.4f} | Time {:.4f}'.
               format(epoch, total_train_loss / ((len(train_seqs) // batch_size) + 1), 100 * (total_train_correct / total_train_items),
                      dev_loss, dev_acc, time.time() - c0))

    return model, cat_anc_to_ix, hv_anc_to_ix, hv_filler_to_ix, \
        cat_lc_to_ix, hv_lc_to_ix, jdecs_to_ix


def main(config):
    j_config = config['JModel']
    torch.manual_seed(j_config.getint('Seed'))
    save_pytorch = j_config.getboolean('SaveTorchModel')
    pytorch_fn = j_config.get('TorchFilename')
    extra_params_fn = j_config.get('ExtraParamsFilename')
    num_transformer_layers = j_config.getint('NumTransformerLayers')
    model, cat_anc_to_ix, hv_anc_to_ix, hv_filler_to_ix, cat_lc_to_ix, \
        hv_lc_to_ix, jdecs_to_ix = train(j_config)

    model.eval()

    params = list()
    params.extend([
        ('pre_attn_fc.weight', 'J P'),
        ('pre_attn_fc.bias', 'J p')
    ])
    for i in range(num_transformer_layers):
        params.extend([
            (f'transformer_layers.{i}.attn.in_proj_weight', f'J I {i}'),
            (f'transformer_layers.{i}.attn.in_proj_bias', f'J i {i}'),
            (f'transformer_layers.{i}.attn.out_proj.weight', f'J O {i}'),
            (f'transformer_layers.{i}.attn.out_proj.bias', f'J o {i}'),
            (f'transformer_layers.{i}.feedforward.weight', f'J F {i}'),
            (f'transformer_layers.{i}.feedforward.bias', f'J f {i}'),
        ])
    params.extend([ 
        ('output_fc.weight', 'J S'),
        ('output_fc.bias', 'J s')
    ])

    # TODO remove this
    eprint('State dict keys:', model.state_dict().keys())

    for param, prefix in params:
        if j_config.getboolean('UseGPU'):
            weights = model.state_dict()[param].data.cpu().numpy()
        else:
            weights = model.state_dict()[param].data.numpy()
        print(prefix, ','.join(map(str, weights.flatten('F').tolist())))

    # write out other info needed by the C++ code
    print('J H', model.num_heads)

    if not j_config.getboolean('AblateSyn'):
        if j_config.getboolean('UseGPU'):
            cat_anc_weights = model.state_dict()['cat_anc_embeds.weight'].data.cpu().numpy()
            cat_lc_weights = model.state_dict()['cat_lc_embeds.weight'].data.cpu().numpy()
        else:
            cat_anc_weights = model.state_dict()['cat_anc_embeds.weight'].data.numpy()
            cat_lc_weights = model.state_dict()['cat_lc_embeds.weight'].data.numpy()
        for cat, ix in sorted(cat_anc_to_ix.items()):
            print('C A ' + str(cat) + ' ' + ','.join(map(str, cat_anc_weights[ix])))
        for cat, ix in sorted(cat_lc_to_ix.items()):
            print('C L ' + str(cat) + ' ' + ','.join(map(str, cat_lc_weights[ix])))

    if not j_config.getboolean('AblateSem'):
        if j_config.getboolean('UseGPU'):
            hv_anc_weights = model.state_dict()['hv_anc_embeds.weight'].data.cpu().numpy()
            hv_filler_weights = model.state_dict()['hv_filler_embeds.weight'].data.cpu().numpy()
            hv_lc_weights = model.state_dict()['hv_lc_embeds.weight'].data.cpu().numpy()
        else:
            hv_anc_weights = model.state_dict()['hv_anc_embeds.weight'].data.numpy()
            hv_filler_weights = model.state_dict()['hv_filler_embeds.weight'].data.numpy()
            hv_lc_weights = model.state_dict()['hv_lc_embeds.weight'].data.numpy()

        for hvec, ix in sorted(hv_anc_to_ix.items()):
            print('K A ' + str(hvec) + ' ' + ','.join(map(str, hv_anc_weights[ix])))
        for hvec, ix in sorted(hv_filler_to_ix.items()):
            print('K F ' + str(hvec) + ' ' + ','.join(map(str, hv_filler_weights[ix])))
        for hvec, ix in sorted(hv_lc_to_ix.items()):
            print('K L ' + str(hvec) + ' ' + ','.join(map(str, hv_lc_weights[ix])))

    for jdec, ix in sorted(jdecs_to_ix.items()):
        print('j ' + str(ix) + ' ' + str(jdec))

    if save_pytorch:
        torch.save(model.state_dict(), pytorch_fn)
        # these are needed to initialize the model if we want to reload it
        extra_params = {
            'cat_anc_vocab_size': len(cat_anc_to_ix),
            'hv_anc_vocab_size': len(hv_anc_to_ix),
            'hv_filler_vocab_size': len(hv_filler_to_ix),
            'cat_lc_vocab_size': len(cat_lc_to_ix),
            'hv_lc_vocab_size': len(hv_lc_to_ix),
            'output_dim': len(jdecs_to_ix),
            'cat_anc_to_ix': cat_anc_to_ix,
            'hv_anc_to_ix': hv_anc_to_ix,
            'hv_filler_to_ix': hv_filler_to_ix,
            'cat_lc_to_ix': cat_lc_to_ix,
            'hv_lc_to_ix': hv_lc_to_ix,
            'jdecs_to_ix': jdecs_to_ix
        }
        pickle.dump(extra_params, open(extra_params_fn, 'wb'))


if __name__ == '__main__':
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(sys.argv[1])
    for section in config:
        eprint(section, dict(config[section]))
    main(config)

