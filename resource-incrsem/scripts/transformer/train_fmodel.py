import sys, configparser, torch, re, os, time, random
from collections import Counter
import torch.nn as nn
import torch.optim as optim

from transformerfmodel import TransformerFModel

PAD = '<PAD>'


# Dummy class for recording information about each F decision
class FInfo:
    pass


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def extract_first_kvec(hvec):
    match = re.findall(r'^\[(.*?)\]', hvec)
    return match[0].split(',')


def hvecIxReplace(hvec, hvec_to_ix):
    new_hvec = list()
    top_count = 0
    for hv in hvec:
        if hv == 'Top':
            top_count += 1
        elif hv not in ['Bot', '']:
            new_hvec.append(hvec_to_ix[hv])
    return new_hvec, top_count


def prepare_data():
    # list of lists; each outer list is an article
    per_article_finfo = list()
    
    # list of f decisions for the current article
    curr_finfo = list()
    
    is_first_line = True
    for line in sys.stdin:
        is_new_article, depth, catb, hv_base, hv_filler, hv_ante, nulla, fdec = line.split()
        is_new_article = bool(int(is_new_article))
        if is_new_article:
            if is_first_line:
                is_first_line = False
            else:
                per_article_finfo.append(curr_finfo)
                curr_finfo = list()

        finfo = FInfo()
        finfo.depth = int(depth)
        finfo.catb = catb
        finfo.hvb = extract_first_kvec(hv_base)
        finfo.hvf = extract_first_kvec(hv_filler)
        finfo.hva = extract_first_kvec(hv_ante)
        finfo.nulla = int(nulla)
        # TODO is there any downside to not splitting the fdec into its
        # constitutents (match and hvec)?
        finfo.fdec = fdec
        curr_finfo.append(finfo)

    per_article_finfo.append(curr_finfo)
           
    all_hvb, all_hvf, all_hva, all_fdecs, all_catb = [set()] * 5
    for article in per_article_finfo:
        for finfo in article:
            all_hvb.update(set(finfo.hvb))
            all_hvf.update(set(finfo.hvf))
            all_hva.update(set(finfo.hva))
            all_fdecs.add(finfo.fdec)
            all_catb.add(finfo.catb)
                
    catb_to_ix = {cat: i for i, cat in enumerate(sorted(all_catb))}
    fdecs_to_ix = {fdecs: i for i, fdecs in enumerate(sorted(all_fdecs))}
    # ID for padding symbol added at end of sequence
    fdecs_to_ix[PAD] = len(fdecs_to_ix)
    hvb_to_ix = {hvec: i for i, hvec in enumerate(sorted(all_hvb))}
    hvf_to_ix = {hvec: i for i, hvec in enumerate(sorted(all_hvf))}
    hva_to_ix = {hvec: i for i, hvec in enumerate(sorted(all_hva))}

    # replace strings with indices
    for article in per_article_finfo:
        for finfo in article:
            finfo.catb = catb_to_ix[finfo.catb]
            finfo.fdec = fdecs_to_ix[finfo.fdec]
            finfo.hvb, finfo.hvb_top = hvecIxReplace(finfo.hvb, hvb_to_ix)
            finfo.hvf, finfo.hvf_top = hvecIxReplace(finfo.hvf, hvf_to_ix)
            finfo.hva, finfo.hva_top = hvecIxReplace(finfo.hva, hva_to_ix)
    return per_article_finfo, catb_to_ix, fdecs_to_ix, hvb_to_ix, hvf_to_ix, hva_to_ix


def get_train_seqs(per_article_finfo, window_size):
    train_seqs = list()
    for article in per_article_finfo:
        for i in range(0, len(article), window_size):
            train_seqs.append(article[i:i+window_size])
    return train_seqs


def pad_target_matrix(targets, symbol):
    '''
    Given a list of per-sequence fDec targets and a padding symbol,
    returns an SxN padded matrix of targets.
    S is max sequence length, and N in number of sequences
    '''
    max_seq_length = max(len(t) for t in targets)
    padded_targets = list()
    for t in targets:
        padded_targets.append(t + [symbol]*(max_seq_length - len(t)))
    return torch.transpose(torch.LongTensor(padded_targets), 0, 1)


def train(f_config):
    per_article_finfo, catb_to_ix, fdecs_to_ix, hvb_to_ix, hvf_to_ix, hva_to_ix = prepare_data()
    model = TransformerFModel(
                f_config=f_config,
                catb_vocab_size=len(catb_to_ix),
                hvb_vocab_size=len(hvb_to_ix),
                hvf_vocab_size=len(hvf_to_ix),
                hva_vocab_size=len(hva_to_ix), 
                output_dim=len(fdecs_to_ix)
    )

    use_gpu = f_config.getboolean('UseGPU')
    window_size = f_config.getint('AttnWindowSize')
    batch_size = f_config.getint('BatchSize')
    epochs = f_config.getint('NEpochs')
    l2_reg = f_config.getfloat('L2Reg')
    learning_rate = f_config.getfloat('LearningRate')
    weight_decay = f_config.getfloat('WeightDecay')

    if use_gpu:
        model = model.cuda()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_seqs = get_train_seqs(per_article_finfo, window_size)
    model.train()

    # TODO add validation on dev set
    for epoch in range(epochs):
        c0 = time.time()
        random.shuffle(train_seqs)
        total_train_correct = 0
        total_train_loss = 0
        total_dev_loss = 0
        # TODO change these
        dev_acc = 0
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
            target = [[fi.fdec for fi in seq] for seq in batch]
            # S x N
            target = pad_target_matrix(target, fdecs_to_ix[PAD])
            if use_gpu:
                target = target.to('cuda')

            # output dimension is S x N x E
            # S: window size
            # N: batch size
            # E: output size
            output = model(batch)

            # fdec dimension is S x N
            _, fdec = torch.max(output.data, 2)
            # TODO don't include padding in loss
            train_correct = (fdec == target).sum().item()
            total_train_correct += train_correct
            # TODO change this
            total_train_items += target.numel()

            # have to rearrange dimensions for NLLLoss function
            # output dimension becomes N x E x S
            output = torch.transpose(output, 0, 2) # SxNxE --> ExNxS
            output = torch.transpose(output, 0, 1) # ExNxS --> NxExS

            # target dimension becomes N x S
            target = torch.transpose(target, 0, 1)
            nll_loss = criterion(output, target)
            loss = nll_loss + l2_reg * l2_loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        eprint('Epoch {:04d} | AvgTrainLoss {:.4f} | TrainAcc {:.4f} | DevLoss {:.4f} | DevAcc {:.4f} | Time {:.4f}'.
               format(epoch, total_train_loss / ((len(train_seqs) // batch_size) + 1), 100 * (total_train_correct / total_train_items),
                      total_dev_loss, dev_acc, time.time() - c0))

    return model, catb_to_ix, fdecs_to_ix, hvb_to_ix, hvf_to_ix, hva_to_ix


def main(config):
    f_config = config['FModel']
    torch.manual_seed(f_config.getint('Seed'))
    model, catb_to_ix, fdecs_to_ix, hvb_to_ix, hvf_to_ix, hva_to_ix = train(f_config)

    model.eval()
    if f_config.getboolean('UseGPU'):
        catb_embeds = model.state_dict()['catb_embeds.weight'].data.cpu().numpy()
        hvecb_embeds = model.state_dict()['hvb_embeds.weight'].data.cpu().numpy()
        hvecf_embeds = model.state_dict()['hvf_embeds.weight'].data.cpu().numpy()
        hveca_embeds = model.state_dict()['hva_embeds.weight'].data.cpu().numpy()
        query_weights = model.state_dict()['query.weight'].data.cpu().numpy()
        key_weights = model.state_dict()['key.weight'].data.cpu().numpy()
        value_weights = model.state_dict()['value.weight'].data.cpu().numpy()
        fc1_weights = model.state_dict()['fc1.weight'].data.cpu().numpy()
        fc1_biases = model.state_dict()['fc1.bias'].data.cpu().numpy()
        fc2_weights = model.state_dict()['fc2.weight'].data.cpu().numpy()
        fc2_biases = model.state_dict()['fc2.bias'].data.cpu().numpy()
    else:
        catb_embeds = model.state_dict()['catb_embeds.weight'].data.numpy()
        hvecb_embeds = model.state_dict()['hvb_embeds.weight'].data.numpy()
        hvecf_embeds = model.state_dict()['hvf_embeds.weight'].data.numpy()
        hveca_embeds = model.state_dict()['hva_embeds.weight'].data.numpy()
        query_weights = model.state_dict()['query.weight'].data.numpy()
        key_weights = model.state_dict()['key.weight'].data.numpy()
        value_weights = model.state_dict()['value.weight'].data.numpy()
        fc1_weights = model.state_dict()['fc1.weight'].data.numpy()
        fc1_biases = model.state_dict()['fc1.bias'].data.numpy()
        fc2_weights = model.state_dict()['fc2.weight'].data.numpy()
        fc2_biases = model.state_dict()['fc2.bias'].data.numpy()

    print('F Q ' + ','.join(map(str, query_weights.flatten('F').tolist())))
    print('F K ' + ','.join(map(str, key_weights.flatten('F').tolist())))
    print('F V ' + ','.join(map(str, value_weights.flatten('F').tolist())))
    print('F F ' + ','.join(map(str, fc1_weights.flatten('F').tolist())))
    print('F f ' + ','.join(map(str, fc1_biases.flatten('F').tolist())))
    print('F S ' + ','.join(map(str, fc2_weights.flatten('F').tolist())))
    print('F s ' + ','.join(map(str, fc2_biases.flatten('F').tolist())))

    if not f_config.getboolean('AblateSyn'):
        for cat, ix in sorted(catb_to_ix.items()):
            print('C B ' + str(cat) + ' ' + ','.join(map(str, catb_embeds[ix])))
    if not f_config.getboolean('AblateSem'):
        for hvec, ix in sorted(hvb_to_ix.items()):
            print('K B ' + str(hvec) + ' ' + ','.join(map(str, hvecb_embeds[ix])))
        for hvec, ix in sorted(hvf_to_ix.items()):
            print('K F ' + str(hvec) + ' ' + ','.join(map(str, hvecf_embeds[ix])))
        for hvec, ix in sorted(hva_to_ix.items()):
            print('K A ' + str(hvec) + ' ' + ','.join(map(str, hveca_embeds[ix])))
        if len(hva_to_ix.items()) == 0:
            print('K A N-aD:ph_0 ' + '0,'*(f_config.getint('AntSize')-1)+'0') #add placeholder so model knows antecedent size
    for fdec, ix in sorted(fdecs_to_ix.items()):
        print('f ' + str(ix) + ' ' + str(fdec))


if __name__ == '__main__':
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(sys.argv[1])
    for section in config:
        eprint(section, dict(config[section]))
    main(config)

