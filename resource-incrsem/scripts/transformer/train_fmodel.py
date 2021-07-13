import sys, configparser, torch, re, os, time, random, pickle
from collections import Counter
import torch.nn as nn
import torch.optim as optim

from transformerfmodel import TransformerFModel

#PAD = '<PAD>'
# The C++ code expects a certain format for each fdec, so this is
# a "fake" fdec used for padding
PAD = '0&&PAD'


# Dummy class for recording information about each F decision
class FInfo:
    pass

# Records predicate context vectors and syntactic
# categories for derivation fragments in the store state
class DerivationFragment:
    def __init__(self, hvec, category, depth):
        self.hvbase = hvec
        self.catbase = category
        self.depth = depth

    def __str__(self):
        return 'hvbase:{} catbase:{} depth:{}'.format(self.hvbase, self.catbase, self.depth)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def extract_first_kvec(hvec):
    if hvec == '':
        return ['']
    match = re.findall(r'^\[(.*?)\]', hvec)
    return match[0].split(',')


def parse_stack(stack_str):
    '''
    Given a store state (stack) from decpars file, returns a list of
    DerivationFragments containing categories and predicate context vectors
    '''
    # example inputs:
    # 1. Q[R-aN-bN:in_0][R-aN-bN:in_1][R-aN-bN:in_2]:R-aN/[R-aN-bN:in_2]:N;
    # 2. Q[R-aN-bN:in_0][R-aN-bN:in_1][R-aN-bN:in_2]:R-aN/[R-aN-bN:in_2]:N;:T;:T;["]:N-hV-hO/[N-b{N-aD}:an_2][]:N-aD-hV-hO;
    # 3. Q[R-aN-bN:in_0][R-aN-bN:in_1][R-aN-bN:in_2]:R-aN/[R-aN-bN:in_2]:N;[N-b{N-aD}:an_2,A-aN-x:!num!_1,N-aD-bV-bO:review_1][N-aD-bV-bO:review_2][N-aD-bV-bO:review_3][N-aD-bV-bO:review_4]:N/[][B-aN-bN:!unk!_1,B-aN-bN:!unk!_2,N-aD:classic_1]:V-aN;[B-aN-bN:take_0][B-aN-bN:take_1][B-aN-bN:take_2]:V-aN/[B-aN-bN:take_2]:N;

    stack = list()
    stack.append(DerivationFragment(['Top'], 'T', 0))
    
#    if stack_str == 'Q':
#        return []
    assert stack_str[0] == 'Q', stack_str
    stack_str = stack_str[1:]

#    # TODO find a better solution for this
#    # this is a very hacky fix for unusual cases like this one from WSJ:
#    # :T;["]:N-h{I-aN}/[N-b{N-aD}:a_2][]:N-aD-h{I-aN};
#    if stack_str.find(';') < stack_str.find('/'):
#        stack_str = stack_str[stack_str.find(';')+1:]
    # to find apices, use this regex:
    # re.findall(^([^/;]+;)*([^/;]+)|;([^/;]+;)*([^/;]+))
    # re.split(r'(?<=[^\\])/', r'a\/b/c') --> ['a\\/b', 'c']
    #fragment_bases = re.findall('/([^/;]+/)*([^/;]+)', stack_str)
    fragment_bases = re.findall(r'(?<=[^\\])/((\\/|[^/;])+(?<=[^\\])/)*((\\/|[^/;])+)', stack_str)
    # the third group contains the base
    fragment_bases = [group[2] for group in fragment_bases]

    curr_depth = 1
    #eprint('stack str:', stack_str)
    for base in fragment_bases:
        #eprint('base:', base)
        colon_ind = base.rfind(':')
        predcon = base[:colon_ind]
        category = base[colon_ind+1:]
        base_hvec = extract_first_kvec(predcon)
        df = DerivationFragment(base_hvec, category, curr_depth)
        stack.append(df)
        curr_depth += 1
    return stack


def hvecIxReplace(hvec, hvec_to_ix):
    new_hvec = list()
    top_count = 0
    for hv in hvec:
        if hv == 'Top':
            top_count += 1
        elif hv not in ['Bot', '']:
            new_hvec.append(hvec_to_ix[hv])
    return new_hvec, top_count


def _initialize_finfo_list(infile):
    all_finfo = list()
    for line in infile:
        # TODO update linetrees2trdecpars so that it doesn't include pointless info
        depth, _, _, hv_filler, hv_ante, nulla, fdec, stack = line.split()
        depth = int(depth)
        curr_finfo = FInfo()
        curr_finfo.hvf = extract_first_kvec(hv_filler)
        curr_finfo.hva = extract_first_kvec(hv_ante)
        curr_finfo.nulla = int(nulla)
        # TODO is there any downside to not splitting the fdec into its
        # constitutents (match and hvec)?
        curr_finfo.fdec = fdec
        curr_finfo.stack = parse_stack(stack)
        assert depth == 0 or curr_finfo.stack[-1].depth == depth, 'line: {}; depth from trdecpars: {}; depth from stack: {}'.format(line, depth, curr_finfo.stack[-1])
        all_finfo.append(curr_finfo)
    return all_finfo


def _map_finfo_list_to_ix(all_finfo, catb_to_ix, fdecs_to_ix, hvb_to_ix, hvf_to_ix, hva_to_ix, dev=False):
    '''
    Replaces strings inside FInfo objects with their corresponding indices.
    These modifications are done in place rather than creating new
    Finfo objects.
    '''

    new_finfo = list()
    for finfo in all_finfo:
        try:
            finfo.fdec = fdecs_to_ix[finfo.fdec]
            finfo.hvf, finfo.hvf_top = hvecIxReplace(finfo.hvf, hvf_to_ix)
            finfo.hva, finfo.hva_top = hvecIxReplace(finfo.hva, hva_to_ix)
            for df in finfo.stack:
                df.catbase = catb_to_ix[df.catbase]
                df.hvbase, df.hvbase_top = hvecIxReplace(df.hvbase, hvb_to_ix)
        except KeyError:
            # dev may contain fdecs, catbases, etc that haven't appeared in
            # training data. Throw out any such data
            if dev: continue
            else: raise
        new_finfo.append(finfo)
    return new_finfo


def prepare_data():
    all_finfo = _initialize_finfo_list(sys.stdin)

#    all_finfo = list()
#    
#    for line in sys.stdin:
#        # TODO update linetrees2trdecpars so that it doesn't include pointless info
#        depth, _, _, hv_filler, hv_ante, nulla, fdec, stack = line.split()
#        depth = int(depth)
#        curr_finfo = FInfo()
#        curr_finfo.hvf = extract_first_kvec(hv_filler)
#        curr_finfo.hva = extract_first_kvec(hv_ante)
#        curr_finfo.nulla = int(nulla)
#        # TODO is there any downside to not splitting the fdec into its
#        # constitutents (match and hvec)?
#        curr_finfo.fdec = fdec
#        curr_finfo.stack = parse_stack(stack)
#        assert depth == 0 or curr_finfo.stack[-1].depth == depth, 'line: {}; depth from trdecpars: {}; depth from stack: {}'.format(line, depth, curr_finfo.stack[-1])
#        all_finfo.append(curr_finfo)
#        
    all_hvf = set()
    all_hva = set()
    all_fdecs = set()
    all_hvb = set()
    all_catb = set()
    for finfo in all_finfo:
        all_hvf.update(set(finfo.hvf))
        all_hva.update(set(finfo.hva))
        all_fdecs.add(finfo.fdec)
        for df in finfo.stack:
            hvb = df.hvbase
            catb = df.catbase
            all_hvb.update(set(hvb))
            all_catb.add(catb)

    catb_to_ix = {cat: i for i, cat in enumerate(sorted(all_catb))}
    fdecs_to_ix = {fdecs: i for i, fdecs in enumerate(sorted(all_fdecs))}
    # ID for padding symbol added at end of sequence
    fdecs_to_ix[PAD] = len(fdecs_to_ix)
    hvb_to_ix = {hvec: i for i, hvec in enumerate(sorted(all_hvb))}
    hvf_to_ix = {hvec: i for i, hvec in enumerate(sorted(all_hvf))}
    hva_to_ix = {hvec: i for i, hvec in enumerate(sorted(all_hva))}

    # replace strings with indices
    all_finfo = _map_finfo_list_to_ix(all_finfo, catb_to_ix, fdecs_to_ix, 
        hvb_to_ix, hvf_to_ix, hva_to_ix)
#    for finfo in all_finfo:
#        finfo.fdec = fdecs_to_ix[finfo.fdec]
#        finfo.hvf, finfo.hvf_top = hvecIxReplace(finfo.hvf, hvf_to_ix)
#        finfo.hva, finfo.hva_top = hvecIxReplace(finfo.hva, hva_to_ix)
#        for df in finfo.stack:
#            df.catbase = catb_to_ix[df.catbase]
#            df.hvbase, df.hvbase_top = hvecIxReplace(df.hvbase, hvb_to_ix)
    return all_finfo, catb_to_ix, fdecs_to_ix, hvb_to_ix, hvf_to_ix, hva_to_ix


def prepare_dev_data(dev_decpars, catb_to_ix, fdecs_to_ix, hvb_to_ix, hvf_to_ix, hva_to_ix):
    all_finfo = _initialize_finfo_list(open(dev_decpars))
    all_finfo = _map_finfo_list_to_ix(all_finfo, catb_to_ix, fdecs_to_ix, 
        hvb_to_ix, hvf_to_ix, hva_to_ix, dev=True)
    return all_finfo


def train(f_config):
    use_dev = f_config.getboolean('UseDev')
    dev_decpars = f_config.get('DevFile')
    use_gpu = f_config.getboolean('UseGPU')
    window_size = f_config.getint('AttnWindowSize')
    batch_size = f_config.getint('BatchSize')
    epochs = f_config.getint('NEpochs')
    l2_reg = f_config.getboolean('L2Reg')
    learning_rate = f_config.getfloat('LearningRate')
    weight_decay = f_config.getfloat('WeightDecay')

    all_finfo, catb_to_ix, fdecs_to_ix, hvb_to_ix, hvf_to_ix, hva_to_ix = prepare_data()
    if use_dev:
        dev_all_finfo = prepare_dev_data(dev_decpars, catb_to_ix, fdecs_to_ix,
             hvb_to_ix, hvf_to_ix, hva_to_ix)

    model = TransformerFModel(
                f_config=f_config,
                catb_vocab_size=len(catb_to_ix),
                hvb_vocab_size=len(hvb_to_ix),
                hvf_vocab_size=len(hvf_to_ix),
                hva_vocab_size=len(hva_to_ix), 
                output_dim=len(fdecs_to_ix)
    )


    if use_gpu:
        model = model.cuda()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    # TODO add validation on dev set
    for epoch in range(epochs):
        model.train()
        c0 = time.time()
        random.shuffle(all_finfo)
        total_train_correct = 0
        total_train_loss = 0

        for j in range(0, len(all_finfo), batch_size):
            if use_gpu:
                l2_loss = torch.cuda.FloatTensor([0])
            else:
                l2_loss = torch.FloatTensor([0])
            for param in model.parameters():
                if torch.numel(param) == 0:
                    continue
                l2_loss += torch.mean(param.pow(2))

            batch = all_finfo[j:j+batch_size]
            batch_target = torch.LongTensor([fi.fdec for fi in batch])
            if use_gpu:
                batch_target = batch_target.to("cuda")

            # output dimension: N x E
            # N: batch size
            # E: output dimensionality (number of classes)
            output = model(batch)

            _, fdec = torch.max(output.data, 1)

            train_correct = (fdec == batch_target).sum().item()
            total_train_correct += train_correct
            nll_loss = criterion(output, batch_target)
            loss = nll_loss + l2_reg * l2_loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if use_dev:
            with torch.no_grad():
                model.eval()
                dev_target = torch.LongTensor(
                    [fi.fdec for fi in dev_all_finfo]
                )
                if use_gpu:
                    dev_target = dev_target.to('cuda')
                dev_output = model(dev_all_finfo)
                _, dev_fdec = torch.max(dev_output.data, 1)
                dev_correct = (dev_fdec == dev_target).sum().item()
                dev_loss = criterion(dev_output, dev_target).item()
                dev_acc = 100 * (dev_correct / len(dev_all_finfo))
        else:
            dev_acc = 0
            dev_loss = 0
            
        eprint('Epoch {:04d} | AvgTrainLoss {:.4f} | TrainAcc {:.4f} | DevLoss {:.4f} | DevAcc {:.4f} | Time {:.4f}'.
               format(epoch, total_train_loss / ((len(all_finfo) // batch_size) + 1), 100 * (total_train_correct / len(all_finfo)),
                      dev_loss, dev_acc, time.time() - c0))

    return model, catb_to_ix, fdecs_to_ix, hvb_to_ix, hvf_to_ix, hva_to_ix


def main(config):
    f_config = config['FModel']
    torch.manual_seed(f_config.getint('Seed'))
    save_pytorch = f_config.getboolean('SaveTorchModel')
    pytorch_fn = f_config.get('TorchFilename')
    extra_params_fn = f_config.get('ExtraParamsFilename')
    model, catb_to_ix, fdecs_to_ix, hvb_to_ix, hvf_to_ix, hva_to_ix = train(f_config)

    model.eval()
    params = [ 
        #('query.weight', 'F Q'),
        #('query.bias', 'F q'),
        #('key.weight', 'F K' ),
        #('key.bias', 'F k'),
        #('value.weight', 'F V' ),
        #('value.bias', 'F v'),
        ('pre_attn_fc.weight', 'F P'),
        ('pre_attn_fc.bias', 'F p'),
        ('attn.in_proj_weight', 'F I'),
        ('attn.in_proj_bias', 'F i'),
        ('attn.out_proj.weight', 'F O'),
        ('attn.out_proj.bias', 'F o'),
        ('fc1.weight', 'F F'),
        ('fc1.bias', 'F f'),
        ('fc2.weight', 'F S'),
        ('fc2.bias', 'F s')
    ]

    for param, prefix in params:
        if f_config.getboolean('UseGPU'):
            weights = model.state_dict()[param].data.cpu().numpy()
        else:
            weights = model.state_dict()[param].data.numpy()
        print(prefix, ','.join(map(str, weights.flatten('F').tolist())))

    if not f_config.getboolean('AblateSyn'):
        if f_config.getboolean('UseGPU'):
            weights = model.state_dict()['catb_embeds.weight'].data.cpu().numpy()
        else:
            weights = model.state_dict()['catb_embeds.weight'].data.numpy()
        for cat, ix in sorted(catb_to_ix.items()):
            print('C B ' + str(cat) + ' ' + ','.join(map(str, weights[ix])))

    if not f_config.getboolean('AblateSem'):
        if f_config.getboolean('UseGPU'):
            b_weights = model.state_dict()['hvb_embeds.weight'].data.cpu().numpy()
            f_weights = model.state_dict()['hvf_embeds.weight'].data.cpu().numpy()
            a_weights = model.state_dict()['hva_embeds.weight'].data.cpu().numpy()
        else:
            b_weights = model.state_dict()['hvb_embeds.weight'].data.numpy()
            f_weights = model.state_dict()['hvf_embeds.weight'].data.numpy()
            a_weights = model.state_dict()['hva_embeds.weight'].data.numpy()

        for hvec, ix in sorted(hvb_to_ix.items()):
            print('K B ' + str(hvec) + ' ' + ','.join(map(str, b_weights[ix])))
        for hvec, ix in sorted(hvf_to_ix.items()):
            print('K F ' + str(hvec) + ' ' + ','.join(map(str, f_weights[ix])))
        for hvec, ix in sorted(hva_to_ix.items()):
            print('K A ' + str(hvec) + ' ' + ','.join(map(str, a_weights[ix])))
        if len(hva_to_ix.items()) == 0:
            print('K A N-aD:ph_0 ' + '0,'*(f_config.getint('AntSize')-1)+'0') #add placeholder so model knows antecedent size

    for fdec, ix in sorted(fdecs_to_ix.items()):
        print('f ' + str(ix) + ' ' + str(fdec))

    if save_pytorch:
        torch.save(model.state_dict(), pytorch_fn)
        # these are needed to initialize the model if we want to reload it
        extra_params = {
            'catb_vocab_size': len(catb_to_ix),
            'hvb_vocab_size': len(hvb_to_ix),
            'hvf_vocab_size': len(hvf_to_ix),
            'hva_vocab_size': len(hva_to_ix),
            'output_dim': len(fdecs_to_ix)
        }
        pickle.dump(extra_params, open(extra_params_fn, 'wb'))


if __name__ == '__main__':
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(sys.argv[1])
    for section in config:
        eprint(section, dict(config[section]))
    main(config)

