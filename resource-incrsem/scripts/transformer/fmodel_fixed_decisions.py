import torch, pickle, configparser, sys, math
from transformerfmodel import TransformerFModel, eprint
from train_fmodel import get_finfo_seqs, pad_target_matrix
from train_fmodel import prepare_dev_data as prepare_data

PAD = '0&&PAD'

def main(config):
    f_config = config['FModel']
    use_gpu = f_config.getboolean('UseGPU')
    pytorch_fn = f_config.get('TorchFilename')
    extra_params_fn = f_config.get('ExtraParamsFilename')
    window_size = f_config.getint('AttnWindowSize')

    init_params = pickle.load(open(extra_params_fn, 'rb'))

    model = TransformerFModel(
                f_config=f_config,
                catb_vocab_size=len(init_params['catb_to_ix']),
                hvb_vocab_size=len(init_params['hvb_to_ix']),
                hvf_vocab_size=len(init_params['hvf_to_ix']),
                hva_vocab_size=len(init_params['hva_to_ix']),
                output_dim=len(init_params['fdecs_to_ix'])
    )

    if use_gpu:
        model = model.cuda()

    model.eval()
    model.load_state_dict(torch.load(pytorch_fn))

    per_sentence_finfo = prepare_data(
        sys.stdin, 
        init_params['catb_to_ix'],
        init_params['fdecs_to_ix'],
        init_params['hvb_to_ix'],
        init_params['hvf_to_ix'],
        init_params['hva_to_ix']
    )

    seqs = get_finfo_seqs(per_sentence_finfo, window_size)

    fdecs_to_ix = init_params['fdecs_to_ix']

    target = [[fi.fdec for fi in seq] for seq in seqs]

    # L x N
    target = pad_target_matrix(target, fdecs_to_ix[PAD])
    if use_gpu:
        target = target.to('cuda')

    # L x N x E
    output = model(seqs, verbose=True)

    # iterate over sequences
    #for i in range(target.shape[1]):
    for i, seq in enumerate(seqs):
        print(" ======== new sequence ======== ")
        # iterate over fdecs in a sequence
        #for j in range(target.shape[0]):
        for j, finfo in enumerate(seq):
            print( "\n ==== word {} ==== ".format(j))
#            print("J cat anc:", jinfo.raw_cat_anc)
#            print("J hv anc:", jinfo.raw_hv_anc)
#            print("J hv filler:", jinfo.raw_hv_filler)
#            print("J cat lc:", jinfo.raw_cat_lc)
#            print("J hv lc:", jinfo.raw_hv_lc)
            fdec_id = target[j, i].item()
            assert fdec_id == fdecs_to_ix[finfo.raw_fdec]
            print("FEK:", finfo.raw_fdec, "probability:", math.e**(output[j, i, fdec_id].item()))

if __name__ == '__main__':
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(sys.argv[1])
    for section in config:
        eprint(section, dict(config[section]))
    main(config)

