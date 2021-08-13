import torch, pickle, configparser, sys, math
from transformerjmodel import TransformerJModel
from train_jmodel import eprint, get_jinfo_seqs, pad_target_matrix
from train_jmodel import prepare_dev_data as prepare_data

PAD = '0&J&J&J'

def main(config):
    j_config = config['JModel']
    use_gpu = j_config.getboolean('UseGPU')
    pytorch_fn = j_config.get('TorchFilename')
    extra_params_fn = j_config.get('ExtraParamsFilename')
    window_size = j_config.getint('AttnWindowSize')

    init_params = pickle.load(open(extra_params_fn, 'rb'))

    model = TransformerJModel(
                j_config=j_config,
                cat_anc_vocab_size=init_params['cat_anc_vocab_size'],
                hv_anc_vocab_size=init_params['hv_anc_vocab_size'],
                hv_filler_vocab_size=init_params['hv_filler_vocab_size'],
                cat_lc_vocab_size=init_params['cat_lc_vocab_size'],
                hv_lc_vocab_size=init_params['hv_lc_vocab_size'],
                output_dim=init_params['output_dim'],
    )

    if use_gpu:
        model = model.cuda()

    model.eval()
    model.load_state_dict(torch.load(pytorch_fn))

    per_sentence_jinfo = prepare_data(
        sys.stdin, 
        init_params['cat_anc_to_ix'], 
        init_params['hv_anc_to_ix'], 
        init_params['hv_filler_to_ix'],
        init_params['cat_lc_to_ix'],
        init_params['hv_lc_to_ix'],
        init_params['jdecs_to_ix']
    )
    seqs = get_jinfo_seqs(per_sentence_jinfo, window_size)

    jdecs_to_ix = init_params['jdecs_to_ix']

    target = [[ji.jdec for ji in seq] for seq in seqs]
    # L x N
    target = pad_target_matrix(target, jdecs_to_ix[PAD])
    if use_gpu:
        target = target.to('cuda')

    # L x N x E
    output = model(seqs, verbose=True)


    # iterate over sequences
    #for i in range(target.shape[1]):
    for i, seq in enumerate(seqs):
        print(" ======== new sequence ======== ")
        # iterate over jdecs in a sequence
        #for j in range(target.shape[0]):
        for j, jinfo in enumerate(seq):
            print( "\n ==== word {} ==== ".format(j))
            print("J cat anc:", jinfo.raw_cat_anc)
            print("J hv anc:", jinfo.raw_hv_anc)
            print("J hv filler:", jinfo.raw_hv_filler)
            print("J cat lc:", jinfo.raw_cat_lc)
            print("J hv lc:", jinfo.raw_hv_lc)
            jdec_id = target[j, i].item()
            assert jdec_id == jdecs_to_ix[jinfo.raw_jdec]
            print("JEOO:", jinfo.raw_jdec, "probability:", math.e**(output[j, i, jdec_id].item()))

if __name__ == '__main__':
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(sys.argv[1])
    for section in config:
        eprint(section, dict(config[section]))
    main(config)

