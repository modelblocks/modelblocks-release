import torch, pickle, configparser, sys
from transformerjmodel import TransformerJModel
from train_jmodel import eprint


def main(config):
    j_config = config['JModel']
    pytorch_fn = j_config.get('TorchFilename')
    extra_params_fn = j_config.get('ExtraParamsFilename')
    j_config['UseGPU'] = 'no'

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

    model.eval()
    model.load_state_dict(torch.load(pytorch_fn))

    # S x N x E
    seq_length = 5
    # the ith dummy vector is filled with 0.1 * (i+1)
    # so  [0.1 0.1 0.1 ...], [0.2 0.2 0.2 ...], ...
    dummy_attn_input = torch.FloatTensor(seq_length, 1, 
        3*j_config.getint('SemDim') + 2*j_config.getint('SynDim') + model.max_depth)
    for i in range(seq_length):
        for j in range(len(dummy_attn_input[i, 0])):
            dummy_attn_input[i, 0, j] = 0.1 * (i+1)
    model.compute(dummy_attn_input, verbose=True)
#    dummy_pred = model.compute(dummy_attn_input, verbose=True)
#    # only look at the last word
#    jdec_log_scores = dummy_pred[-1, 0]
#    jdec_scores = torch.exp(jdec_log_scores)
#    jdec_scores_normalized = jdec_scores/sum(jdec_scores)
#    print('Dummy output:')
#    for jdec_prob in jdec_scores_normalized:
#        print(jdec_prob.item())

    
if __name__ == '__main__':
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(sys.argv[1])
    for section in config:
        eprint(section, dict(config[section]))
    main(config)

