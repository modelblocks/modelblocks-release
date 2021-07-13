import torch, pickle, configparser, sys
from transformerfmodel import TransformerFModel
from train_fmodel import eprint


def main(config):
    f_config = config['FModel']
    #use_gpu = f_config.getboolean('UseGPU')
    pytorch_fn = f_config.get('TorchFilename')
    extra_params_fn = f_config.get('ExtraParamsFilename')
    f_config['UseGPU'] = 'no'

    init_params = pickle.load(open(extra_params_fn, 'rb'))

    model = TransformerFModel(
                f_config=f_config,
                catb_vocab_size=init_params['catb_vocab_size'],
                hvb_vocab_size=init_params['hvb_vocab_size'],
                hvf_vocab_size=init_params['hvf_vocab_size'],
                hva_vocab_size=init_params['hva_vocab_size'],
                output_dim=init_params['output_dim']
    )

    model.eval()
    model.load_state_dict(torch.load(pytorch_fn))

    # S x N x E
    stack_length = 5
    dummy_attn_input = torch.FloatTensor(stack_length, 1, 
        f_config.getint('SemDim') + f_config.getint('SynDim') + model.max_depth+1)

    dummy_post_attn_input = torch.ones(
        1,
        f_config.getint('AntDim') + f_config.getint('SemDim') + 1
    )
    for i in range(stack_length):
        for j in range(len(dummy_attn_input[i, 0])):
            dummy_attn_input[i, 0, j] = 0.1 * (i+1)
    fdec_log_scores = model.compute(dummy_attn_input, dummy_post_attn_input, verbose=True)
    fdec_log_scores = torch.reshape(fdec_log_scores, (-1,))
    fdec_scores = torch.exp(fdec_log_scores)
    fdec_scores_normalized = fdec_scores/sum(fdec_scores)
    eprint('Dummy output:')
    for fdec_prob in fdec_scores_normalized:
        eprint(fdec_prob.item())

    
if __name__ == '__main__':
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(sys.argv[1])
    for section in config:
        eprint(section, dict(config[section]))
    main(config)

