import torch, math, sys
import torch.nn as nn
import torch.nn.functional as F

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# source:
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerFModel(nn.Module):
    def __init__(self, f_config, catb_vocab_size, hvb_vocab_size,
                 hvf_vocab_size, hva_vocab_size, output_dim):
        super(TransformerFModel, self).__init__()
        self.syn_dim = f_config.getint('SynDim')
        self.sem_dim = f_config.getint('SemDim')
        self.ant_dim = f_config.getint('AntDim')
        self.hidden_dim = f_config.getint('HiddenDim')
        self.dropout_prob = f_config.getfloat('DropoutProb')
        self.ablate_syn = f_config.getboolean('AblateSyn')
        self.ablate_sem = f_config.getboolean('AblateSem')
        self.use_positional_encoding = f_config.getboolean('UsePositionalEncoding')
        # TODO this is int in the old config -- why?
        self.use_gpu = f_config.getboolean('UseGPU')
        # TODO add num_blocks option for multiple transformer blocks
        self.num_heads = f_config.getint('NumHeads')
        self.attn_dim = f_config.getint('AttnDim')
        #self.attn_q_dim = f_config.getint('AttnQDim')
        #self.attn_k_dim = f_config.getint('AttnKDim')
        #self.attn_v_dim = f_config.getint('AttnVDim')
        # TODO decide whether to use this at runtime or just for
        # training
        self.attn_window_size = f_config.getint('AttnWindowSize')
        self.hvb_vocab_size = hvb_vocab_size
        self.hvf_vocab_size = hvf_vocab_size
        self.hva_vocab_size = hva_vocab_size
        self.output_dim = output_dim
        self.max_depth = 7

        # TODO should there just be one big embedding? inputs would be
        # multi-hot; not sure if that's easy
        self.catb_embeds = nn.Embedding(catb_vocab_size, self.syn_dim)
        self.hvb_embeds = nn.Embedding(hvb_vocab_size, self.sem_dim)
        self.hvf_embeds = nn.Embedding(hvf_vocab_size, self.sem_dim)
        self.hva_embeds = nn.Embedding(hva_vocab_size, self.ant_dim)

        # attn intput is (one-hot) depth, base syn cat embedding,
        # base hvec embedding, and filler hvec embedding
        attn_input_dim = 7 + self.syn_dim + 2*self.sem_dim
        #self.query = nn.Linear(attn_input_dim, self.attn_q_dim, bias=True)
        #self.key = nn.Linear(attn_input_dim, self.attn_k_dim, bias=True)
        #self.value = nn.Linear(attn_input_dim, self.attn_v_dim, bias=True)
        #self.query = nn.Linear(attn_input_dim, self.attn_dim, bias=True)
        #self.key = nn.Linear(attn_input_dim, self.attn_dim, bias=True)
        #self.value = nn.Linear(attn_input_dim, self.attn_dim, bias=True)

        # project the attention input to the right dimensionality
        self.pre_attn_fc = nn.Linear(attn_input_dim, self.attn_dim, bias=True)

        # this layer adds positional encodings to the output of pre_attn_fc
        # its output is used for queries, keys, and values
        self.positional_encoding = PositionalEncoding(self.attn_dim)

        # TODO define this as a block so you can have multiple attention layers
        self.attn =  nn.MultiheadAttention(
            embed_dim=self.attn_dim,
            #embed_dim=self.attn_q_dim,
            num_heads=self.num_heads,
            #kdim=self.attn_k_dim,
            #vdim=self.attn_v_dim,
            bias=True
        )

        # TODO make this resnet?
        # the input to fc1 is the output of the attention layer concatenated
        # with the antecedent hVec embedding and the null antecedent indicator
        # (hence the +1)
        self.fc1 = nn.Linear(self.attn_dim+self.ant_dim+1, self.hidden_dim, bias=True)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.relu = F.relu
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim, bias=True)


    def get_sparse_hv_matrix(self, hv, vocab_size):
        rows = list()
        cols = list()
        for i, h in enumerate(hv):
            for predcon in h:
                rows.append(i)
                cols.append(predcon)
        values = [1 for i in range(len(rows))]
        hv_sparse =  torch.sparse_coo_tensor(
            [rows, cols], values, (len(hv), vocab_size), dtype=torch.float
        )
        if self.use_gpu:
            hv_sparse = hv_sparse.to('cuda')
        return hv_sparse

    
    def get_per_sequence_x(self, batch_finfo):
        #per_sequence_x = list()
        per_seq_attn_input = list()
        per_seq_coref_emb = list()
        for seq in batch_finfo:
            # base category embeddings
            if self.ablate_syn:
                catb_embed = torch.zeros(
                    [len(seq), self.syn_dim],
                    dtype=torch.float
                )
            else:
                catb = [fi.catb for fi in seq]
                catb = torch.LongTensor(catb)
                if self.use_gpu:
                    catb = catb.to('cuda')
                catb_embed = self.catb_embeds(catb)


            # base, filler, antecedent hvec embeddings
            # TODO fix this -- max_seq_length isn't defined
            if self.ablate_sem:
                hvb_embed = torch.zeros(
                    [max_seq_length, batch_size, self.sem_dim], dtype=torch.float
                )
                hvf_embed = torch.zeros(
                    [max_seq_length, batch_size, self.sem_dim], dtype=torch.float
                )
                hva_embed = torch.zeros(
                    [max_seq_length, batch_size, self.ant_dim], dtype=torch.float
                )

            else:
                hvb = [fi.hvb for fi in seq]
                hvb_sparse = self.get_sparse_hv_matrix(
                    hvb, self.hvb_vocab_size
                )
                hvb_embed = torch.sparse.mm(hvb_sparse, self.hvb_embeds.weight)

                hvf = [fi.hvf for fi in seq]
                hvf_sparse = self.get_sparse_hv_matrix(
                    hvf, self.hvf_vocab_size
                )
                hvf_embed = torch.sparse.mm(hvf_sparse, self.hvf_embeds.weight)

                hva = [fi.hva for fi in seq]
                hva_sparse = self.get_sparse_hv_matrix(
                    hva, self.hva_vocab_size
                )
                hva_embed = torch.sparse.mm(hva_sparse, self.hva_embeds.weight)

            # null antecendent and depth
            nulla = torch.FloatTensor([fi.nulla for fi in seq])
            depth = [fi.depth for fi in seq]
            depth = F.one_hot(torch.LongTensor(depth), self.max_depth).float()

            if self.use_gpu:
                nulla = nulla.to('cuda')
                depth = depth.to('cuda')

            seq_attn_input = torch.cat(
                (catb_embed, hvb_embed, hvf_embed, depth), dim=1
            ) 
            seq_coref_emb = torch.cat(
                (hva_embed, nulla.unsqueeze(dim=1)), dim=1
            )

            per_seq_attn_input.append(seq_attn_input)
            per_seq_coref_emb.append(seq_coref_emb)
    
        return per_seq_attn_input, per_seq_coref_emb


    def get_padded_input_matrix(self, per_sequence_x):
        max_length = max(len(seq) for seq in per_sequence_x)
        width = per_sequence_x[0].size()[1]
        padded_seqs = list()
        for seq_x in per_sequence_x:
            seq_length = seq_x.size()[0]
            padded_seq = torch.Tensor(max_length, width)
            # -1 is used for padding
            padded_seq.fill_(-1)
            padded_seq[:seq_length, :] = seq_x
            padded_seqs.append(padded_seq)
        x = torch.stack(padded_seqs, dim=1)
        if self.use_gpu:
            x = x.to('cuda')
        return x


    def get_attn_mask(self, seq_length):
        # entries marked as True are what we want to mask
        mask = torch.ones(seq_length, seq_length, dtype=bool)
        if self.use_gpu:
            mask = mask.to('cuda')
        return torch.triu(mask, diagonal=1)


    def forward(self, batch_finfo):
        # list of matrices, one matrix for each sequence
        per_seq_attn_input, per_seq_coref_emb = \
            self.get_per_sequence_x(batch_finfo)
        # attn_input_3d and coref_emb_3d are 3D tensors of dimensionality SxNxE
        # S: sequence length
        # N: batch size (number of sequences)
        # E: embedding size
        attn_input_3d = self.get_padded_input_matrix(per_seq_attn_input)
        coref_emb_3d = self.get_padded_input_matrix(per_seq_coref_emb)
        return self.compute(attn_input_3d, coref_emb_3d)


    def compute(self, attn_input, coref_emb, verbose=False):
        # the same matrix is used as query, key, and value. Within the attn
        # layer this will be projected to a separate q, k, and v for each
        # attn head
#        if verbose:
#            eprint('F final word attn input:')
#            for x in attn_input[-1, 0]:
#                eprint(x.item())
#            weights = self.state_dict()['pre_attn_fc.weight'].data.cpu().numpy()
#            eprint('F pre attn fc weights shape:', weights.shape)
#            eprint('F pre attn fc weights numpy:')
#            eprint(weights)
#            eprint('F pre attn fc weights:')
#            # F is column-major order
#            for x in weights.flatten('F'):
#                eprint(x)
#            eprint('F final word pre_attn_fc bias:')
#            bias = self.state_dict()['pre_attn_fc.bias'].data.cpu().numpy()
#            for x in bias:
#                eprint(x)
        qkv = self.pre_attn_fc(attn_input)
        if verbose:
            eprint('F final word\'s qkv:')
            for x in qkv[-1, 0]:
                eprint(x.item())
        if self.use_positional_encoding:
            qkv = self.positional_encoding(qkv)
        # use mask to hide future inputs
        mask = self.get_attn_mask(len(attn_input))
        # second output is attn weights
        #attn_output, _ = self.attn(q, k, v, attn_mask=mask)
        attn_output, _ = self.attn(qkv, qkv, qkv, attn_mask=mask)
        if verbose:
            eprint('F final word\'s attn output:')
            for x in attn_output[-1, 0]:
                eprint(x.item())
        x = torch.cat((attn_output, coref_emb), dim=2)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        result = F.log_softmax(x, dim=2)
        if verbose:
            for i in range(result.shape[0]):
                log_scores = result[i, 0]
                scores = torch.exp(log_scores)
                norm_scores = scores/sum(scores)
                eprint('F ==== output for word {} ===='.format(i))
                for x in norm_scores:
                    eprint(x.item())
        return result

