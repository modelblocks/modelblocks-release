import torch, math, sys
import torch.nn as nn
import torch.nn.functional as F

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def print_tensor(t, maxlen=10):
    eprint('Printing first {} items...'.format(maxlen))
    for w in t[:maxlen]:
        eprint(round(w.item(), 8))
    eprint()


class TransformerLayer(nn.Module):
    def __init__(self, attn_dim, num_heads, ff_dim, dropout_prob, use_gpu):
        super(TransformerLayer, self).__init__()
        self.attn =  nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            bias=True
        )
        # TODO make this resnet?
        self.feedforward = nn.Linear(attn_dim, ff_dim, bias=True)
        self.dropout_prob = dropout_prob
        self.dropout = nn.Dropout(self.dropout_prob)
        self.relu = F.relu
        self.use_gpu = use_gpu


    def get_attn_mask(self, seq_length):
        # entries marked as True are what we want to mask
        mask = torch.ones(seq_length, seq_length, dtype=bool)
        if self.use_gpu:
            mask = mask.to('cuda')
        return torch.triu(mask, diagonal=1)


    def forward(self, x, verbose=False):
        # use mask to hide future inputs
        mask = self.get_attn_mask(len(x))

        # second output is attn weights
        # the input matrix is used for queries, keys, and values
        x, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.feedforward(x)
        x = self.dropout(x)
        if verbose:
            for i in range(x.shape[0]):
                eprint('J word {} pre-relu feedforward output'.format(i))
                print_tensor(x[i, 0])

        if verbose:
            ff_out = self.relu(x)
            for i in range(ff_out.shape[0]):
                eprint('J word {} feedforward output'.format(i))
                print_tensor(ff_out[i, 0])
            
        return self.relu(x)


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
        self.num_transformer_layers = f_config.getint('NumTransformerLayers')
        self.num_heads = f_config.getint('NumHeads')
        self.attn_dim = f_config.getint('AttnDim')
        # TODO decide whether to use this at runtime or just for
        # training
        self.attn_window_size = f_config.getint('AttnWindowSize')
        self.hvb_vocab_size = hvb_vocab_size
        self.hvf_vocab_size = hvf_vocab_size
        self.hva_vocab_size = hva_vocab_size
        self.output_dim = output_dim
        self.max_depth = 7

        self.catb_embeds = nn.Embedding(catb_vocab_size, self.syn_dim)
        self.hvb_embeds = nn.Embedding(hvb_vocab_size, self.sem_dim)
        self.hvf_embeds = nn.Embedding(hvf_vocab_size, self.sem_dim)
        self.hva_embeds = nn.Embedding(hva_vocab_size, self.ant_dim)

        # attn intput is (one-hot) depth, base syn cat embedding,
        # base hvec embedding, and filler hvec embedding
        attn_input_dim = 7 + self.syn_dim + 2*self.sem_dim

        # project the attention input to the right dimensionality
        self.pre_attn_fc = nn.Linear(attn_input_dim, self.attn_dim, bias=True)

        # this layer adds positional encodings to the output of pre_attn_fc
        # its output is used for queries, keys, and values
        self.positional_encoding = PositionalEncoding(self.attn_dim)

        # TODO hidden_dim and attn_dim need to be the same I think
        # TODO make this resnet?
        self.transformer_layers = nn.ModuleList()
        for i in range(self.num_transformer_layers):
            self.transformer_layers.append(
                TransformerLayer(
                    attn_dim=self.attn_dim,
                    num_heads=self.num_heads,
                    ff_dim=self.hidden_dim,
                    dropout_prob=self.dropout_prob,
                    use_gpu=self.use_gpu
                )
            )

        # NOTE: in the earlier implemenetation that only allowed for a single
        # attnetion layer, the antecedent stuff was added before the first
        # feedforward:
        #self.fc1 = nn.Linear(self.attn_dim+self.ant_dim+1, self.hidden_dim, bias=True)
        # the input to fc2 is the output of the transformer layers concatenated
        # with the antecedent hVec embedding and the null antecedent indicator
        # (hence the +1)
        self.output_fc = nn.Linear(self.hidden_dim+self.ant_dim+1, self.output_dim, bias=True)

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
            if self.ablate_sem:
                hvb_embed = torch.zeros(
                    [len(seq), self.sem_dim], dtype=torch.float
                )
                hvf_embed = torch.zeros(
                    [len(seq), self.sem_dim], dtype=torch.float
                )
                hva_embed = torch.zeros(
                    [len(seq), self.ant_dim], dtype=torch.float
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


    def forward(self, batch_finfo, verbose=False):
        # list of matrices, one matrix for each sequence
        per_seq_attn_input, per_seq_coref_emb = \
            self.get_per_sequence_x(batch_finfo)
        # attn_input_3d and coref_emb_3d are 3D tensors of dimensionality SxNxE
        # S: sequence length
        # N: batch size (number of sequences)
        # E: embedding size
        attn_input_3d = self.get_padded_input_matrix(per_seq_attn_input)
        coref_emb_3d = self.get_padded_input_matrix(per_seq_coref_emb)
        return self.compute(attn_input_3d, coref_emb_3d, verbose)


    def compute(self, attn_input, coref_emb, verbose=False):
        if verbose:
            for i in range(attn_input.shape[0]):
                eprint('F word {} pre-fwpm attn input'.format(i))
                print_tensor(attn_input[i, 0])
        x = self.pre_attn_fc(attn_input)
        if self.use_positional_encoding:
            x = self.positional_encoding(x)
        if verbose:
            for i in range(x.shape[0]):
                eprint('F word {} proj'.format(i))
                print_tensor(x[i, 0])

        for i, tr_layer in enumerate(self.transformer_layers):
            if verbose:
                eprint('\n ==== transformer layer {} ===='.format(i))
                for i in range(x.shape[0]):
                    eprint('F word {} curr attn inputs'.format(i))
                    print_tensor(x[i, 0])
            x = tr_layer(x, verbose)

        x = torch.cat((x, coref_emb), dim=2)
        ff2_output = self.output_fc(x)
        result = F.log_softmax(ff2_output, dim=2)

        if verbose:
            # note: this assumes that there is only one sequence in the batch
            for i in range(result.shape[0]):
                eprint('\nF ==== word {} ===='.format(i))
                attn_input_i = attn_input[i, 0]
                eprint('F attn input')
                print_tensor(attn_input_i)

                second_ff_input_i = x[i, 0]
                eprint('F second ff input')
                print_tensor(second_ff_input_i, 1000)

                second_ff_output_i = ff2_output[i, 0]
                eprint('F second ff output')
                print_tensor(second_ff_output_i)

                log_softmax = result[i, 0]
                scores = torch.exp(log_softmax)
                eprint('\nF scores')
                print_tensor(scores)
                eprint()
        return result

