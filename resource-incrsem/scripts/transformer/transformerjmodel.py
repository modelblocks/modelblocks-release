import torch, math
import torch.nn as nn
import torch.nn.functional as F

from transformerfmodel import PositionalEncoding, eprint

def print_weight(w):
    eprint(round(w, 4))

class TransformerJModel(nn.Module):
    def __init__(self, j_config, cat_anc_vocab_size, hv_anc_vocab_size,
                 hv_filler_vocab_size, cat_lc_vocab_size, hv_lc_vocab_size,
                 output_dim):
        super(TransformerJModel, self).__init__()
        self.syn_dim = j_config.getint('SynDim')
        self.sem_dim = j_config.getint('SemDim')
        self.hidden_dim = j_config.getint('HiddenDim')
        self.dropout_prob = j_config.getfloat('DropoutProb')
        self.ablate_syn = j_config.getboolean('AblateSyn')
        self.ablate_sem = j_config.getboolean('AblateSem')
        self.use_positional_encoding = j_config.getboolean('UsePositionalEncoding')
        # TODO this is int in the old config -- why?
        self.use_gpu = j_config.getboolean('UseGPU')
        # TODO add num_blocks option for multiple transformer blocks
        self.num_heads = j_config.getint('NumHeads')
        self.attn_dim = j_config.getint('AttnDim')
        # TODO decide whether to use this at runtime or just for
        # training
        self.attn_window_size = j_config.getint('AttnWindowSize')
        self.hv_anc_vocab_size = hv_anc_vocab_size
        self.hv_filler_vocab_size = hv_filler_vocab_size
        self.hv_lc_vocab_size = hv_lc_vocab_size
        self.output_dim = output_dim
        self.max_depth = 7

        self.cat_anc_embeds = nn.Embedding(cat_anc_vocab_size, self.syn_dim)
        self.cat_lc_embeds = nn.Embedding(cat_lc_vocab_size, self.syn_dim)
        self.hv_anc_embeds = nn.Embedding(hv_anc_vocab_size, self.sem_dim)
        self.hv_filler_embeds = nn.Embedding(hv_filler_vocab_size, self.sem_dim)
        self.hv_lc_embeds = nn.Embedding(hv_lc_vocab_size, self.sem_dim)

        # attn intput comprises:
        # - depth (max_depth)
        # - ancestor cat embedding (syn_dim)
        # - ancestor hvec (sem_dim)
        # - filler hvec (sem_dim)
        # - left child cat (syn_dim)
        # - left child hvec (sem_dim)
        attn_input_dim = self.max_depth + 2*self.syn_dim + 3*self.sem_dim

        # project the attention input to the right dimensionality
        self.pre_attn_fc = nn.Linear(attn_input_dim, self.attn_dim, bias=True)

        # this layer adds positional encodings to the output of pre_attn_fc
        # its output is used for queries, keys, and values
        self.positional_encoding = PositionalEncoding(self.attn_dim)

        # TODO define this as a block so you can have multiple attention layers
        self.attn =  nn.MultiheadAttention(
            embed_dim=self.attn_dim,
            num_heads=self.num_heads,
            bias=True
        )

        # TODO make this resnet?
        self.fc1 = nn.Linear(self.attn_dim, self.hidden_dim, bias=True)
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

    
    def get_per_sequence_x(self, batch_jinfo, verbose=False):
        per_sequence_x = list()
        for i, seq in enumerate(batch_jinfo):
            if self.ablate_syn:
                cat_anc_embed = torch.zeros(
                    [len(seq), self.syn_dim],
                    dtype=torch.float
                )
                cat_lc_embed = torch.zeros(
                    [len(seq), self.syn_dim],
                    dtype=torch.float
                )
            else:
                cat_anc = [ji.cat_anc for ji in seq]
                cat_lc = [ji.cat_lc for ji in seq]
                cat_anc = torch.LongTensor(cat_anc)
                cat_lc = torch.LongTensor(cat_lc)
                if self.use_gpu:
                    cat_anc = cat_anc.to('cuda')
                    cat_lc = cat_lc.to('cuda')
                cat_anc_embed = self.cat_anc_embeds(cat_anc)
                cat_lc_embed = self.cat_lc_embeds(cat_lc)


            # ancestor, filler, left child hvec embeddings
            if self.ablate_sem:
                hv_anc_embed = torch.zeros(
                    [len(seq), self.sem_dim], dtype=torch.float
                )
                hv_filler_embed = torch.zeros(
                    [len(seq), self.sem_dim], dtype=torch.float
                )
                hv_lc_embed = torch.zeros(
                    [len(seq), self.ant_dim], dtype=torch.float
                )

            else:
                hv_anc = [ji.hv_anc for ji in seq]
                hv_anc_sparse = self.get_sparse_hv_matrix(
                    hv_anc, self.hv_anc_vocab_size
                )
                hv_anc_embed = torch.sparse.mm(
                    hv_anc_sparse, self.hv_anc_embeds.weight
                )

                hv_filler = [ji.hv_filler for ji in seq]
                hv_filler_sparse = self.get_sparse_hv_matrix(
                    hv_filler, self.hv_filler_vocab_size
                )
                hv_filler_embed = torch.sparse.mm(
                    hv_filler_sparse, self.hv_filler_embeds.weight
                )

                hv_lc = [ji.hv_lc for ji in seq]
                hv_lc_sparse = self.get_sparse_hv_matrix(
                    hv_lc, self.hv_lc_vocab_size
                )
                hv_lc_embed = torch.sparse.mm(
                    hv_lc_sparse, self.hv_lc_embeds.weight
                )


            depth = [ji.depth for ji in seq]
            depth = F.one_hot(torch.LongTensor(depth), self.max_depth).float()

            if self.use_gpu:
                depth = depth.to('cuda')

            if verbose and i == 0:
                eprint('J ======== first sequence\'s inputs ======== ')
                for j, emb in enumerate(cat_anc_embed):
                    eprint('\nJ ==== word {} ==== '.format(j))
                    eprint('\nJ cat anc emb:')
                    for x in emb:
                        eprint_weight(x.item())
                    eprint('\nJ hv anc emb:')
                    for x in hv_anc_embed[j]:
                        eprint_weight(x.item())
                    eprint('\nJ hv filler emb:')
                    for x in hv_filler_embed[j]:
                        eprint_weight(x.item())
                    eprint('\nJ cat lc emb:')
                    for x in cat_lc_embed[j]:
                        eprint_weight(x.item())
                    eprint('\nJ hv lc emb:')
                    for x in hv_lc_embed[j]:
                        eprint_weight(x.item())
                        

            seq_x = torch.cat(
                (cat_anc_embed, hv_anc_embed, hv_filler_embed, 
                 cat_lc_embed, hv_lc_embed, depth),
                dim=1
            ) 
            per_sequence_x.append(seq_x)
    
        return per_sequence_x


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


    def forward(self, batch_jinfo, verbose=False):
        # list of matrices, one matrix for each sequence
        per_sequence_x = self.get_per_sequence_x(batch_jinfo, verbose)

        # attn_input is a 3D tensor of dimensionality SxNxE
        # S: sequence length
        # N: batch size (number of sequences)
        # E: embedding size
        attn_input = self.get_padded_input_matrix(per_sequence_x)
        return self.compute(attn_input, verbose)


    def compute(self, attn_input, verbose=False):
        # the same matrix is used as query, key, and value. Within the attn
        # layer this will be projected to a separate q, k, and v for each
        # attn head
        qkv = self.pre_attn_fc(attn_input)
        if self.use_positional_encoding:
            qkv = self.positional_encoding(qkv)
        # use mask to hide future inputs
        mask = self.get_attn_mask(len(attn_input))
        # second output is attn weights
        #attn_output, _ = self.attn(q, k, v, attn_mask=mask)
        attn_output, _ = self.attn(qkv, qkv, qkv, attn_mask=mask)
        x = self.fc1(attn_output)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        result = F.log_softmax(x, dim=2)
        if verbose:
            # note: this assumes that there is only one sequence in the batch
            for i in range(result.shape[0]):
                eprint('\nJ ==== word {} ===='.format(i))
                attn_input_i = attn_input[i, 0]
                eprint('J attn input')
                for x in attn_input_i:
                    print_weight(x.item())
                qkv_i = qkv[i, 0]
                eprint('\nJ qkv')
                for x in qkv_i:
                    print_weight(x.item())
                attn_output_i = attn_output[i, 0]
                eprint('\nJ attn output')
                for x in attn_output_i:
                    print_weight(x.item())
                log_scores = result[i, 0]
                scores = torch.exp(log_scores)
                norm_scores = scores/sum(scores)
                eprint('\nJ result')
                for x in norm_scores:
                    print_weight(x.item())
                eprint()
        return result

