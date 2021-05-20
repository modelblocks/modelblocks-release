import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerFModel(nn.Module):
    def __init__(self, f_config, catb_vocab_size, hvb_vocab_size,
                 hvf_vocab_size, hva_vocab_size, output_dim):
        super(TransformerFModel, self).__init__()
        self.syn_size = f_config.getint('SynSize')
        self.sem_size = f_config.getint('SemSize')
        self.ant_size = f_config.getint('AntSize')
        self.hidden_dim = f_config.getint('HiddenSize')
        self.dropout_prob = f_config.getfloat('DropoutProb')
        self.ablate_syn = f_config.getboolean('AblateSyn')
        self.ablate_sem = f_config.getboolean('AblateSem')
        # TODO this is int in the old config -- why?
        self.use_gpu = f_config.getboolean('UseGPU')
        # TODO add num_blocks option for multiple transformer blocks
        self.num_heads = f_config.getint('NumHeads')
        self.attn_q_dim = f_config.getint('AttnQDim')
        self.attn_k_dim = f_config.getint('AttnKDim')
        self.attn_v_dim = f_config.getint('AttnVDim')
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
        self.catb_embeds = nn.Embedding(catb_vocab_size, self.syn_size)
        self.hvb_embeds = nn.Embedding(hvb_vocab_size, self.sem_size)
        self.hvf_embeds = nn.Embedding(hvf_vocab_size, self.sem_size)
        self.hva_embeds = nn.Embedding(hva_vocab_size, self.ant_size)

        # the 8 is for (one-hot) depth
        input_dim = 8 + self.syn_size + 2*self.sem_size + self.ant_size
        self.query = nn.Linear(input_dim, self.attn_q_dim)
        self.key = nn.Linear(input_dim, self.attn_k_dim)
        self.value = nn.Linear(input_dim, self.attn_v_dim)

        # TODO define this as a block so you can have multiple attention layers
        self.attn =  nn.MultiheadAttention(
            embed_dim=self.attn_q_dim,
            num_heads=self.num_heads,
            kdim=self.attn_k_dim,
            vdim=self.attn_v_dim
        )

        # TODO make this resnet?
        self.fc1 = nn.Linear(self.attn_v_dim, self.hidden_dim, bias=True)
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
        per_sequence_x = list()
        for seq in batch_finfo:
            # base category embeddings
            if self.ablate_syn:
                catb_embed = torch.zeros(
                    [len(seq), self.syn_size],
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
                    [max_seq_length, batch_size, self.sem_size], dtype=torch.float
                )
                hvf_embed = torch.zeros(
                    [max_seq_length, batch_size, self.sem_size], dtype=torch.float
                )
                hva_embed = torch.zeros(
                    [max_seq_length, batch_size, self.ant_size], dtype=torch.float
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

            hvb_top = torch.FloatTensor([fi.hvb_top for fi in seq]).reshape(-1, 1)
            hvf_top = torch.FloatTensor([fi.hvf_top for fi in seq]).reshape(-1, 1)
            hva_top = torch.FloatTensor([fi.hva_top for fi in seq]).reshape(-1, 1)

            if self.use_gpu:
                hvb_top = hvb_top.to('cuda')
                hvf_top = hvf_top.to('cuda')
                hva_top = hva_top.to('cuda')

            hvb_embed = hvb_embed + hvb_top
            hvf_embed = hvf_embed + hvf_top
            hva_embed = hva_embed + hva_top

            # null antecendent and depth
            nulla = torch.FloatTensor([fi.nulla for fi in seq])
            depth = [fi.depth for fi in seq]
            depth = F.one_hot(torch.LongTensor(depth), self.max_depth).float()

            if self.use_gpu:
                nulla = nulla.to('cuda')
                depth = depth.to('cuda')

            sequence_x = torch.cat(
                (catb_embed, hvb_embed, hvf_embed, 
                 hva_embed, nulla.unsqueeze(dim=1), depth), 1
            ) 
            per_sequence_x.append(sequence_x)
    
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


    def forward(self, batch_finfo):
        # list of matrices, one matrix for each sequence
        per_sequence_x = self.get_per_sequence_x(batch_finfo)
        # 3D tensor of dimensionality SxNxE
        # S: sequence length
        # N: batch size (number of sequences)
        # E: embedding size
        x = self.get_padded_input_matrix(per_sequence_x)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        # use mask to hide future inputs
        mask = self.get_attn_mask(len(x))
        # second output is attn weights
        x, _ = self.attn(q, k, v, attn_mask=mask)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=2)

