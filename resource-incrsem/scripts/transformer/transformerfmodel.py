import torch, math, sys
import torch.nn as nn
import torch.nn.functional as F


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


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
        self.max_depth = 6

        # TODO should there just be one big embedding? inputs would be
        # multi-hot; not sure if that's easy
        self.catb_embeds = nn.Embedding(catb_vocab_size, self.syn_dim)
        self.hvb_embeds = nn.Embedding(hvb_vocab_size, self.sem_dim)
        self.hvf_embeds = nn.Embedding(hvf_vocab_size, self.sem_dim)
        self.hva_embeds = nn.Embedding(hva_vocab_size, self.ant_dim)

        # attn intput is (one-hot) depth, base syn cat embedding, and
        # base hvec embedding
        attn_input_dim = 7 + self.syn_dim + self.sem_dim

        # project the attention input to the right dimensionality
        self.pre_attn_fc = nn.Linear(attn_input_dim, self.attn_dim, bias=True)

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
        # with the filler hVec embedding, antecedent hVec embedding, and the
        # null antecedent indicator (hence the +1)
        self.fc1 = nn.Linear(
            self.attn_dim+self.sem_dim+self.ant_dim+1, 
            self.hidden_dim,
            bias=True
        )
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

    
#    def get_per_sequence_x(self, batch_finfo):
#        #per_sequence_x = list()
#        per_seq_attn_input = list()
#        per_seq_coref_emb = list()
#        for seq in batch_finfo:
#            # base category embeddings
#            if self.ablate_syn:
#                catb_embed = torch.zeros(
#                    [len(seq), self.syn_dim],
#                    dtype=torch.float
#                )
#            else:
#                catb = [fi.catb for fi in seq]
#                catb = torch.LongTensor(catb)
#                if self.use_gpu:
#                    catb = catb.to('cuda')
#                catb_embed = self.catb_embeds(catb)
#
#
#            # base, filler, antecedent hvec embeddings
#            if self.ablate_sem:
#                hvb_embed = torch.zeros(
#                    [max_seq_length, batch_size, self.sem_dim], dtype=torch.float
#                )
#                hvf_embed = torch.zeros(
#                    [max_seq_length, batch_size, self.sem_dim], dtype=torch.float
#                )
#                hva_embed = torch.zeros(
#                    [max_seq_length, batch_size, self.ant_dim], dtype=torch.float
#                )
#
#            else:
#                hvb = [fi.hvb for fi in seq]
#                hvb_sparse = self.get_sparse_hv_matrix(
#                    hvb, self.hvb_vocab_size
#                )
#                hvb_embed = torch.sparse.mm(hvb_sparse, self.hvb_embeds.weight)
#
#                hvf = [fi.hvf for fi in seq]
#                hvf_sparse = self.get_sparse_hv_matrix(
#                    hvf, self.hvf_vocab_size
#                )
#                hvf_embed = torch.sparse.mm(hvf_sparse, self.hvf_embeds.weight)
#
#                hva = [fi.hva for fi in seq]
#                hva_sparse = self.get_sparse_hv_matrix(
#                    hva, self.hva_vocab_size
#                )
#                hva_embed = torch.sparse.mm(hva_sparse, self.hva_embeds.weight)
#
#            hvb_top = torch.FloatTensor([fi.hvb_top for fi in seq]).reshape(-1, 1)
#            hvf_top = torch.FloatTensor([fi.hvf_top for fi in seq]).reshape(-1, 1)
#            hva_top = torch.FloatTensor([fi.hva_top for fi in seq]).reshape(-1, 1)
#
#            if self.use_gpu:
#                hvb_top = hvb_top.to('cuda')
#                hvf_top = hvf_top.to('cuda')
#                hva_top = hva_top.to('cuda')
#
#            hvb_embed = hvb_embed + hvb_top
#            hvf_embed = hvf_embed + hvf_top
#            hva_embed = hva_embed + hva_top
#
#            # null antecendent and depth
#            nulla = torch.FloatTensor([fi.nulla for fi in seq])
#            depth = [fi.depth for fi in seq]
#            depth = F.one_hot(torch.LongTensor(depth), self.max_depth).float()
#
#            if self.use_gpu:
#                nulla = nulla.to('cuda')
#                depth = depth.to('cuda')
#
#            seq_attn_input = torch.cat(
#                (catb_embed, hvb_embed, hvf_embed, depth), dim=1
#            ) 
#            seq_coref_emb = torch.cat(
#                (hva_embed, nulla.unsqueeze(dim=1)), dim=1
#            )
#
#            per_seq_attn_input.append(seq_attn_input)
#            per_seq_coref_emb.append(seq_coref_emb)
#    
#        return per_seq_attn_input, per_seq_coref_emb


    def get_input_matrices(self, batch_finfo):
        '''
        Returns two matrices:
        - The inputs that get passed into the attention layer
        - The inputs that get concatenated with the attention output and
            fed into the hidden layer
        '''
        per_stack_pre_attn_input = list()
        per_stack_post_attn_input = list()

        for finfo in batch_finfo:
            stack = finfo.stack
            # we want the beginning of the list to be the deepest fragment
            stack.reverse()

            # pre-attention inputs
            if self.ablate_syn:
                catb_embed = torch.zeros(
                    [len(stack), self.syn_dim],
                    dtype=torch.float
                )
            else:
                catb = [df.catbase for df in stack]
                catb = torch.LongTensor(catb)
                if self.use_gpu:
                    catb = catb.to('cuda')
                catb_embed = self.catb_embeds(catb)

            if self.ablate_sem:
                hvb_embed = torch.zeros(
                    [len(stack), self.sem_dim],
                    dtype=torch.float
                )

            else:
                hvb = [df.hvbase for df in stack]
                hvb_sparse = self.get_sparse_hv_matrix(
                    hvb, self.hvb_vocab_size
                )
                hvb_embed = torch.sparse.mm(hvb_sparse, self.hvb_embeds.weight)
            hvb_top = torch.FloatTensor([df.hvbase_top for df in stack]).reshape(-1, 1)
            if self.use_gpu:
                hvb_top = hvb_top.to('cuda')
            hvb_embed = hvb_embed + hvb_top

            d = [df.depth for df in stack]
            d = F.one_hot(torch.LongTensor(d), self.max_depth+1).float()

            if self.use_gpu:
                d = d.to('cuda')

            # shape: len(stack) x (syn_dim + sem_dim + max_depth+1)
            stack_pre_attn_input = torch.cat(
                (catb_embed, hvb_embed, d), dim=1
            )

            # pad the pre-attn input so that every stack's first dimension is
            # max_depth + 1. The +1 is because the stack includes a fragment at
            # depth 0
            padding = torch.zeros(
                [self.max_depth-len(stack)+1, stack_pre_attn_input.shape[1]],
                dtype=torch.float
            )
            if self.use_gpu:
                padding = padding.to('cuda')
            stack_pre_attn_input = torch.cat((stack_pre_attn_input, padding))
            per_stack_pre_attn_input.append(stack_pre_attn_input)

            # post-attention inputs. these come from the deepest derivation
            # fragment, like in the MLP F model
            if self.ablate_sem:
                hvf_embed = torch.zeros(
                    [self.sem_dim], dtype=torch.float
                )
                hva_embed = torch.zeros(
                    [self.ant_dim], dtype=torch.float
                )

            else:
                hvf_sparse = self.get_sparse_hv_matrix(
                    [finfo.hvf], self.hvf_vocab_size
                )
                hvf_embed = torch.sparse.mm(hvf_sparse, self.hvf_embeds.weight)
                # reshape from (1, sem_dim) to (sem_dim)
                hvf_embed = hvf_embed.reshape(self.sem_dim)

                hva_sparse = self.get_sparse_hv_matrix(
                    [finfo.hva], self.hva_vocab_size
                )
                hva_embed = torch.sparse.mm(hva_sparse, self.hva_embeds.weight)
                # reshape from (1, ant_dim) to (ant_dim)
                hva_embed = hva_embed.reshape(self.ant_dim)

            nulla = torch.FloatTensor([finfo.nulla])
            if self.use_gpu:
                nulla = nulla.to('cuda')
            # 1D Tensor of length sem_dim + ant_dim + 1
            stack_post_attn_input = torch.cat((hvf_embed, hva_embed, nulla))
            per_stack_post_attn_input.append(stack_post_attn_input)

        
        pre_attn_input = torch.stack(per_stack_pre_attn_input, dim=1)
        assert pre_attn_input.shape == torch.Size(
            [self.max_depth+1, len(batch_finfo), self.syn_dim+self.sem_dim+self.max_depth+1]
        )
        post_attn_input = torch.stack(per_stack_post_attn_input, dim=0)
        assert post_attn_input.shape == torch.Size(
            [len(batch_finfo), self.syn_dim+self.ant_dim+1]
        )
        return pre_attn_input, post_attn_input
            



#        else:
#            hvb = [fi.hvb for fi in seq]
#            hvb_sparse = self.get_sparse_hv_matrix(
#                hvb, self.hvb_vocab_size
#            )
#            hvb_embed = torch.sparse.mm(hvb_sparse, self.hvb_embeds.weight)
#
#            hvf = [fi.hvf for fi in seq]
#            hvf_sparse = self.get_sparse_hv_matrix(
#                hvf, self.hvf_vocab_size
#            )
#            hvf_embed = torch.sparse.mm(hvf_sparse, self.hvf_embeds.weight)
#
#            hva = [fi.hva for fi in seq]
#            hva_sparse = self.get_sparse_hv_matrix(
#                hva, self.hva_vocab_size
#            )
#            hva_embed = torch.sparse.mm(hva_sparse, self.hva_embeds.weight)

#            depth = [fi.depth for fi in seq]
#            depth = F.one_hot(torch.LongTensor(depth), self.max_depth).float()


        

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
        # batch_finfo is a list of FInfo objects
        attn_input, post_attn_input = self.get_input_matrices(batch_finfo)
        #return self.compute(attn_input, post_attn_input)
        # TODO make verbosity configurable
        return self.compute(attn_input, post_attn_input, verbose=True)


    def compute(self, attn_input, post_attn_input, verbose=False):
        # the same matrix is used as query, key, and value. Within the attn
        # layer this will be projected to a separate q, k, and v for each
        # attn head
        qkv = self.pre_attn_fc(attn_input)

        # second output is attn weights
        attn_output, _ = self.attn(qkv, qkv, qkv)

        # the first row of attn_output is the result for the deepest
        # derivation fragment. That's the only one we care about
        attn_output = attn_output[0]
        x = torch.cat((attn_output, post_attn_input), dim=1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        result = F.log_softmax(x, dim=1)
#        if verbose:
#            for i in range(result.shape[0]):
#                log_scores = result[i, 0]
#                scores = torch.exp(log_scores)
#                norm_scores = scores/sum(scores)
#                print('F ==== output for word {} ===='.format(i))
#                for x in norm_scores:
#                    print(x.item())
        return result

