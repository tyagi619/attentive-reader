import torch
import torch.nn as nn

import layers


class BiDAF(nn.Module):
    def __init__(self, word_vectors, hidden_size,
                 n_highway_layers=2, drop_prob=0.0):
        super(BiDAF, self).__init__()

        self.w_emb = layers.WordEmbedding(word_vectors=word_vectors,
                                        n_highway_layers=n_highway_layers,
                                        hidden_size=hidden_size,
                                        drop_prob=drop_prob)
        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)
        self.att = layers.BiDAFAttention(hidden_size=2*hidden_size,
                                         drop_prob=drop_prob)
        self.mod = layers.RNNEncoder(input_size=8*hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)
        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.w_emb(cw_idxs)
        q_emb = self.w_emb(qw_idxs)

        # lengths must be on cpu for pack_padded_sequence if
        # provided as tensor (official documentation pytorch)
        c_enc = self.enc(c_emb, c_len.cpu())
        q_enc = self.enc(q_emb, q_len.cpu())

        att = self.att(c_enc, q_enc, c_mask, q_mask)
        mod = self.mod(att, c_len.cpu())
        out = self.out(att, mod, c_mask)
        return out
