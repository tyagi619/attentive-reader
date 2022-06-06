'''
This script defines different layers for the BiDAF
model for question answering.
1- WordEmbedding
2- CharEmbedding
3- RNNEncoder
4- BiDAFAttention
5- BiDAFOutput
'''

from sklearn.preprocessing import scale
import torch
import torch.nn  as nn
import torch.nn.functional as F


class Highway(nn.Module):
    def __init__(self, num_layers, hidden_size):
        super(Highway, self).__init__()
        self.gate = [nn.Linear(in_features=hidden_size,
                              out_features=hidden_size,
                              bias=True)
                     for _ in range(num_layers)]
        self.transform = [nn.Linear(in_features=hidden_size,
                                   out_features=hidden_size,
                                   bias=True)
                          for _ in range(num_layers)]

    def forward(self, x):
        for gate, transform in zip(self.gate, self.transform):
            g = F.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g*t + (1-g)*t
        return x


class WordEmbedding(nn.Module):
    def __init__(self, word_vectors, num_highway_layers, hidden_size, drop_prob):
        super(WordEmbedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(embeddings=word_vectors,
                                                    freeze=True,
                                                    padding_idx=0,
                                                    scale_grad_by_freq=False)
        self.proj = nn.Linear(in_features=word_vectors.size(1),
                              out_features=hidden_size,
                              bias=False)
        self.highway = Highway(num_highway_layers, hidden_size)

    def forward(self, x):
        embed = self.embed(x)
        h_proj = self.proj(embed)
        h_embed = self.highway(h_proj)
        h_embed = F.dropout(input=h_embed, p=self.drop_prob, 
                           training=self.training, inplace=False)
        return h_embed


class CharEmbedding(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


class RNNEncoder(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


class BiDAFAttention(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


class BiDAFOutput(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass