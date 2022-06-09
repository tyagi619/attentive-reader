'''
This script defines different layers for the BiDAF
model for question answering.
1- WordEmbedding
2- CharEmbedding
3- RNNEncoder
4- BiDAFAttention
5- BiDAFOutput
'''

import torch
import torch.nn  as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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


class CNN(nn.Module):
    def __init__(self, e_char, e_word, len_chars):
        super(CNN, self).__init__()
        self.kernel = 5

        self.conv1d = nn.Conv1d(in_channels=e_char,
                                out_channels=e_word,
                                kernel=self.kernel,
                                bias=True)
        self.maxPool = nn.MaxPool1d(kernel_size=len_chars-self.kernel+1)

    def forward(self, x):
        x_conv = self.conv1d(x)
        x_conv = F.relu(x_conv)
        x_conv = self.maxPool(x_conv).squeeze(dim=-1)
        return x_conv


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
        x_proj = self.proj(embed)
        x_embed = self.highway(x_proj)
        x_embed = F.dropout(input=x_embed, p=self.drop_prob, 
                           training=self.training, inplace=False)
        return x_embed


class CharEmbedding(nn.Module):
    def __init__(self, char_vectors, e_char, e_word, len_chars, num_highway_layers, drop_prob):
        super(CharEmbedding, self).__init__()
        self.kernel = 5
        self.drop_prob = drop_prob
        
        self.embed = nn.Embedding.from_pretrained(embeddings=char_vectors,
                                                  freeze=False,
                                                  padding_idx=0,
                                                  scale_grad_by_freq=False)
        self.cnn = CNN(e_char, e_word, len_chars)
        self.highway = Highway(num_highway_layers, e_word)

    def forward(self, x):
        embed = self.embed(x)
        x_reshaped = embed.reshape(embed.size(0)*embed.size(1),
                                   embed.size(3), embed.size(2))
        x_conv = self.cnn(x_reshaped)
        x_conv = x_conv.reshape(x.size(0), x.size(1), x_conv.size(1))
        x_embed = self.highway(x_conv)
        x_embed = F.dropout(input=x_embed, p=self.drop_prob,
                          training=self.training, inplace=False)
        return x_embed


class RNNEncoder(nn.Module):
    def __init__(self, e_word, hidden_size, num_layers, drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob

        self.lstm = nn.LSTM(input_size=e_word, hidden_size=hidden_size,
                            num_layers=num_layers, bias=True,
                            batch_first=True, bidirectional=True,
                            dropout=drop_prob)

    def forward(self, x, lengths):
        orig_len = x.size(1)
        x_packed = pack_padded_sequence(input=x, lengths=lengths,
                                        batch_first=True,
                                        enforce_sorted=False)
        x_lstm = self.lstm(x_packed)
        x_out = pad_packed_sequence(sequence=x_lstm, batch_first=True,
                                    total_length=orig_len)
        x_out = F.dropout(input=x, p=self.drop_prob, training=self.training)
        return x_out


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