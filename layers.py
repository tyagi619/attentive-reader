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
    def __init__(self, input_size, hidden_size, num_layers, drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
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
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob

        self.bias = nn.Parameter(torch.zeros(1))
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.kaiming_normal_(weight)

    def forward(self, c, q, c_mask, q_mask):
        # c - (batch_size, c_len, hidden_size)
        # q - (batch_size, q_len, hidden_size)
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)

        s = self.get_similarity_matrix(c, q)
        c_mask = c_mask.view((batch_size, c_len, 1))
        q_mask = q_mask.view((batch_size, 1, q_len))

        s1 = BiDAFAttention.masked_softmax(s, q_mask, dim=2)
        s2 = BiDAFAttention.masked_softmax(s, c_mask, dim=1)

        # torch.matmul is a general function to multiply 2 tensor
        # torch.bmm multiplies 2 3-dimensional matrices only. In
        # the below scenario, we can equivalently use torch.matmul
        # instead of torch.bmm and should get the same results
        a = torch.bmm(s1, q)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)
        x = torch.cat((c, a, c*a, c*b), dim=2)
        return x

    def get_similarity_matrix(self, c, q):
        c_len = c.size(1)
        q_len = q.size(1)

        c = F.dropout(c, p=self.drop_prob, training=self.training)
        q = F.dropout(q, p=self.drop_prob, training=self.training)

        sim_c = torch.matmul(c, self.c_weight).expand(-1, -1, q_len)
        sim_q = torch.matmul(q, self.q_weight).transpose(1, 2).expand(-1, c_len, -1)
        sim_cq = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = sim_c + sim_q + sim_cq + self.bias
        return s
    
    @staticmethod
    def masked_softmax(s, mask, dim=-1):
        mask = mask.type(torch.float32)
        masked_s = mask * s + (1-mask) * -1e30
        probs = F.softmax(masked_s, dim=dim)
        return probs


class BiDAFOutput(nn.Module):
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.drop_prob = drop_prob

        self.rnn = RNNEncoder(input_size=2*hidden_size,
                               hidden_size=hidden_size,
                               num_layers=1, drop_prob=drop_prob)

        self.att_linear_1 = nn.Linear(in_features=8*hidden_size,
                                      out_features=1, bias=True)
        self.mod_linear_1 = nn.Linear(in_features=2*hidden_size,
                                      out_features=1, bias=True)
        
        self.att_linear_2 = nn.Linear(in_features=8*hidden_size,
                                      out_features=1, bias=True)
        self.mod_linear_2 = nn.Linear(in_features=2*hidden_size,
                                      out_features=1, bias=True)

    def forward(self, att, mod, mask):
        p_start = self.att_linear_1(att) + self.mod_linear_1(mod)

        lengths = mask.sum(-1)
        mod_2 = self.rnn(mod, lengths)
        p_end = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        log_p_start = BiDAFOutput.masked_log_softmax(p_start.squeeze(),
                                                     mask, dim=-1)
        log_p_end = BiDAFOutput.masked_log_softmax(p_end.squeeze(), mask,
                                                   dim=-1)

        return log_p_start, log_p_end

    @staticmethod
    def masked_log_softmax(s, mask, dim=-1):
        mask = mask.type(torch.float32)
        masked_s = mask * s + (1-mask) * -1e30
        probs = F.log_softmax(masked_s, dim=dim)
        return probs