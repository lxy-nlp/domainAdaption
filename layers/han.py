import torch
import torch.nn.functional as F
from torch import nn


class SelfAttention(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(SelfAttention, self).__init__()
        self.W = nn.Linear(input_size, hidden_size, True)
        self.u = nn.Linear(hidden_size, 1)

    def forward(self, x):
        u = torch.tanh(self.W(x))
        a = F.softmax(self.u(u), dim=1)
        x = a.mul(x).sum(1)
        return x


class HAN(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, num_classes, max_word_length, max_seq_length,pretrained_embeddings, n_layers=1,
                 bidirectional=True):
        super(HAN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.max_word_length = max_word_length
        self.max_seq_length = max_seq_length
        self.n_layers = n_layers
        self.bidirecional = bidirectional
        self.embedded = nn.Embedding(vocab_size, embedding_size)
        # 使用预训练的词向量
        self.embedded.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.tanh = nn.Tanh()
        self.gru_word_net = nn.GRU(embedding_size, hidden_size, bidirectional=bidirectional, batch_first=True)
        self.gru_seq_net = nn.GRU(hidden_size, hidden_size, bidirectional=bidirectional, batch_first=True)
        if bidirectional:
            self.attention_layer_word = SelfAttention(hidden_size * 2, hidden_size)
            self.attention_layer_seq = SelfAttention(hidden_size*2,hidden_size)
        else:
            self.attention_layer_word = SelfAttention(hidden_size,hidden_size)
            self.attention_layer_seq = SelfAttention(hidden_size, hidden_size)

    def forward(self, x):
        hidden_0 = torch.randn(2, x.size(0), self.hidden_size)  # 初始化隐状态　lstm和gru是共有的
        # word_ebd = word_ebd.permute(1, 0, 2)
        init_hidden = hidden_0
        x = x.view(-1, self.max_word_length, self.hidden_size)
        word_emb = self.embedded(x)
        gru_word, _ = self.gru_word_net(word_emb,init_hidden)
        gru_word = self.attention_layer_word(gru_word)
        gru_word = gru_word.view(-1, self.max_seq_length,self.hidden_size)
        gru_seq, _ = self.gru_seq_net(gru_word,init_hidden)
        gru_seq = self.attention_layer_seq(gru_seq)
        return gru_seq


