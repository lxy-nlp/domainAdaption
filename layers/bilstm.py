import torch
import torch.nn as nn
import torch.nn.functional as F


# 目前需要将ｌｓｔｍ－－ｇｒｕ
# 特征提取器
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_classes, pretrained_embeddings, n_layers=1, bidirectional=True):
        super(BiLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.n_layers = n_layers
        self.bidirecional = bidirectional
        self.embedded = nn.Embedding(vocab_size, embedding_size)
        # 使用预训练的词向量
        self.embedded.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.tanh = nn.Tanh()

        self.lstm_net = nn.LSTM(embedding_size, hidden_size, bidirectional=bidirectional, batch_first=True)

        if bidirectional:
            self.attention_layer = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
                nn.ReLU(inplace=True)
            )
        else:
            self.attention_layer = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(inplace=True)
            )

    def attention_net_with_w(self, rnn_out):
        m = F.tanh(rnn_out)
        atten_w = self.attention_layer(m)
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))
        soft_w = F.softmax(atten_context, dim=-1)
        context = torch.bmm(soft_w, rnn_out)
        result = context.squeeze(1)
        result = torch.sum(result, 1)
        return result, soft_w

    def forward(self, x):
        word_ebd = self.embedded(x)  # x->input
        # 　初始化隐状态
        hidden_0 = torch.randn(2, x.size(0), self.hidden_size)  # 初始化隐状态　lstm和gru是共有的
        cell_0 = torch.randn(2, x.size(0), self.hidden_size)  #初始化细胞状态
        # word_ebd = word_ebd.permute(1, 0, 2)
        init_hidden = (hidden_0, cell_0)
        # gru_out shape batch_size seq_length hidden_size
        # hidden_state shape num_layers*num_direction batch_size hidden_size
        lstm_out, (hidden_state,cell_state)  = self.lstm_net(word_ebd, init_hidden)
        atten_out, soft_w = self.attention_net_with_w(lstm_out)
        return atten_out, soft_w  # 返回注意力权重和隐状态


    '''

    def attention_net(self, lstm_output):
        #print(lstm_output.size()) = (squence_length, batch_size, hidden_size*layer_size)

        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size*self.layer_size])
        #print(output_reshape.size()) = (squence_length * batch_size, hidden_size*layer_size)

        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        #print(attn_tanh.size()) = (squence_length * batch_size, attention_size)

        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        #print(attn_hidden_layer.size()) = (squence_length * batch_size, 1)

        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, self.sequence_length])
        #print(exps.size()) = (batch_size, squence_length)

        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        #print(alphas.size()) = (batch_size, squence_length)

        alphas_reshape = torch.Tensor.reshape(alphas, [-1, self.sequence_length, 1])
        #print(alphas_reshape.size()) = (batch_size, squence_length, 1)

        state = lstm_output.permute(1, 0, 2)
        #print(state.size()) = (batch_size, squence_length, hidden_size*layer_size)

        attn_output = torch.sum(state * alphas_reshape, 1)
        #print(attn_output.size()) = (batch_size, hidden_size*layer_size)

        return attn_output

    '''