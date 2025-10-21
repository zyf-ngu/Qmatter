import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    def __init__(self, input_size, enc_hiddenc_dim, num_layers,dec_hid_dim):
        super().__init__()
        #和传统seq2seq一样定义输入特征维度，隐藏层特征维度，隐藏层数，输出层特征维度
        self.input_size = input_size
        self.enc_hiddenc_dim = enc_hiddenc_dim
        self.num_layers = num_layers
        self.dec_hid_dim = dec_hid_dim
        #选择encoder模块的模型，这里选用gru,并传入参数输入特征维度，隐藏层特征维度，隐藏层数，
        self.gru = nn.GRU(input_size = input_size,
                          hiddenc_size = enc_hiddenc_dim,
                          num_layers = num_layers,
                          bidirectional=True)
        #根据利用全连接层将隐藏层特征维度转换为输出层特征维度
        self.fc = nn.Linear(enc_hiddenc_dim * 2, dec_hid_dim)  # 拼接后

    def forward(self, init_input, h0):
        #前向传播得到encoder模块的输出层和隐藏层（输出层是隐藏层的全连接变换，也可以直接用隐藏层作为输出层）
        enc_output, hidden = self.gru(init_input, h0)
        # 将正向最后时刻的输出与反向最后时刻的输出进行拼接，得到的维度应该是[batch,enc_hiddenc_dim*2]
        # hidden[-1,:,:] 反向最后时刻的输出, hidden[-2,:,:] 正向最后时刻的输出
        h_m = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1) # 在第一个维度进行拼接
        #这里的s0就是decoder模块的第0个隐藏层特征向量
        s0 = self.fc(h_m)
        #enc_output的维度是[seq_len,batch,enc_hiddenc_dim*2]，s0的维度是[batch,dec_hiddenc_dim]。
        return enc_output,s0


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        #第一步求query和key向量，query向量根据decoder模块中上一时刻的隐藏层st-1求得，key向量由encoder模块中所有时刻的隐藏层h1: x分别求得，value向量这里不经过变换
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim, bias=False)  # 输出的维度是任意的
        #用来计算相关性系数
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)  # 将输出维度置为1

    def forward(self, s, enc_output):
        # s = [batch_size, dec_hiddenc_dim]
        # enc_output = [seq_len, batch_size, enc_hid_dim * 2]

        batch_size = enc_output.shape[1]
        seq_len = enc_output.shape[0]

        # repeat decoder hidden state seq_len times
        # s = [seq_len, batch_size, dec_hid_dim]
        ##维度变化： [batch_size, dec_hid_dim]=>[1, batch_size, dec_hid_dim]=>[seq_len, batch_size, dec_hid_dim]
        s = s.unsqueeze(0).repeat(seq_len, 1, 1)

        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim=2)))
        #计算相关性系数，维度变化： [seq_len, batch_size, dec_hid_dim]=>[seq_len，batch_size, 1] => [seq_len, batch_size]
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=0).transpose(0, 1)  # [batch_size, seq_len]


class Decoder(nn.Module):
    #初始化基本参数，输出层特征维度，encoder模块隐藏层特征维度，deco模块隐藏层特征维度，attention层
    def __init__(self, output_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        # self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + 1, dec_hid_dim)
        # self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + 1, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_input, s, enc_output):

        # dec_input = [batch_size]
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim * 2]

        dec_input = dec_input.unsqueeze(1) # dec_input = [batch_size, 1]

        # embedded = self.dropout(self.embedding(dec_input)).transpose(0, 1) # embedded = [1, batch_size, emb_dim]
        dropout_dec_input = self.dropout(dec_input).transpose(0, 1).unsqueeze(2) #  [1, batch_size]=>[1,batch,1]
       #与传统decoder模块的核心变化，这里需要计算隐藏层中上一个时刻的特征向量与encoder模块的所有时刻隐藏层特征向量的相关性
        # a = [batch_size, 1, src_len]
        a = self.attention(s, enc_output).unsqueeze(1)
        print("a.shape:",a.shape)

        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        enc_output = enc_output.transpose(0, 1)
        #然后根据相关性系数计算得到“attention”之后的中间特征向量c，并作为decoder模块的输入
        # c = [1, batch_size, enc_hid_dim * 2]
        c = torch.bmm(a, enc_output).transpose(0, 1)

        # rnn_input = [1, batch_size, (enc_hid_dim * 2) + emb_dim]
        # rnn_input = torch.cat((embedded, c), dim = 2)
        rnn_input = torch.cat((dropout_dec_input, c), dim = 2) # rnn_input = [1, batch_size, (enc_hid_dim * 2) + 1]

        # dec_output = [src_len(=1), batch_size, dec_hid_dim]
        # dec_hidden = [n_layers * num_directions(=1), batch_size, dec_hid_dim]
        dec_output, dec_hidden = self.rnn(rnn_input, s.unsqueeze(0)) #  s.unsqueeze(0):[1,batch_size, dec_hid_dim]

        dec_output = dec_output.squeeze(0) # dec_output:[batch_size, dec_hid_dim]
        c = c.squeeze(0)  # c:[batch_size, enc_hid_dim * 2]
        # dec_input:[batch_size, 1]

        # pred = [batch_size, output_dim]
        pred = self.fc_out(torch.cat((dec_output, c, dec_input), dim = 1))

        return torch.tanh(pred), dec_hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]
        # teacher_forcing_ratio是选择几率

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # 存储所有时刻的输出
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)  # 存储decoder的所有输出

        enc_output, s = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        dec_input = trg[0, :]  # target的第一列，即全是<SOS>
        #传统 Seq2Seq 是直接将句子中每个词连续不断输入 Decoder 进行训练，
        # 而引入 Attention 机制之后，需要能够人为控制一个词一个词进行输入（因为输入每个词到 Decoder，需要再做一些运算），
        # 所以在代码中会看到使用了 for 循环，循环 trg_len-1 次（开头的 手动输入，所以循环少一次）。

        for t in range(1, trg_len):
            dec_output, s = self.decoder(dec_input, s, enc_output)

            # 存储每个时刻的输出
            outputs[t] = dec_output

            # 用TeacherForce机制
            teacher_force = random.random() < teacher_forcing_ratio

            # 获取预测值
            top1 = dec_output.argmax(1)

            dec_input = trg[t] if teacher_force else top1

        return outputs
