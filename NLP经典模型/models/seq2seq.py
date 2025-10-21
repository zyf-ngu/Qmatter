import random

import torch
import torch.nn as nn
import random

# seq2seq框架组件之编码器。这里也会产生输出——中间状态向量。
class Encoder(nn.Module):
    # 输入和RNN系列模型一致，这里增加了从原始输入特征维度到词向量维度的embedding过程
    def __init__(self,input_dim,emb_dim,hid_dim,n_layers,dropout):
        super(Encoder,self).__init__()
        # 隐藏层的维度，即h_t,c_t的维度
        self.hid_dim=hid_dim
        # lstm的层数
        self.n_layers=n_layers
        # input_dim即输入的特征维度，emb_dim即词向量维度，手动设置
        self.embedding=nn.Embedding(input_dim,emb_dim)
        # encoder层真实的输入维度即词向量维度
        self.rnn=nn.LSTM(emb_dim,hid_dim,n_layers,dropout=dropout)
        self.dropout=nn.Dropout(dropout)

    def forward(self,src):
        # src = [seq_len, batch_size]
    #     src = nn.tensor([[2, 2, 2, ..., 2, 2, 2],
    #               [4, 4, 4, ..., 4, 4, 4],
    #               [93, 69, 589, ..., 141, 86, 912],
    #               ...,
    #               [1, 1, 1, ..., 1, 1, 1],
    #               [1, 1, 1, ..., 1, 1, 1],
    #               [1, 1, 1, ..., 1, 1, 1]])
    # torch.Size([33, 128]) 33为句子长度（填充后的）
    # 对输入的数据进行embedding操作
        # embedded = [seq_len, batch size, emb_dim] embed后的输入张量，包括序列长度，batch大小，嵌入后的词向量特征维度。
        embedded=self.dropout(self.embedding(src))
        # outputs = [src_len, batch_size, hid_dim * n_directions]
        # hidden(ht) = [n_layers * n_directions, batch_size, hid_dim]
        # cell(ct) = [n_layers * n_directions, batch_size, hid_dim]
        outputs,(hidden,cell)=self.rnn(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self,output_dim,emb_dim,hid_dim,n_layers,dropout):
        super(Decoder,self).__init__()
        self.output_dim=output_dim
        self.hid_dim=hid_dim
        self.n_layers=n_layers
        self.embedding=nn.Embedding(output_dim,emb_dim)
        #注意这里的emb_dim就是输入特征词向量处理后的维度，
        self.rnn=nn.LSTM(emb_dim,hid_dim,n_layers,dropout=dropout)
        self.fc_out=nn.Linear(hid_dim,output_dim)
        self.dropout=nn.Dropout(dropout)

    # input = [batch_size]
    # hidden = [n_layers * n_directions, batch_size, hid_dim]
    # cell = [n_layers * n_directions, batch_size, hid_dim]
    # n_directions in the decoder will both always be 1, therefore:
    # hidden = [n_layers, batch_size, hid_dim]
    # context = [n_layers, batch_size, hid_dim]
    def forward(self,input,hidden,cell):
        # input = [1, batch size]
        input=input.unsqueeze(0)
        # embedded = [1, batch size, emb dim]
        embedded=self.dropout(self.embedding(input))
        # output = [seq_len, batch_size, hid_dim * n_directions]
        # hidden = [n_layers * n_directions, batch_size, hid_dim]
        # cell = [n_layers * n_directions, batch_size, hid_dim]
        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]
        output,(hidden,cell)=self.rnn(embedded,(hidden,cell))
        # prediction = [batch size, output dim]
        prediction=self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder,device):
        super(Seq2Seq,self).__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.device=device
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    # src = [seq_len, batch_size]
    # trg = [trg_len, batch_size]
    # teacher_forcing_ratio is probability to use teacher forcing
    # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
    def forward(self,src,trg,teacher_forcing_ratio=0.5):
        batch_size=trg.shape[1]
        trg_len=trg.shape[0]
        trg_vocab_size=self.decoder.output_dim
        #
        outputs=torch.zeros(trg_len,batch_size,trg_vocab_size).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden,cell=self.encoder(src)
        #decoder模块的第一个输入是<sos> tokens
        input=trg[0,:]
        for t in range(1,trg_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output,hidden,cell=self.decoder(input,hidden,cell)
            # place predictions in a tensor holding predictions for each token
            outputs[t]=output
            # decide if we are going to use teacher forcing or not
            teacher_force=random.Random()<teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1=output.argmax(1)
            #决定输入是根据预测得到的（预测可能是错误的），还是真实值（起纠偏作用），
            input=trg[t] if teacher_force else top1
        return outputs

