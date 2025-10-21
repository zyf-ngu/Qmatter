import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=5000):
        super().__init__()
        # 形状为 (max_len, d_model) 的矩阵（max_len 是最大句子序列长度，d_model 是模型维度），每行代表一个位置的编码。
        pe=torch.zeros(max_len,d_model)
        # 每个单词在句子中的位置，从0开始，形状为[max_len,]。unsqueeze在第1维上扩展，形状为[max_len,1]，方便后续广播计算
        position=torch.arange(0,max_len).float().unsqueeze(1)
        # 计算每一行（单词）在不同表征维度上的差异点，形状为[d_model/2,]
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        # 通过广播机制计算位置编码在不同表征维度上的值，两个多维张量的广播机制从最后一个维度开始，[max_len,1]*[d_model/2,]=[max_len,d_model/2]
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        # 第一步扩展维度，batch_size，第二步转换维度，将batch_size和max_len维度转换，保证和输入的x维度意义一致
        pe=pe.unsqueeze(0).transpose(0,1)
        # register_buffer 是 PyTorch 中 nn.Module 提供的方法，用于注册不参与训练的参数（即不会被优化器更新的张量）。
        # 注册后，pe 会被存储为模块的属性 self.pe，可以在 forward 方法中直接使用。
        # 同时，self.pe 会随模型一起被移动到设备（如 GPU），并在保存 / 加载模型时被自动处理。
        self.register_buffer('pe',pe)

    def forward(self,x):
        # x: (seq_len, batch_size, d_model)，为什么要转换一下？
        # 将位置编码矩阵与输入序列的嵌入向量相加，为输入注入位置信息,位置编码矩阵的形状？
        x=x+self.pe[:x.size(0),:]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        assert d_model % num_heads ==0

        self.d_model=d_model
        self.num_heads=num_heads
        self.d_head=d_model//num_heads

        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        self.w_o=nn.Linear(d_model,d_model)

    def scale_dot_product_attention(self,q,k,v,mask):
        # q:[batch_size,num_heads,seq_len_q,d_head]查询向量
        # k:[batch_size,num_heads,seq_len_k,d_head]键向量
        # v:[batch_size,num_heads,seq_len_v,d_head]值向量
        # 计算查询向量和键向量的注意力得分
        # atten_scores=[batch_size,num_heads,seq_len_q,d_head]*[batch_size,num_heads,d_head,seq_len_k]=
        # [batch_size,num_heads,seq_len_q,seq_len_k]
        attn_scores=torch.matmul(q,k.transpose(-1,-2))/math.sqrt(self.d_head)
        # 如果有掩码，则对得分进行一次掩码计算，使得后面计算概率值的时候为0
        if mask is not None:
            attn_scores=torch.masked_fill(attn_scores,mask==1,-1e-9)
        # 最后一个维度归一化，相当于查询单词所对应的每一个键单词得分都归一化
        attn_proba=F.softmax(attn_scores,dim=-1)
        # 多维数组的乘法，只考虑最后2个维度，前面统一作为批次按照广播机制保留
        # output=[batch_size,num_heads,seq_len_q,seq_len_k]*[batch_size,num_heads,seq_len_v,d_head]=
        # [batch_size,num_heads,seq_len_q,d_head]
        output=torch.matmul(attn_proba,v)
        return attn_proba,output

    def forward(self,q,k,v,mask=None):
        # q:[batch_size,seq_len_q,d_model]查询向量
        # k:[batch_size,seq_len_k,d_model]键向量
        # v:[batch_size,seq_len_v,d_model]值向量
        batch_size=q.size(0)
        q=self.w_q(q).view(batch_size,-1,self.num_heads,self.d_head).transpose(1,2)
        k=self.w_k(k).view(batch_size,-1,self.num_heads,self.d_head).transpose(1,2)
        v=self.w_v(v).view(batch_size,-1,self.num_heads,self.d_head).transpose(1,2)

        attn_proba,output=self.scale_dot_product_attention(q,k,v,mask)
        output=output.transpose(1,2).contiguous().view(batch_size,-1,self.d_model)

        return self.w_o(output),attn_proba


class PositionWiseFeedForward(nn.Module):

    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()
        self.fc1=nn.Linear(d_model,d_ff)
        self.fc2=nn.Linear(d_ff,d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        x=self.fc1(x)
        # 只在第一个线性连接层上应用了激活函数和丢失率
        x=F.relu(x)
        x=self.dropout(x)
        x=self.fc2(x)
        return x

class EncoderLayer(nn.Module):

    def __init__(self,d_model,num_heads,d_ff,dropout=0.1):
        super().__init__()
        self.self_attn=MultiHeadAttention(d_model,num_heads)
        self.feed_forward=PositionWiseFeedForward(d_model,d_ff,dropout)

        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)

        self.dropout1=nn.Dropout(dropout)
        self.dropout2=nn.Dropout(dropout)

    def forward(self,x,mask):
        # 多头自注意力 + 残差连接,注意这里的输出需新建一个变量接收，不能使用x，因为后面要残差连接，保持原来的x
        attn_output,_=self.self_attn(x,x,x,mask)
        x=self.norm1(x+self.dropout1(attn_output))

        # 前馈网络 + 残差连接
        ff_output=self.feed_forward(x)
        x=self.norm2(x+self.dropout2(ff_output))

        return x


class DecoderLayer(nn.Module):

    def __init__(self,d_model,num_heads,d_ff,dropout=0.1):
        super().__init__()
        self.self_attn=MultiHeadAttention(d_model,num_heads)
        self.cross_attn=MultiHeadAttention(d_model,num_heads)
        self.feed_forward=PositionWiseFeedForward(d_model,d_ff)

        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.norm3=nn.LayerNorm(d_model)

        self.dropout1=nn.Dropout(dropout)
        self.dropout2=nn.Dropout(dropout)
        self.dropout3=nn.Dropout(dropout)

    def forward(self,x,enc_output,self_mask,cross_mask):
        # 自注意力层 + 残差连接
        self_attn_output,_=self.self_attn(x,x,x,self_mask)
        x=self.norm1(x+self.dropout1(self_attn_output))

        # 交叉注意力层，以x为查询向量q，编码器层的输出向量为键值q,v,这里的x是自注意力层计算后的x
        cross_attn_output,_=self.cross_attn(x,enc_output,enc_output,cross_mask)
        x=self.norm2(x+self.dropout2(cross_attn_output))

        # 前馈连接层

        ff_output=self.feed_forward(x)
        x=self.norm3(x+self.dropout3(ff_output))

        return x


class Transformer(nn.Module):

    def __init__(self,src_vocab_size,tgt_vocab_size,d_model=512,num_heads=8,
                 num_layers=6,d_ff=2048,dropout=0.1,max_len=5000):
        super().__init__()
        # 编码器部分
        self.encoder_embedding=nn.Embedding(src_vocab_size,d_model)
        self.encoder_postion=PositionalEncoding(d_model,max_len)

        self.encoder_layers=nn.ModuleList(
           [EncoderLayer(d_model,num_heads,d_ff,dropout) for _ in range(num_layers)]
        )

        # 解码器部分
        self.decoder_embedding=nn.Embedding(tgt_vocab_size,d_model)
        self.decoder_postion=PositionalEncoding(d_model, max_len)
        self.decoder_layers=nn.ModuleList(
            [DecoderLayer(d_model,num_heads,d_ff,dropout) for _ in range(num_layers)]
        )

        self.fc=nn.Linear(d_model,tgt_vocab_size)  # 输出维度是目标语言的大小
        self.dropout=nn.Dropout(dropout)

    # src.shape[batch_size,src_len],tgt.shape[batch_size,tgt_len]
    def generate_mask(self,src,tgt):
        src_mask=(src!=0).unsqueeze(1).unsqueeze(2)  # [batch_size,1,1,src_len]
        tgt_mask=(tgt!=0).unsqueeze(1).unsqueeze(3)  # [batch_size,1,tgt_len,1]

        tgt_len=tgt.size(1)
        subsquent_mask=torch.triu(torch.ones(tgt_len,tgt_len),diagonal=1).bool() # 创建上三角矩阵
        tgt_mask=tgt_mask&(~subsquent_mask)
        return src_mask,tgt_mask

    # src.shape[batch_size,src_len,src_vocab_size],src_mask.shape[batch_size,1,1,src_len]
    def encode(self,src,src_mask):
        # x.shape[batch_size, src_len,d_model]
        x=self.encoder_embedding(src)*math.sqrt(self.encoder_embedding.embedding_dim)
        # x.shape[batch_size, src_len,d_model]
        x=self.encoder_postion(x.transpose(0,1)).transpose(0,1)
        x=self.dropout(x)
        # x.shape[batch_size,num_heads, src_len,d_heads]
        for layer in self.encoder_layers:
            x=layer(x,src_mask)
        return x

    def decode(self,tgt,enc_output,tgt_mask,cross_mask):
        x=self.decoder_embedding(tgt)*math.sqrt(self.decoder_embedding.embedding_dim)
        x=self.decoder_postion(x.transpose(0,1)).transpose(0,1)
        x = self.dropout(x)
        for layer in self.decoder_layers:
            x = layer(x,enc_output,tgt_mask,cross_mask)
        return x

    def forward(self,src,tgt):
        # 生成掩码
        src_mask,tgt_mask=self.generate_mask(src,tgt)
        # 编码器输出
        enc_output=self.encode(src,src_mask)
        # 解码器输出
        dec_output=self.decode(tgt,enc_output,tgt_mask,src_mask)
        # 最终输出
        output=self.fc(dec_output)
        return output


# 测试代码
if __name__ == "__main__":
    # 超参数设置
    src_vocab_size = 1000  # 源语言词汇表大小
    tgt_vocab_size = 1000  # 目标语言词汇表大小
    d_model = 512
    num_heads = 8
    num_layers = 3
    d_ff = 2048

    # 创建模型
    transformer = Transformer(
        src_vocab_size, tgt_vocab_size,
        d_model, num_heads, num_layers, d_ff
    )

    # 随机生成输入（batch_size=2, src_len=10, tgt_len=8）
    src = torch.randint(1, src_vocab_size, (2, 10))  # 源序列（避免0，0作为填充符）
    tgt = torch.randint(1, tgt_vocab_size, (2, 8))  # 目标序列

    # 前向传播
    output = transformer(src, tgt)
    print(f"输入源序列形状: {src.shape}")
    print(f"输入目标序列形状: {tgt.shape}")
    print(f"输出序列形状: {output.shape}")  # 应输出 (2, 8, tgt_vocab_size)
