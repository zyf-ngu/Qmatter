import torch
import torch.nn as nn
import random

#直接使用pytorch自带的LSTM类
#可以看到，LSTM网络也是继承自nn.Module的
class LSTM(nn.Module):
    # 这里的输入参数和RNN相同，包括最开始输入特征“词向量”维度，隐藏层的每个隐状态的特征维度，隐藏层数量，输出层的特征维度（一般和隐状态特征维度一致）
    def __init__(self,feature_size,hidden_size,num_layers,output_size):
        super(LSTM,self).__init__()
        self.lstm=nn.LSTM(
            input_size=feature_size,hidden_size=hidden_size,
            num_layers=num_layers,batch_first=True
        )
        for k in self.lstm.parameters():
            nn.init.normal_(k,mean=0.0,std=0.001)
            # 输入和输出的维度可以是任意,只需要保证最后一个维度是特征维度in_features&out_features就行
            # #- Input: :math:`(*, H_{in})` where :math:`*` means any number of
            #   dimensions including none and :math:`H_{in} = \text{in\_features}`.
            # - Output: :math:`(*, H_{out})` where all but the last dimension
            #   are the same shape as the input and :math:`H_{out} = \text{out\_features}
            # Examples::
            # >> > m = nn.Linear(20, 30)
            # >> > input = torch.randn(128, 20)
            # >> > output = m(input)
            # >> > print(output.size())
            # torch.Size([128, 30])
        self.linear=nn.Linear(hidden_size,output_size)
        self.hidden_size=hidden_size

    # 这里比RNN多了一个参数，c_prev，这也是LSTM的核心单元状态，具体参照原理讲解
    def forward(self,x,hidden_prev,c_prev):
        # 每一次调用rnn层返回的就是输出层和隐状态值和单元状态，隐状态和单元状态又是下一循环的上一状态值，所以用hidden_prev&c_prev表示
        out,(hidden_prev,c_prev)=self.lstm(x,(hidden_prev,c_prev))
        print("out1&hidden_prev.shape",out.shape,hidden_prev.shape)
        #view()相当于reshape、resize，重新调整PyTorch 中的 Tensor 形状，若非 Tensor 类型，可使用 data = torch.tensor(data)来进行转换。
        #out=out.view(-1,self.hidden_size)
        print("out2.shape", out.shape)
        out=self.linear(out)
        print("out3.shape", out.shape)
       # out=out.unsqueeze(0)
        print("out4.shape", out.shape)
        #输出的维度是batch_size*T_seq*hidden_size
        return out,(hidden_prev,c_prev)


#自己实现一个LSTMN函数
#这里的函数参数需要手动给定网络结构参数，
def My_LSTM(input,initial_states,w_ih,w_hh,b_ih,b_hh):
    #比RNN多了一个初始状态c0
    h0,c0=initial_states
    batch_size,T_seq,feature_size=input.shape
    hidden_size=w_ih.shape[0]//4

    prev_h=h0
    prev_c=c0
    batch_w_ih=w_ih.unsqueeze(0).tile(batch_size,1,1)
    batch_w_hh=w_hh.unsqueeze(0).tile(batch_size,1,1)

    output_feature_size=hidden_size
    output=torch.zeros(batch_size,T_seq,output_feature_size)

    for t in range(T_seq):
        #当前时刻的输入向量，(batch_size*feature_size)
        x=input[:,t,:]
        #计算两个tensor的矩阵乘法，torch.bmm(a,b),tensor a 的size为(b,h,w),tensor b的size为(b,w,m)
        # 也就是说两个tensor的第一维是相等的，然后第一个数组的第三维和第二个数组的第二维度要求一样，其实就是第一维不变，后面二维张量相乘，h*w*w*m=h*m
        # 对于剩下的则不做要求，输出维度 （b,h,m）
        # batch_w_ih=batch_size*(4*hidden_size)*feature_size
        #x=batch_size*feature_size*1
        #w_times_x=batch_size*(4*hidden_size)*1
        ##squeeze，在给定维度（维度值必须为1）上压缩维度，负数代表从后开始数
        w_times_x=torch.bmm(batch_w_ih,x.unsqueeze(-1))
        w_times_x=w_times_x.squeeze(-1)
       # print(batch_w_ih.shape, x.shape)
        # batch_w_hh=batch_size*(4*hidden_size)*hidden_size
        # prev_h=batch_size*hidden_size*1
        # w_times_h_prev=batch_size*(4*hidden_size)*1
       # print(batch_w_hh.shape,prev_h.shape)
        w_times_h_prev=torch.bmm(batch_w_hh,prev_h.unsqueeze(-1))
        w_times_h_prev=w_times_h_prev.squeeze(-1)

        #分别计算输入门(i),遗忘门(f)，cell门(g)，输出门(o)，这里可以看到参数是共享的
        i_t=torch.sigmoid(w_times_x[:,:hidden_size]+b_ih[:hidden_size]+
                          w_times_h_prev[:,:hidden_size]+b_hh[:hidden_size])
        f_t = torch.sigmoid(w_times_x[:, hidden_size:2 * hidden_size] + b_ih[hidden_size:2 * hidden_size] +
                            w_times_h_prev[:, hidden_size:2 * hidden_size] + b_hh[hidden_size:2 * hidden_size])
        g_t = torch.sigmoid(w_times_x[:, 2*hidden_size:3 * hidden_size] + b_ih[2*hidden_size:3 * hidden_size] +
                            w_times_h_prev[:, 2*hidden_size:3 * hidden_size] + b_hh[2*hidden_size:3 * hidden_size])
        o_t = torch.sigmoid(w_times_x[:, 3 * hidden_size:4 * hidden_size] + b_ih[3 * hidden_size:4 * hidden_size] +
                            w_times_h_prev[:, 3 * hidden_size:4 * hidden_size] + b_hh[3 * hidden_size:4 * hidden_size])
        prev_c=f_t*prev_c+i_t*g_t
        prev_h=o_t*torch.tanh(prev_c)
        output[:,t,:]=prev_h

    return output,(prev_h,prev_c)




# 测试代码
        # 每个python模块（python文件）都包含内置的变量 __name__，当该模块被直接执行的时候，__name__ 等于文件名（包含后缀 .py ）
        # 如果该模块 import 到其他模块中，则该模块的 __name__ 等于模块名称（不包含后缀.py）
        # “__main__” 始终指当前执行模块的名称（包含后缀.py）
        # if确保只有单独运行该模块时，此表达式才成立，才可以进入此判断语法，执行其中的测试代码，反之不行

if __name__=="__main__":
    batch_size=2
    T_seq=5
    feature_size=8
    hidden_size=6

    input=torch.randn(batch_size,T_seq,feature_size)
    c0=torch.randn(batch_size,hidden_size)
    h0=torch.randn(batch_size,hidden_size)

    lstm_layer=nn.LSTM(feature_size,hidden_size,batch_first=True)
    output,(h_final,c_final)=lstm_layer(input,(h0.unsqueeze(0),c0.unsqueeze(0)))
    print(output,(h_final,c_final))
    #.named_parameters()遍历得到网络参数
    for k,v in lstm_layer.named_parameters():
        print(k,v.shape)

    my_output,(my_h_final,my_c_final)=My_LSTM(input,(h0,c0),lstm_layer.weight_ih_l0,lstm_layer.weight_hh_l0,
                                              lstm_layer.bias_ih_l0,lstm_layer.bias_hh_l0)

    print(my_output,(my_h_final,my_c_final))