import torch
import torch.nn as nn

import random
#直接使用pytorch自带的RNN类
#可以看到，RNN网络也是继承自nn.Module的
class RNN(nn.Module):
    #这里的输入参数包括最开始输入特征“词向量”维度，隐藏层的每个隐状态的特征维度，隐藏层数量，输出层的特征维度（一般和隐状态特征维度一致）
    def __init__(self,feature_size,hidden_size,num_layers,output_size):
        super(RNN,self).__init__()

        self.rnn=nn.RNN(
            input_size=feature_size,hidden_size=hidden_size,
            num_layers=num_layers,batch_first=True
        )
        #参数初始化，
        for k in self.rnn.parameters():
            nn.init.normal_(k,mean=0.0,std=0.001)
        #linear层的输入和输出的维度可以是任意的,只需要保证最后一个维度是特征维度in_features&out_features就行
        # #- Input: :math:`(*, H_{in})` where :math:`*` means any number of
        #   dimensions including none and :math:`H_{in} = \text{in\_features}`.
        # - Output: :math:`(*, H_{out})` where all but the last dimension
        #   are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
        #
        # Examples::
        # >> > m = nn.Linear(20, 30)
        # >> > input = torch.randn(128, 20)
        # >> > output = m(input)
        # >> > print(output.size())
        # torch.Size([128, 30])
        self.linear=nn.Linear(hidden_size,output_size)
        self.hidden_size=hidden_size

    def forward(self,x,hidden_prev):
        #每一次调用rnn层返回的就是输出层和隐状态值，隐状态又是下一循环的上一状态值，所以用hidden_prev表示
        out,hidden_prev=self.rnn(x,hidden_prev)
        print("out1&hidden_prev.shape",out.shape,hidden_prev.shape)
        #view()相当于reshape、resize，重新调整PyTorch 中的 Tensor 形状，若非 Tensor 类型，可使用 data = torch.tensor(data)来进行转换。
       # out=out.view(-1,self.hidden_size)
        print("out2.shape", out.shape)
        out=self.linear(out)
        print("out3.shape", out.shape)
       # out=out.unsqueeze(0)
        print("out4.shape", out.shape)
        #输出的维度是batch_size*T_seq*hidden_size
        return out,hidden_prev

#自己实现一个RNN函数
#这里的函数参数需要手动给定网络结构参数，
def rnn_forward(input,weight_ih,weight_hh,bias_ih,bias_hh,h_prev):
    #input的shape就是batch_size*T_seq*feature_size(设置batch_first=TRUE)
    batch_size,T_seq,feature_size=input.shape
    hidden_size=weight_ih.shape[0]
    h_out=torch.zeros(batch_size,T_seq,hidden_size)
    for t in range(T_seq):
        x=input[:,t,:].unsqueeze(2)
      #  print("xt.shape",x.shape)
            #unsqueeze，在给定维度上（从0开始）扩展1个维度，负数代表从后开始数
        #具体到下面，就是先在第0维度上扩展成1*hidden_size*feature_size；
        # 然后.tile就是在第0维度复制batch_size次，变成batch_size*hidden_size*feature_size
        weight_ih_batch=weight_ih.unsqueeze(0).tile(batch_size,1,1)
     #   print("weight_ih_batch.shape", weight_ih_batch.shape)
        weight_hh_batch=weight_hh.unsqueeze(0).tile(batch_size,1,1)
     #   print("weight_hh_batch.shape", weight_hh_batch.shape)

        #计算两个tensor的矩阵乘法，torch.bmm(a,b),tensor a 的size为(b,h,w),tensor b的size为(b,w,m)
        # 也就是说两个tensor的第一维是相等的，然后第一个数组的第三维和第二个数组的第二维度要求一样，其实就是第一维不变，后面二维张量相乘，h*w*w*m=h*m
        # 对于剩下的则不做要求，输出维度 （b,h,m）
        # weight_ih_batch=batch_size*hidden_size*feature_size
        #x=batch_size*feature_size*1
        #w_times_x=batch_size*hidden_size*1
        ##squeeze，在给定维度（维度值必须为1）上压缩维度，负数代表从后开始数

        w_times_x=torch.bmm(weight_ih_batch,x).squeeze(-1)#
      #  print("w_times_x.shape", w_times_x.shape)

        w_times_h=torch.bmm(weight_hh_batch,h_prev.unsqueeze(2)).squeeze(-1)
      #  print("w_times_h.shape", w_times_h.shape)

        h_prev=torch.tanh(w_times_x+bias_ih+w_times_h+bias_hh)
        print("h_prev.shape", h_prev.shape)
        h_out[:,t,:]=h_prev
        print("h_out.shape", h_out.shape)

    return h_out,h_prev.unsqueeze(0)

# 测试代码
        # 每个python模块（python文件）都包含内置的变量 __name__，当该模块被直接执行的时候，__name__ 等于文件名（包含后缀 .py ）
        # 如果该模块 import 到其他模块中，则该模块的 __name__ 等于模块名称（不包含后缀.py）
        # “__main__” 始终指当前执行模块的名称（包含后缀.py）
        # if确保只有单独运行该模块时，此表达式才成立，才可以进入此判断语法，执行其中的测试代码，反之不行
if __name__=="__main__":
    # input=torch.randn(batch_size,T_seq,feature_size)
    # h_prev=torch.zeros(batch_size,hidden_size)

    # rnn=nn.RNN(input_size,hidden_size,batch_first=True)
    # output,state_final=rnn(input,h_prev.unsqueeze(0))

    # print(output)
    # print(state_final)
    batch_size, T_seq =10, 30  # 批大小，输入序列长度
    feature_size, hidden_size = 5, 8  #
    num_layers, output_size=1,3
    input = torch.randn(batch_size, T_seq, feature_size)
    h_prev = torch.zeros(1,batch_size, hidden_size)#.unsqueeze(0)
    # my_rnn=RNN(feature_size,hidden_size,num_layers,output_size)
    rnn=nn.RNN(feature_size,hidden_size,batch_first=True)
    # rnn_output, state_final = rnn(input, h_prev.unsqueeze(0))
    # for k,v in rnn.named_parameters():
    #     print(k,v.shape)
    my_rnn_output,my_state_final=rnn_forward(input,rnn.weight_ih_l0,rnn.weight_hh_l0,
                                             rnn.bias_ih_l0,rnn.bias_hh_l0,h_prev)
    print(my_rnn_output.shape)
    print(my_state_final.shape)







