import torch
import torch.nn as nn


def my_gru(input,initial_states,w_ih,w_hh,b_ih,b_hh):
    h_prev=initial_states
    batch_size,T_seq,feature_size=input.shape
    hidden_size=w_ih.shape[0]//3

    batch_w_ih=w_ih.unsqueeze(0).tile(batch_size,1,1)
    batch_w_hh=w_hh.unsqueeze(0).tile(batch_size,1,1)

    output=torch.zeros(batch_size,T_seq,hidden_size)

    for t in range(T_seq):
        x=input[:,t,:]
        w_times_x=torch.bmm(batch_w_ih,x.unsqueeze(-1))
        w_times_x=w_times_x.squeeze(-1)

       # print(batch_w_hh.shape,h_prev.shape)
        # 计算两个tensor的矩阵乘法，torch.bmm(a,b),tensor a 的size为(b,h,w),tensor b的size为(b,w,m)
        # 也就是说两个tensor的第一维是相等的，然后第一个数组的第三维和第二个数组的第二维度要求一样，
        # 对于剩下的则不做要求，输出维度 （b,h,m）
        # batch_w_hh=batch_size*(3*hidden_size)*hidden_size
        # h_prev=batch_size*hidden_size*1
        # w_times_x=batch_size*hidden_size*1
        ##squeeze，在给定维度（维度值必须为1）上压缩维度，负数代表从后开始数
        w_times_h_prev=torch.bmm(batch_w_hh,h_prev.unsqueeze(-1))
        w_times_h_prev=w_times_h_prev.squeeze(-1)

        r_t=torch.sigmoid(w_times_x[:,:hidden_size]+w_times_h_prev[:,:hidden_size]+b_ih[:hidden_size]
                          +b_hh[:hidden_size])
        z_t=torch.sigmoid(w_times_x[:,hidden_size:2*hidden_size]+w_times_h_prev[:,hidden_size:2*hidden_size]
                          +b_ih[hidden_size:2*hidden_size]+b_hh[hidden_size:2*hidden_size])
        n_t=torch.tanh(w_times_x[:,2*hidden_size:3*hidden_size]+w_times_h_prev[:,2*hidden_size:3*hidden_size]
                          +b_ih[2*hidden_size:3*hidden_size]+b_hh[2*hidden_size:3*hidden_size])

        h_prev=(1-z_t)*n_t+z_t*h_prev
        output[:,t,:]=h_prev

    return output,h_prev




if __name__=="__main__":

    fc=nn.Linear(12,6)
    

    batch_size=2
    T_seq=5
    feature_size=4

    hidden_size=3
   # output_feature_size=3

    input=torch.randn(batch_size,T_seq,feature_size)
    h_prev=torch.randn(batch_size,hidden_size)

    gru_layer=nn.GRU(feature_size,hidden_size,batch_first=True)
    output,h_final=gru_layer(input,h_prev.unsqueeze(0))
    # for k,v in gru_layer.named_parameters():
    #     print(k,v.shape)
    # print(output,h_final)

    my_output, my_h_final=my_gru(input,h_prev,gru_layer.weight_ih_l0,gru_layer.weight_hh_l0,gru_layer.bias_ih_l0,gru_layer.bias_hh_l0)

    # print(my_output, my_h_final)
    # print(torch.allclose(output,my_output))
