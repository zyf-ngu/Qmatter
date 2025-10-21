#import LSTM 仅仅是把LSTM.py导入进来,当我们创建LSTM的实例的时候需要通过指定LSTM.py中的具体类.
#例如:我的LSTM.py中的类名是LSTM,则后面的模型实例化LSTM需要通过**LSTM.LSTM()**来操作
#还可以通过 from 还可以通过 from LSTM import * 直接把LSTM.py中除了以 _ 开头的内容都导入
from Qmatter.models.nlp import LSTM
from Qmatter.models.nlp.LSTM import *

import datetime
import torch.optim as optim
#导入画图的库，后面将主要学习使用axes方法来画图
from matplotlib import pyplot as plt

batch_size=2#批大小
T_seq=30#输入序列长度(时间步）
feature_size=3#输入特征维度

hidden_size=3#隐含层维度
output_size=2#输出层维度

num_layers=1
lr_rate=0.001
epoch=1000
#input 即LSTM网络的输入，维度应该为(T_seq, batch_size, input_size)。如果设置batch_first=True，输入维度则为(batch, seq_len, input_size)
input=torch.randn(batch_size,T_seq,feature_size)

def train(input):
    model= LSTM(feature_size, hidden_size, num_layers, output_size)
    print("model:\n",model)
    # 设置损失函数
    loss_fn=nn.MSELoss()
    # 设置优化器
    optimizer=optim.Adam(model.parameters(),lr_rate)
    # 初始化h_prev，它和输入x本质是一样的，hidden_size就是它的特征维度
    #维度应该为(num_layers * num_directions, batch, hidden_size)。num_layers表示堆叠的RNN网络的层数。
    # 对于双向RNNs而言num_directions= 2，对于单向RNNs而言，num_directions= 1
    hidden_prev=torch.zeros(1,batch_size,hidden_size)
    c_prev=torch.zeros(1,batch_size,hidden_size)
    loss_plt=[]
    #开始训练
    for iter in range(epoch):
        x = input
        print("x:", x.shape)

        output,(hidden_prev,c_prev)=model(x,hidden_prev,c_prev)
        print("output_size:",output.shape)
        y = torch.randn(batch_size,T_seq,output_size)
        print("y:", y.shape)
        #返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,
        # 不同之处只是requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad。
        hidden_prev=hidden_prev.detach()
        c_prev = c_prev.detach()

        loss=loss_fn(output,y)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        if iter%100==0:
            print("iteration:{} loss {}".format(iter,loss.item()))
            loss_plt.append(loss.item())

    fig,ax=plt.subplots(1,1)
    ax.plot(loss_plt, 'r')
    ax.set_xlabel('epcoh')
    ax.set_ylabel('loss')
    ax.set_title('LSTM-train-loss')

    return hidden_prev,c_prev, model


if __name__ == '__main__':
    # 计算训练时间，结束时间减去开始时间
    start_time = datetime.datetime.now()
    hidden_pre,c_prev, model = train(input)
    end_time = datetime.datetime.now()
    print('The training time: %s' % str(end_time - start_time))
    plt.show()
