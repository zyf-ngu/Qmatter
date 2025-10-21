#import seq2seq 仅仅是把seq2seq.py导入进来,当我们创建seq2seq的实例的时候需要通过指定seq2seq.py中的具体类.
#例如:我的seq2seq.py中的类名是seq2seq,则后面的模型实例化seq2seq需要通过**seq2seq.seq2seq()**来操作
#还可以通过 from 还可以通过 from seq2seq import * 直接把seq2seq.py中除了以 _ 开头的内容都导入

from Qmatter.models.nlp.seq2seq import *

import numpy as np


# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output

# P: Symbol that will fill in blank sequence if current batch data size is short than time steps, pad 补充，不够长度就pad


##  seq_data = [['man', 'women'], ['black', 'white']]
def make_batch():
    input_batch, output_batch, target_batch = [], [], []

    for seq in seq_data:
        for i in range(2):
            seq[i] = seq[i] + 'P' * (n_step - len(seq[i]))  ### 不够长度的 补充pad
            print(" seq[i] =", seq[i])
        input = [num_dic[n] for n in seq[0]]  ##  seq = ['manPP', 'women']
        output = [num_dic[n] for n in ('S' + seq[1])]
        # output = [num_dic[n] for n in ('S' + 'P' * n_step)]  ## test is ok ?
        target = [num_dic[n] for n in (seq[1] + 'E')]  ### 表示输出结果

        input_batch.append(np.eye(n_class)[input])  ## np.eye(n_class)[input]  生成 one-hot词向量  5*29
        output_batch.append(np.eye(n_class)[output])
        target_batch.append(target)  # not one-hot

    # make tensor
    return torch.FloatTensor(input_batch), torch.FloatTensor(output_batch), torch.LongTensor(target_batch)


# make test batch 测试数据构建
def make_testbatch(input_word):
    input_batch, output_batch = [], []

    input_w = input_word + 'P' * (n_step - len(input_word))
    input = [num_dic[n] for n in input_w]
    output = [num_dic[n] for n in ('S' + 'P' * n_step)]

    input_batch = np.eye(n_class)[input]
    output_batch = np.eye(n_class)[output]

    return torch.FloatTensor(input_batch).unsqueeze(0), torch.FloatTensor(output_batch).unsqueeze(0)

if __name__ == '__main__':

    n_step = 5  ##单词长度，不够的用padding补充
    hidden_dim = 128
    emb_dim=25
# 创建一个含有26个字母以及S E P的字母列表
    char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz']
    # 创建一个字母列表的字母和下标为键值对的字典
    num_dic = {n: i for i, n in enumerate(char_arr)}
    # 数据序列是batch_size*seq_len，句子的长度seq_len为2.
    seq_data = [['man', 'man'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'down'], ['high', 'low']]

    n_class = len(num_dic)
    batch_size = len(seq_data)

    encoder=Encoder(n_class,emb_dim,hidden_dim,1,0.5)
    decoder=Decoder(hidden_dim,emb_dim,hidden_dim,1,0.5)
    device='cuda' if torch.cuda.is_available() else 'cpu'

    model = Seq2Seq(encoder,decoder,device)
# 设置损失函数
    criterion = nn.CrossEntropyLoss()
    #设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 预处理数据，将自然语言转换成数字
    input_batch, output_batch, target_batch = make_batch()
    input_batch=input_batch.long()

    for epoch in range(5000):
        # make hidden shape [num_layers * num_directions, batch_size, n_hidden]
        hidden = torch.zeros(1, batch_size,hidden_dim)  ## 隐层向量初始化
        # pdb.set_trace()

        optimizer.zero_grad()
        # input_batch : [batch_size, max_len(=n_step, time step), n_class]
        # output_batch : [batch_size, max_len+1(=n_step, time step) (becase of 'S' or 'E'), n_class]
        # target_batch : [batch_size, max_len+1(=n_step, time step)], not one-hot
        print("input_batch.shape:",input_batch.shape)
        output = model(input_batch, hidden, output_batch)
        # output : [max_len+1, batch_size, n_class]
        output = output.transpose(0, 1)  # [batch_size, max_len+1(=6), n_class]
        loss = 0
        for i in range(0, len(target_batch)):
            # output[i] : [max_len+1, n_class, target_batch[i] : max_len+1]
            loss += criterion(output[i], target_batch[i])
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

    print(' now is starting test ....')


    # Test
    def translate(word):
        input_batch, output_batch = make_testbatch(word)

        # make hidden shape [num_layers * num_directions, batch_size, n_hidden]
        hidden = torch.zeros(1, 1, n_hidden)  ## 隐层向量初始化
        output = model(input_batch, hidden, output_batch)
        # output : [max_len+1(=6), batch_size(=1), n_class]

        predict = output.data.max(2, keepdim=True)[1]  # select n_class dimension  get index
        decoded = [char_arr[i] for i in predict]
        end = decoded.index('E')
        translated = ''.join(decoded[:end])

        return translated.replace('P', '')


    print('test')
    print('man ->', translate('man'))
    print('mans ->', translate('mans'))
    print('king ->', translate('king'))
    print('black ->', translate('black'))
    print('ups ->', translate('ups'))