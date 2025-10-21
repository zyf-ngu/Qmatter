import torch
from torch import nn

#从自己创建的models库里导入LeNet模块
#import LeNet 仅仅是把LeNet.py导入进来,当我们创建LeNet的实例的时候需要通过指定LeNet.py中的具体类.
#例如:我的LeNet.py中的类名是LeNet,则后面的模型实例化LeNet需要通过**LeNet.LeNet()**来操作
#还可以通过 from 还可以通过 from LeNet import * 直接把LeNet.py中除了以 _ 开头的内容都导入

from tradtional_models.models.cv import LeNet
#from LeNet import *
# lr_scheduler：提供一些根据epoch训练次数来调整学习率的方法
from torch.optim import lr_scheduler
# torchvision：PyTorch的一个图形库，服务于PyTorch深度学习框架的，主要用来构建计算机视觉模型
# transforms：主要是用于常见的一些图形变换
# datasets：包含加载数据的函数及常用的数据集接口
from torchvision import datasets,transforms
# os：operating system（操作系统），os模块封装了常见的文件和目录操作
import os

import torch
from torchvision import datasets, transforms
import torchvision.datasets.mnist as mnist
import os


# 替换MNIST数据集的下载链接为阿里云镜像
def set_mnist_mirror():
    # 阿里云MNIST镜像地址
    mirror_url = "https://oss.aliyuncs.com/xianyunpan/data/mnist/"

    # 原始文件列表与对应的镜像文件
    file_list = [
        ("train-images-idx3-ubyte.gz", "train-images-idx3-ubyte.gz"),
        ("train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte.gz"),
        ("t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz"),
        ("t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
    ]

    # 替换mnist的下载链接
    for filename, mirror_filename in file_list:
        mnist.URLs[filename] = f"{mirror_url}{mirror_filename}"


# 数据预处理
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
])

# 设置国内镜像
set_mnist_mirror()

# 加载数据集（此时会使用国内镜像下载）
try:
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        transform=data_transform,
        download=True  # 现在可以自动下载了
    )

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        transform=data_transform,
        download=True
    )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print("数据集加载成功！")
except Exception as e:
    print(f"加载数据集时出错: {e}")

# 设置数据转化方式，如数据转化为Tensor格式，数据切割等
# Compose()：将多个transforms的操作整合在一起
# ToTensor(): 将numpy的ndarray或PIL.Image读的图片转换成形状为(C,H, W)的Tensor格式，且归一化到[0,1.0]之间
# data_transform=transforms.Compose([transforms.ToTensor()])
# 加载训练数据集
# MNIST数据集来自美国国家标准与技术研究所, 训练集 (training set)、测试集(test set)由分别由来自250个不同人手写的数字构成
# MNIST数据集包含：Training set images、Training set images、Test set images、Test set labels
# train = true是训练集，false为测试集
# train_dataset=datasets.MNIST(root='./data',train=True,transform=data_transform,download=True)
# DataLoader：将读取的数据按照batch size大小封装并行训练
# dataset (Dataset)：加载的数据集
# batch_size (int, optional)：每个batch加载多少个样本(默认: 1)
# shuffle (bool, optional)：设置为True时会在每个epoch重新打乱数据(默认: False)
train_dataloader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=16,shuffle=True)

# 加载测试数据集
# test_dataset=datasets.MNIST(root='./data',train=False,transform=data_transform,download=False)
test_dataloader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=16,shuffle=True)
# 如果有NVIDA显卡，转到GPU训练，否则用CPU
device='cuda' if torch.cuda.is_available() else 'cpu'


# 模型实例化，将模型转到device
model= LeNet.LeNet().to(device)
# 定义损失函数（交叉熵损失）
loss_fn=nn.CrossEntropyLoss()
# 定义优化器(随机梯度下降法)
# params(iterable)：要训练的参数，一般传入的是model.parameters()
# lr(float)：learning_rate学习率，也就是步长
# momentum(float, 可选)：动量因子（默认：0），矫正优化率
optimizer=torch.optim.SGD(model.parameters(),lr=1e-3,momentum=0.9)
# 学习率，每隔10轮变为原来的0.1
# StepLR：用于调整学习率，一般情况下会设置随着epoch的增大而逐渐减小学习率从而达到更好的训练效果
# optimizer （Optimizer）：需要更改学习率的优化器
# step_size（int）：每训练step_size个epoch，更新一次参数
# gamma（float）：更新lr的乘法因子
lr_scheduler=lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)

# 定义训练函数
def train(dataloader,model,loss_fn,optimizer):
    loss,current,n=0.0,0.0,0
    # dataloader: 传入数据（数据包括：训练数据和标签）
    # x：训练数据（inputs）(tensor类型的），y：标签（labels）(tensor类型的）
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，
    # 一般用在 for 循环当中。可以用这个循环加载batchsize下面的每一元组，来得到相应的图片及标签
    # 和dataloader结合使用时返回数据下标是batch（在创建dataloader时会把batch size作为参数传入），
    # 从0开始，最大数为样本总数除以batch size大小，数据是一个batch的数据和标签
    for batch, (x,y) in enumerate(dataloader):
        # 前向传播
        x,y=x.to(device),y.to(device)
        # 计算训练值
        output=model(x)
        # 计算观测值（label）与训练值的损失函数
        cur_loss=loss_fn(output,y)

        print("output:",output)
        print("y:",y)
        # torch.max(input, dim)函数
        # input是具体的tensor，dim是max函数索引的维度，0是每列的最大值，1是每行的最大值输出
        # 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引
        _,pred=torch.max(output,axis=1)
        # 计算每批次的准确率
        # output.shape[0]一维长度为该批次的数量
        # torch.sum()对输入的tensor数据的某一维度求和
        cur_acc=torch.sum(y==pred)/output.shape[0]
        # 反向传播
        # 清空过往梯度
        optimizer.zero_grad()
        # 反向传播，计算当前梯度
        cur_loss.backward()
        # 根据梯度更新网络参数
        optimizer.step()
        # .item()：得到元素张量的元素值
        loss+=cur_loss.item()
        current+=cur_acc.item()
        n=n+1

    train_loss=loss/n
    train_acc=current/n
    # 计算训练的错误率
    print('train_loss'+str(train_loss))
    # 计算训练的准确率
    print('train_acc'+str(train_acc))

# 定义验证函数
def val(dataloader,model,loss_fn):
    # model.eval()：设置为验证模式，如果模型中有Batch Normalization或Dropout，则不启用，以防改变权值
    model.eval()
    loss,current,n=0.0,0.0,.0
    # with torch.no_grad()：将with语句包裹起来的部分停止梯度的更新，从而节省了GPU算力和显存，但是并不会影响dropout和BN层的行为
    with torch.no_grad():
        for batch,(x,y) in enumerate(dataloader):
            # 前向传播
            x,y=x.to(device),y.to(device)
            output=model(x)
            cur_loss=loss_fn(output,y)
            _,pred=torch.max(output,axis=1)
            cur_acc=torch.sum(y==pred)/output.shape[0]
            loss+=cur_loss.item()
            current+=cur_acc.item()
            n=n+1
        # 计算验证的错误率
        print("val_loss:"+str(loss/n))
        # 计算验证的准确率
        print("val_acc：" + str(current / n))
        # 返回模型准确率
        return current / n
# 开始训练
# 训练次数
epoch=10
# 用于判断最佳模型
min_acc=0
for t in range(epoch):
    print(f'epoch {t+1}\n --------')
    # 训练模型
    train(train_dataloader,model,loss_fn,optimizer)
    # 验证模型
    a=val( test_dataloader,model,loss_fn)
    # 保存最好的模型权重
    if a>min_acc:
        folder='save_model'
        # path.exists：判断括号里的文件是否存在，存在为True，括号内可以是文件路径
        if not os.path.exists(folder):
            # os.mkdir() ：用于以数字权限模式创建目录
            os.mkdir('../save_model')
        min_acc=a
        print('save better model')
        # torch.save(state, dir)保存模型等相关参数，dir表示保存文件的路径+保存文件名
        # model.state_dict()：返回的是一个OrderedDict，存储了网络结构的名字和对应的参数
        torch.save(model.state_dict(), '../save_model/better_model.pth')

print('Done!')



