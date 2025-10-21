# 从自己创建的models库里导入AlexNet模块
# import AlexNet 仅仅是把AlexNet.py导入进来,当我们创建AlexNet的实例的时候需要通过指定AlexNet.py中的具体类.
# 例如:我的AlexNet.py中的类名是AlexNet,则后面的模型实例化AlexNet需要通过**AlexNet.AlexNet()**来操作
# 还可以通过 from 还可以通过 from AlexNet import * 直接把AlexNet.py中除了以 _ 开头的内容都导入
from tradtional_models.models.cv.AlexNet import *
# lr_scheduler：提供一些根据epoch训练次数来调整学习率的方法
import torch
from torch.optim import lr_scheduler
# torchvision：PyTorch的一个图形库，服务于PyTorch深度学习框架的，主要用来构建计算机视觉模型
# transforms：主要是用于常见的一些图形变换
# datasets：包含加载数据的函数及常用的数据集接口
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
# os：operating system（操作系统），os模块封装了常见的文件和目录操作
import os
# 导入画图的库，后面将主要学习使用axes方法来画图
import matplotlib.pyplot as plt

# 设置数据转化方式，如数据转化为Tensor格式，数据切割等
# Compose()：将多个transforms的操作整合在一起
# ToTensor(): 将numpy的ndarray或PIL.Image读的图片转换成形状为(C,H, W)的Tensor格式，且归一化到[0,1.0]之间
# compose的参数为列表[]
train_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    # normalize的意义
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
# ImageFolder(root, transform``=``None``, target_transform``=``None``, loader``=``default_loader)
# root 指定路径加载图片;  transform：对PIL Image进行的转换操作，transform的输入是使用loader读取图片的返回对象
# target_transform：对label的转换   loader：给定路径后如何读取图片，默认读取为RGB格式的PIL Image对象
# label是按照文件夹名顺序排序后存成字典，即{类名:类序号(从0开始)}，一般来说最好直接将文件夹命名为从0开始的数字，举例来说，两个类别，
# 狗和猫，把狗的图片放到文件夹名为0下；猫的图片放到文件夹名为1的下面。
# 这样会和ImageFolder实际的label一致， 如果不是这种命名规范，建议看看self.class_to_idx属性以了解label和文件夹名的映射关系
# python中\是转义字符，Windows 路径如果只有一个\，会把它识别为转义字符。
# 可以用r''把它转为原始字符，也可以用\\,也可以用Linux的路径字符/。
train_dataset = ImageFolder(r"E:\计算机\data\fer2013_数据增强版本\train", train_transform)
test_dataset = ImageFolder(r"E:\计算机\data\fer2013_数据增强版本\test", test_transform)

# DataLoader：将读取的数据按照batch size大小封装并行训练
# dataset (Dataset)：加载的数据集
# batch_size (int, optional)：每个batch加载多少个样本(默认: 1)
# shuffle (bool, optional)：设置为True时会在每个epoch重新打乱数据(默认: False)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = AlexNet().to(device)
# 定义损失函数（交叉熵损失）
loss_fn = nn.CrossEntropyLoss()
# 定义优化器(随机梯度下降法)
# params(iterable)：要训练的参数，一般传入的是model.parameters()
# lr(float)：learning_rate学习率，也就是步长
# momentum(float, 可选)：动量因子（默认：0），矫正优化率
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 学习率，每隔10轮变为原来的0.1
# StepLR：用于调整学习率，一般情况下会设置随着epoch的增大而逐渐减小学习率从而达到更好的训练效果
# optimizer （Optimizer）：需要更改学习率的优化器
# step_size（int）：每训练step_size个epoch，更新一次参数
# gamma（float）：更新lr的乘法因子
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


def train(train_dataloader, model, loss_fn, optimizer):
    loss, acc, n = 0.0, 0.0, 0
    # dataloader: 传入数据（数据包括：训练数据和标签）
    # enumerate()：用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，一般用在for循环当中
    # enumerate返回值有两个：一个是序号，一个是数据（包含训练数据和标签）
    # x：训练数据（inputs）(tensor类型的），y：标签（labels）(tensor类型的）
    # 和dataloader结合使用时返回数据下标是batch（在创建dataloader时会把batch size作为参数传入），
    # 从0开始，最大数为样本总数除以batch size大小，数据是一batch的数据和标签
    for batch, (x, y) in enumerate(train_dataloader):
        x, y = x.to(device), y.to(device)
        # print("x-shape",x.size())
        output = model(x)
        cur_loss = loss_fn(output, y)
        # torch.max(input, dim)函数
        # input是具体的tensor，dim是max函数索引的维度，0是每列的最大值，1是每行的最大值输出
        # 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引
        _, pred = torch.max(output, axis=1)
        # 计算每批次的准确率
        # output.shape[0]一维长度为该批次的数量
        # torch.sum()对输入的tensor数据的某一维度求和
        cur_acc = torch.sum(pred == y) / output.shape[0]

        # 清除过往梯度值，防止上个batch的数据的梯度值累积
        optimizer.zero_grad()
        # 后向传播
        cur_loss.backward()
        # 优化迭代
        optimizer.step()

        loss += cur_loss.item()
        acc += cur_acc.item()
        n = n + 1
    train_loss = loss / n
    train_acc = acc / n
    # 计算训练的损失函数变化
    print('train_loss==' + str(train_loss))
    # 计算训练的准确率
    print('train_acc' + str(train_acc))
    return train_loss, train_acc


# 测试函数里参数无优化器，不需要再训练，只需要测试和验证即可
def test(dataloader, model, loss_fn):
    loss, acc, n = 0.0, 0.0, 0

    ## model.eval()：设置为验证模式，如果模型中有Batch Normalization或Dropout，则不启用，以防改变权值
    model.eval()
    # with torch.no_grad()：将with语句包裹起来的部分停止梯度的更新，从而节省了GPU算力和显存，但是并不会影响dropout和BN层的行为
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(pred == y) / output.shape[0]
            loss += cur_loss.item()
            acc += cur_acc.item()
            n = n + 1
        test_loss = loss / n
        test_acc = acc / n
        print('test_loss==' + str(test_loss))
        # 计算训练的准确率
        print('train_acc' + str(test_acc))
        return test_loss, test_acc

    # 定义画图函数
    # 错误率


def matplot_loss(train_loss, test_loss):
    # 参数label = ''传入字符串类型的值，也就是图例的名称
    fig, ax = plt.subplots(1, 1)
    ax.plot(train_loss, label='train_loss')
    ax.plot(test_loss, label='test_loss')
    # loc代表了图例在整个坐标轴平面中的位置（一般选取'best'这个参数值）
    ax.legend(loc='best')
    ax.set_xlabel('loss')
    ax.set_ylabel('epoch')
    ax.set_title("训练集和验证集的loss值对比图")
    plt.show()

    # 准确率


def matplot_acc(train_acc, test_acc):
    fig, ax = plt.subplots(1, 1)
    ax.plot(train_acc, label='train_acc')
    ax.plot(test_acc, label='test_acc')
    ax.legend(loc='best')
    ax.set_xlabel('acc')
    ax.set_ylabel('epoch')
    ax.set_title("训练集和验证集的acc值对比图")
    plt.show()


loss_train = []
acc_train = []
loss_test = []
acc_test = []

epoch = 1
min_acc = 0
for t in range(epoch):
    lr_scheduler.step()
    print(f"epcoh{t + 1}\n-------")
    train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
    test_loss, test_acc = test(test_dataloader, model, loss_fn)

    loss_train.append(train_loss)
    acc_train.append(train_acc)

    loss_test.append(test_loss)
    acc_test.append(test_acc)

    if test_acc > min_acc:
        folder = 'save_model'
        if not os.path.exists(folder):
            os.mkdir("../save_model")
        min_acc = test_acc
        print(f"save model {t + 1}轮")
        # torch.save(model.state_dict(),path)只保存模型参数，推荐使用的方式
        torch.save(model.state_dict(), '../save_model/alexnet-best-model.pth')
    if t == epoch - 1:
        torch.save(model.state_dict(), '../save_model/alexnet-best-model.pth')
matplot_loss(loss_train, loss_test)
matplot_acc(acc_train, acc_test)
