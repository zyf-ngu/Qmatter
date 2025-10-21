import torch
from torch import nn
import torch.nn.functional as F
import random


# 定义AlexNet网络模型
# AlexNet（子类）继承nn.Module（父类）
class AlexNet(nn.Module):
    # 子类继承中重新定义Module类的__init__()和forward()函数，这也是网络模型必须包含的两个函数
    # init()函数：进行初始化，申明模型中各层的定义
    def __init__(self):
        # super：引入父类的初始化方法给子类进行初始化
        # super(AlexNet, self).__init__()
        super().__init__()
        # 卷积层，输入大小为224*224，输出大小为55*55，输入通道为3，输出通道为96，卷积核尺寸为11，扩充边缘为2
        self.c1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        # 引入了ReLu激活函数
        self.Relu = nn.ReLU()
        # 最大池化层，输入为55×55×96，经Overlapping pooling(重叠池化)pool_size = 3,stride = 2后得到尺寸为27×27×96 的特征图
        self.s2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # Conv层： 输入尺寸为27×27×96，经256个5×5×96的filter卷积，padding=same得到尺寸27×27×256。
        self.c3 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        # 池化层： 输入为27×27×256，，经pool size = 3,stride = 2的重叠池化，得到尺寸为13×13×256的池化层。
        self.s4 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 输入为13×13×256，经384个3×3×256的filter卷积得到13×13×384的卷积层。
        self.c5 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        # 输入为13×13×384，经384组3×3×384的filter卷积得到13×13×384
        self.c6 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        # 输入为13×13×384，经256个3×3×384的filter卷积得到13×13×256
        self.c7 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        # 输入尺寸为13×13×256，经pool size = 3,stride = 2的重叠池化得到尺寸为6×6×256
        self.s8 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 展平flatten，通过展平得到6×6×256=9216个特征后与之后的全连接层相连
        self.flatten = nn.Flatten()
        # 第10~12层神经元个数分别为4096，4096,1000。其中前两层在使用relu后还使用了Dropout对神经元随机失活，
        # 最后一层全连接层用softmax输出1000个分类（分类数量根据具体应用的数量变化，比如数据集中有10个类别，则最后输出10）
        self.f9 = nn.Linear(in_features=6 * 6 * 256, out_features=4096)
        self.f10 = nn.Linear(in_features=4096, out_features=4096)
        self.output = nn.Linear(in_features=4096, out_features=7)

    # forward()：定义前向传播过程,描述了各层之间的连接关系
    def forward(self, x):
        x = self.Relu(self.c1(x))
        x = self.s2(x)
        x = self.Relu(self.c3(x))
        x = self.s4(x)
        x = self.Relu(self.c5(x))
        x = self.Relu(self.c6(x))
        x = self.Relu(self.c7(x))
        x = self.s8(x)
        x = self.flatten(x)
        x = F.dropout(x, p=0.5)
        x = self.f9(x)
        x = F.dropout(x, p=0.5)
        x = self.f10(x)
        x = F.dropout(x, p=0.5)
        output = self.output(x)
        return output

        # 测试代码
        # 每个python模块（python文件）都包含内置的变量 __name__，当该模块被直接执行的时候，__name__ 等于文件名（包含后缀 .py ）
        # 如果该模块 import 到其他模块中，则该模块的 __name__ 等于模块名称（不包含后缀.py）
        # “__main__” 始终指当前执行模块的名称（包含后缀.py）
        # if确保只有单独运行该模块时，此表达式才成立，才可以进入此判断语法，执行其中的测试代码，反之不行


if __name__ == "__main__":
    # rand:返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数，此处为四维张量
    x = torch.randn([1, 3, 227, 227])
    # # 模型实例化
    model = AlexNet()
    print("模型各层参数信息：")
    for name, param in model.named_parameters():
        print(f"层名称: {name}")
        print(f"参数形状: {param.shape}")
        print(f"参数数量: {param.numel()}")  # numel() 计算参数总数
        print("------------------------")
    y = model(x)
    print(y.size())
