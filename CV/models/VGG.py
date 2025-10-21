import torch
from torch import nn
import torch.nn.functional as F
import random

# 定义VGG网络模型
# VGG（子类）继承nn.Module（父类）
class VGG(nn.Module):
    # 子类继承中重新定义Module类的__init__()和forward()函数，这也是网络模型必须包含的两个函数
    #init 类的init函数里传需要预定义的实参数，类里的其它函数也可以使用，这里可以定义一些类的其它函数或变量
    # features：make_features(cfg: list)生成提取特征的网络结构，这里是将重复的网络层抽象出统一的结构，可以直接改变参数生成不同的网络层
    # num_classes：需要分类的类别个数
    # init_weights：是否对网络进行权重初始化
    def __init__(self,features,num_classes=1000,init_weights=False):
        # super：引入父类的初始化方法给子类进行初始化
        super(VGG,self).__init__()
        # 生成提取特征的网络结构,这里直接调用传来的参数features
        self.features=features
        # 生成分类的网络结构
        # Sequential：自定义顺序连接成模型，生成网络结构
        #网络层的一种组织形式，连续形，和前面直接按照顺序排列网络层相同，这里利用Sequential将网络层组织起来
        self.classifier=nn.Sequential(
            # 这里等价于前面Lenet和Alexnet介绍的init里初始化网络层，只不过这里没有将网络层赋给某个变量，而是统一保存后将其赋给classifier
            # Dropout：随机地将输入中50%的神经元激活设为0，即去掉了一些神经节点，防止过拟合
            nn.Dropout(p=0.5),
            nn.Linear(in_features=7*7*512,out_features=4096),
            # ReLU(inplace=True)：将tensor直接修改，不找变量做中间的传递，节省运算内存，不用多存储额外的变量
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096,out_features=4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
        # 如果为真，则对网络参数进行初始化
        if init_weights:
            self._initialize_weights()

    # 在init函数里还根据传入的参数判断是否网络参数初始化，这个是LeNet和AlexNet代码里未体现的，但是好的参数初始化可以提高网络效果，加快网络收敛
    # 网络参数初始化的方法前面已经介绍过，这里使用其中一种xaiver

    def _initialize_weights(self):
        # 初始化需要对每一个网络层的参数进行操作，所以利用继承nn.Module类中的一个方法:self.modules()遍历返回所有module
        for m in self.modules():
            # isinstance(object, type)：如果指定对象是指定类型，则isinstance()函数返回True
            # 如果是卷积层
            if isinstance(m, nn.Conv2d):
                # 首先是对权重参数进行初始化，uniform_(tensor, a=0, b=1)：服从~U(a,b)均匀分布，进行初始化
                nn.init.xavier_uniform_(m.weight)
                # 然后如果存在偏置参数，一般将其初始化为0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # 如果是全连接层
            elif isinstance(m, nn.Linear):
                # 权重参数正态分布初始化
                nn.init.xavier_uniform_(m.weight)
                # 偏置参数初始化为0
                nn.init.constant_(m.bias, 0)

    # 处理好网络层的定义和网络参数的初始化，便可以和前面那样利用forward()函数定义前向传播过程， 描述各层之间的连接关系
    def forward(self, x):
        # 将数据输入至提取特征的网络结构，N x 3 x 224 x 224，最笨的方法是按照之前学习的，将所有网络层一一列出来，但这样一方面太过冗杂，另一方面灵活性太差。
        x = self.features(x)
        # N x 512 x 7 x 7
        # 图像经过提取特征网络结构之后，得到一个7*7*512的特征矩阵，进行展平
        # Flatten()：将张量（多维数组）平坦化处理，神经网络中第0维表示的是batch_size，所以Flatten()默认从第二维开始平坦化
        x = torch.flatten(x, start_dim=1)
        # 将数据输入分类网络结构，N x 512*7*7
        x = self.classifier(x)
        return x

#init函数里直接利用传入的参数定义了特提取网络，这里要定义如何创建,
# 之所以单独用一个函数定义，是因为vgg有多种配置，需要根据配置创建不同的网络结构，而配置则是用列表逐一描述了网络层的类型和通道数
# 定义cfgs字典文件，每一个key-value对代表一个模型的配置文件，在模型实例化时，我们根据选用的模型名称key，将对应的值-配置列表作为参数传到函数里
# 如：VGG11代表A配置，也就是一个11层的网络， 数字代表卷积层中卷积核的个数，'M'代表池化层
# 通过函数make_features(cfg: list)生成提取特征网络结构
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
def make_features(cfg:list):
    #首先定义一个保存网络结构的空列表
    layers=[]
    #根据最初的输入图像通道数定义，一般是RGB 3通道
    in_channels=3
    #然后遍历配置表，根据遇到的情况（池化层还是卷积层）增加不同的网络层结构
    for v in cfg:
        # 如果列表的值是M字符，说明该层是最大池化层
        if v=="M":
            # 创建一个最大池化层，在VGG中所有的最大池化层的kernel_size=2，stride=2
            layers+=[nn.MaxPool2d(kernel_size=2,stride=2)]
        # 否则是卷积层
        else:
            # 在Vgg中，所有的卷积层的kernel_size=3,padding=1，stride=1
            conv2d=nn.Conv2d(in_channels, v, kernel_size=3, stride=1,padding=1)
            # 将卷积层和ReLU放入列表
            layers+=[conv2d,nn.ReLU(True)]
            #网络列表每加一层，本层输入通道数都要改成上层的输出通道数
            in_channels=v
    # 将列表通过非关键字参数的形式返回，*layers可以接收任意数量的参数
    return nn.Sequential(*layers)


#vgg实例化和LeNet,AlexNet有点不同，因为要先手动选择网络名称，以VGG16为例，定义如下
# **kwargs表示可变长度的字典变量，在调用VGG函数时传入的字典变量
def vgg(model_name="vgg16",**kwargs):
    # 如果model_name不在cfgs，序会抛出AssertionError错误，报错为参数内容“ ”
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg=cfgs[model_name]
    # 这个字典变量包含了分类的个数以及是否初始化权重的布尔变量
    model=VGG(make_features(cfg),**kwargs)
    return model

# 测试代码
        # 每个python模块（python文件）都包含内置的变量 __name__，当该模块被直接执行的时候，__name__ 等于文件名（包含后缀 .py ）
        # 如果该模块 import 到其他模块中，则该模块的 __name__ 等于模块名称（不包含后缀.py）
        # “__main__” 始终指当前执行模块的名称（包含后缀.py）
        # if确保只有单独运行该模块时，此表达式才成立，才可以进入此判断语法，执行其中的测试代码，反之不行
if __name__=="__main__":
    x=torch.randn([1,3,224,224])
    model=vgg("vgg16")
    model_name = "vgg16"
    #model = vgg(model_name=model_name, num_classes=7, init_weights=True)
    y=model(x)
    print(y.size())