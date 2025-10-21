import torch
import torch.nn as nn
import random
import torch.nn.functional as F

# 定义基础残差模块，基础残差模块的主要变化点就是不同模块之间的切换时的输入通道变化，其它的都可以复用，下采样也发生在模块的第一个子结构的第一个卷积层
class BasicBlock(nn.Module):
    expansion=1
    #change_channels：输入通道数。out_channels：输出通道数。
    # stride：默认为1，第一个卷积层的 stride才会变化。downsample：从 make_layer 中传入的 downsample 层。
    def __init__(self,change_channels,out_channels,stride=1,downsample=None):
        super(BasicBlock,self).__init__()
        #定义第 1 组 conv3x3 -> norm_layer -> relu，这里使用传入的 stride 和 change_channels。
        # （如果是 layer2 ，layer3 ，layer4 里的第一个 BasicBlock，那么 stride=2，这里会降采样和改变通道数）。
        self.conv1=nn.Conv2d(in_channels=change_channels,out_channels=out_channels,kernel_size=3,stride=stride,padding=1)
        self.bn1=nn.BatchNorm2d(num_features=out_channels)
        #注意，2 个卷积层都需要经过 relu 层，但它们使用的是同一个 relu 层。
        self.relu=nn.ReLU()
        #定义第 2 组 conv3x3 -> norm_layer -> relu，这里不使用传入的 stride （默认为 1），
        # 输入通道数和输出通道数使用out_channels，也就是不需要降采样和改变通道数。
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.downsample=downsample
# 输入数据分成两条路，一条路经过两个3 * 3卷积，另一条路直接短接，二者相加经过relu输出。
# 核心就是判断短接的路要不要downsample，下采样同时则增加通道数，这点在make_layer 里面通过stride是否！=1（即特征尺寸变化）或者输入输出通道数是否一致来判断
    def forward(self,x):
        # x 赋值给 identity，用于后面的 shortcut 连接。
        identity=x
        # x 经过第 1 组 conv3x3 -> norm_layer，得到 out。
        # 如果是 layer2 ，layer3 ，layer4 里的第一个 BasicBlock，那么 downsample 不为空，会经过 downsample 层，得到 identity。
        if self.downsample:
            identity=self.downsample(x)
#x 经过第 1 组 conv3x3 -> norm_layer -> relu，如果是 layer2 ，layer3 ，layer4 里的第一个 BasicBlock，那么 stride=2，第一个卷积层会降采样。
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)

        x=self.conv2(x)
        x=self.bn2(x)
        print("identity",identity.size())
        print("x",x.size())
#最后将 identity 和 out 相加，经过 relu ，得到输出。
        x=x+identity
        x=self.relu(x)
        return x
class BottleBlock(nn.Module):
    expansion=4
    #in_channel：输入通道数。out_channel：输出通道数。
    #stride：第一个卷积层的 stride。downsample：从 layer 中传入的 downsample 层。
    def __init__(self,in_channel,out_channel,stride=1,downsample=None):
        super(BottleBlock,self).__init__()
        #定义子结构里的1*1卷积层 conv1x1 -> norm_layer，注意这里的out_channel会降维。将上一子结构最后的通道数降下来，stride默认为1。
        self.conv1=nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1, stride=1)
        self.bn1=nn.BatchNorm2d(num_features=out_channel)
       #定义子结构里的3*3卷积层 conv3x3 -> norm_layer，这里使用传入的 stride，这里输出通道数不发生变化。
        # （如果是 layer2 ，layer3 ，layer4 里的第一个 Bottleneck，那么 stride=2，这里会降采样）。
        self.conv2=nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=3,stride=stride,padding=1)
        self.bn2=nn.BatchNorm2d(num_features=out_channel)
       #定义子结构里的第二个1*1卷积层，conv1x1 -> norm_layer，使用out_channel * self.expansion作为输出通道数升维,stride默认为1。
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(num_features=out_channel*self.expansion)
        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample


    def forward(self,x):
        identity=x
        if self.downsample:
            identity=self.downsample(x)
       #1x1 的卷积是为了降维，减少通道数
        x=self.conv1(x)
        x=self.bn1(x)
        x = self.relu(x)
        # 3x3 的卷积是为了改变图片大小，不改变通道数
        x=self.conv2(x)
        x=self.bn2(x)
        x = self.relu(x)
        # 1x1 的卷积是为了升维，增加通道数，增加到 out_channel * 4
        x=self.conv3(x)
        x=self.bn3(x)

        print("identity", identity.size())
        print("x", x.size())

        x=x+identity
        x = self.relu(x)
        return x
class ResNet(nn.Module):
    def __init__(self,block,blocks_num,num_classes):
        super(ResNet,self).__init__()
        #定义残差模块最开始的输入通道数为64
        self.change_channels=64
        #输入部分首先经过为一个size=7x7，stride为2的卷积处理，得到112*112*64的特征图，
        self.conv1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3)
        self.bn1=nn.BatchNorm2d(num_features=64)
        self.relu=nn.ReLU()
        # 然后经过size=3x3, stride=2的最大池化处理，一个224x224的输入图像就会变56x56*64大小的特征图，极大减少了存储所需大小。并将此特征图传到残差模块
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1=self._make_layer(block,blocks_num[0],64,stride=1)

        self.layer2=self._make_layer(block,blocks_num[1],128,stride=2)

        self.layer3=self._make_layer(block,blocks_num[2],256,stride=2)

        self.layer4=self._make_layer(block,blocks_num[3],512,stride=2)
       #通过全局自适应平滑池化，把所有的特征图拉成1*1，
        #网络的最后阶段不再是连续的三个全连接层，而是使用一个全局的平均pooling层和一个1000 类的包含softmax的全连接层。
        # 全局平均池化是在NIN网络中提出来的，简单来说就是最后卷积层输出的特征图尺寸是[Batch,Channel,Height,Width]，
        # 经过全局平均池化后，尺寸将变为[Batch,Channel,1,1]，也就是说，全局平均池化其实就是对每一个通道特征图所有像素（特征）值求平均值，得到一个1*1的特征图。
        # 对于res18来说，就是1x512x7x7 的输入数据拉成 1x512x1x1，然后接全连接层输出，输出节点个数与预测类别个数一致。
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.flateen=nn.Flatten()
        self.fc=nn.Linear(512*block.expansion,num_classes)

        # 遍历网络中的每一层
        # 继承nn.Module类中的一个方法:self.modules(), 他会返回该网络中的所有modules
        for m in self.modules():
            # isinstance(object, type)：如果指定对象是指定类型，则isinstance()函数返回True
            # 如果是卷积层
            if isinstance(m, nn.Conv2d):
                # kaiming正态分布初始化，使得Conv2d卷积层反向传播的输出的方差都为1
                # fan_in：权重是通过线性层（卷积或全连接）隐性确定
                # fan_out：通过创建随机矩阵显式创建权重
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)
        print("layer1")
        x=self.layer1(x)
        print("layer2")
        x=self.layer2(x)
        print("layer3")
        x=self.layer3(x)
        print("layer4")
        x=self.layer4(x)
        x=self.avgpool(x)
        x=self.flateen(x)
        x=self.fc(x)

        return x

    #block：每个layer里面使用的block，可以是 BasicBlock，BottleBlock。
    #block_num,一个整数，表示该层 layer 有多少个 block，根据给定的blocks_num里遍历的
    #out_channels输出的通道数
    #stride：第一个 block 的卷积层的 stride，默认为 1。注意，BasicBlock只有在每个 layer 的第一个子结构的卷积层使用该参数。
    #BottleBlock只有在每个 layer 的第一个子结构的第二个卷积层使用该参数。
    def _make_layer(self, block, block_num, out_channels, stride):
        layers = []
#判断 stride 是否为 1，输入通道和输出通道是否相等。如果这两个条件都不成立，那么表明需要建立一个 1 X 1 的卷积层将原始输入x下采样升维变成和输出尺寸通道一致，
        # 来改变通道数和改变图片大小。具体是建立 downsample 层，包括 1x1卷积层和norm_layer，1x1卷积层注意输出通道数和stride。
        downsample=None
        if stride!=1 or self.change_channels!=out_channels*block.expansion:
            #建立第一个 block，把 downsample 传给 block 作为降采样的层，并且 stride 也使用传入的 stride（stride=2）
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.change_channels,out_channels=out_channels*block.expansion,kernel_size=1, stride=stride),
                nn.BatchNorm2d(num_features=out_channels*block.expansion)
            )
        #第一个子结构需要单独传参创建，后面的子结构才可以遍历创建
        layers.append(block(self.change_channels, out_channels, stride, downsample))
        #改变输入通道数self.change_channels=out_channels*block.expansion，这个变量是整个类的全局变量
        #o在 BasicBlock 里，expansion=1，因此这一步不会改变通道数。在 BottleBlock 里，expansion=4，因此这一步会改变通道数。
        self.change_channels=out_channels*block.expansion
        #图片经过第一个 block后，就会改变通道数和图片大小。接下来 for 循环添加剩下的 block。
        # 从第 2 个 block 起，输入和输出通道数是相等的，因此就不用传入 downsample 和 stride（那么 block 的 stride 默认使用 1，
        for i in range(1, block_num):
            layers.append(block(self.change_channels, out_channels))

        return nn.Sequential(*layers)

# 测试代码
        # 每个python模块（python文件）都包含内置的变量 __name__，当该模块被直接执行的时候，__name__ 等于文件名（包含后缀 .py ）
        # 如果该模块 import 到其他模块中，则该模块的 __name__ 等于模块名称（不包含后缀.py）
        # “__main__” 始终指当前执行模块的名称（包含后缀.py）
        # if确保只有单独运行该模块时，此表达式才成立，才可以进入此判断语法，执行其中的测试代码，反之不行
if __name__=="__main__":
    x=torch.randn([1,3,224,224])
   # [3, 4, 6, 3] 等则代表了bolck的重复堆叠次数
    # blocks_num=[3,4,6,3]
    # model=ResNet(BasicBlock,blocks_num,num_classes=7)
    blocks_num = [3, 4, 6, 3]
    model = ResNet(BottleBlock, blocks_num, num_classes=7)
    y=model(x)
    print(y.size())
    print(model)

