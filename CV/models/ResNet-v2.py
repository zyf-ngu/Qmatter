import torch
import torch.nn as nn
import random
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self,change_channels,out_channels,stride=1,downsample=None):
        #change_channels=self
        super(BasicBlock,self).__init__()
        self.conv1=nn.Conv2d(in_channels=change_channels,out_channels=out_channels,kernel_size=3,stride=stride,padding=1)
        self.bn1=nn.BatchNorm2d(num_features=out_channels)
        self.relu=nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.downsample=downsample

    def forward(self,x):
        identity=x
        if self.downsample:
            identity=self.downsample(x)
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)

        x=self.conv2(x)
        x=self.bn2(x)

        print("identity",identity.size())
        print("x",x.size())

        x=x+identity
        x=self.relu(x)

        return x

class ResNet34(nn.Module):
    def __init__(self,block,blocks_num,num_classes):
        super(ResNet34,self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3)
        self.bn1=nn.BatchNorm2d(num_features=64)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1=self._make_layer(BasicBlock,blocks_num[0],64,64,stride=1)

        self.layer2=self._make_layer(BasicBlock,blocks_num[1],64,128,stride=2)

        self.layer3=self._make_layer(BasicBlock,blocks_num[2],128,256,stride=2)

        self.layer4=self._make_layer(BasicBlock,blocks_num[3],256,512,stride=2)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.flateen=nn.Flatten()
        self.fc=nn.Linear(512,num_classes)

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
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
    def _make_layer(self, block, block_num, change_channels, out_channels, stride):
        layers = []
        downsample=None
        if stride!=1:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=change_channels,out_channels=out_channels,kernel_size=1, stride=2),
                nn.BatchNorm2d(num_features=out_channels)
            )
        layers.append(block(change_channels, out_channels, stride, downsample))
        for i in range(1, block_num):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)




if __name__=="__main__":
    x=torch.randn([1,3,224,224])
    blocks_num=[3,4,6,3]
    model=ResNet34(BasicBlock,blocks_num,num_classes=7)
    y=model(x)
    print(y.size())
    print(model)

