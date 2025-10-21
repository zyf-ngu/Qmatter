import torch
import torch.nn as nn
import random
import torch.nn.functional as F

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18,self).__init__()
        self.c1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu=nn.ReLU()
        self.s2=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.c21=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=64),
          #  nn.ReLU()
        )
        self.c22 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=64),
         #   nn.ReLU()
        )
        self.downsample1=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=1, stride=2),
            nn.BatchNorm2d(num_features=128)
        )
        self.c31 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=128),
          #  nn.ReLU()
        )
        self.c32 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2),
            nn.BatchNorm2d(num_features=256)
        )
        self.c41 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2,padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )
        self.c42 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2),
            nn.BatchNorm2d(num_features=512)
        )
        self.c51 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2,padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU()
        )
        self.c52 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU()
        )
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512,7)
    def forward(self,x):
        x=self.c1(x)
        print("11", x.size())
        x=self.bn1(x)
        print("12", x.size())
        x=self.relu(x)
        print("13", x.size())

        identity=x
        print("21",identity.size())
        x=self.c21(x)
        print("21", x.size())
        x=x+identity

        identity=x
        x=self.c22(x)
        x=identity+x

        identity=self.downsample1(x)
        print("31", identity.size())
        x=self.c31(x)
        print("31", x.size())
        x=identity+x
        x=self.relu(x)

        identity=x
        print("32", identity.size())
        x=self.c32(x)
        print("32", x.size())
        x=identity+x

        identity = self.downsample2(x)
        x = self.c41(x)
        x = identity + x

        identity = x
        x = self.c42(x)
        x = identity + x

        identity = self.downsample3(x)
        x = self.c51(x)
        x = identity + x

        identity = x
        x = self.c52(x)
        x = identity + x

        x=self.avgpool(x)
        x = torch.flatten(x, 1)
        x=self.fc(x)

        return x

if __name__=="__main__":
    x=torch.randn([1,3,224,224])
    model=ResNet18()
    y=model(x)
    print(y.size())
    print(model)

