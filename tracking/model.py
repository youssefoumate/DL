""" Model Architecture Module
"""

from tinygrad.tensor import Tensor
import tinygrad
import torch
import torch.nn as nn

class tinyMDNet():
    def __init__(self, K=1):
        super(tinyMDNet, self).__init__()
        self.K = K
        self.conv1 = tinygrad.nn.Conv2d(3, 96, kernel_size=7, stride=2)
        self.bn1 = tinygrad.nn.BatchNorm2D(96)
        self.conv2 = tinygrad.nn.Conv2d(96, 256, kernel_size=5, stride=2)
        self.bn2 = tinygrad.nn.BatchNorm2D(256)
        self.conv3 = tinygrad.nn.Conv2d(256, 512, kernel_size=3, stride=2)
        self.linear1 = tinygrad.nn.Linear(512*4*4, 512)
        self.linear2 = tinygrad.nn.Linear(512, 512)
        self.branches = [tinygrad.nn.Linear(512, 2) for _ in range(K)]
    
    def forward(self, x, k=0):
        x = self.conv1(x)
        x = x.relu()
        x = self.bn1(x)
        x = x.max_pool2d(kernel_size=(3,3))
        x = self.conv2(x)
        x = self.bn2(x)
        x = x.max_pool2d(kernel_size=(2,2))
        x = x.relu()
        x = self.conv3(x)
        x = x.relu()
        x = self.linear1(x.reshape(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]))
        x = x.dropout(0.5)
        x = self.linear2(x)
        x = x.dropout(0.5)
        x = self.branches[k](x)
        return x

    def __call__(self, x):
        return self.forward(x)
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)

class ResNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        resblock = ResBlock
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )

        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False)
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class BboxClassifier(nn.Module):
    def __init__(self, outputs=1):
        super().__init__()
        self.backbone = ResNet()
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(512, outputs)
    def forward(self, x):
        #x = self.gap(x)
        x = torch.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    mdnet = tinyMDNet()
    resnet = BboxClassifier()
    x = Tensor.ones(1,3,256,256)
    out = mdnet(x)
    x = torch.rand(1,3,512,512)
    out_res = resnet(x)
    print(out.shape, out_res.size())
