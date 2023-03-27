""" Model Architecture Module
"""

from tinygrad.tensor import Tensor
import tinygrad.nn as nn
import torch

class tinyMDNet():
    def __init__(self, K=1):
        super(tinyMDNet, self).__init__()
        self.K = K
        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=2)
        self.bn1 = nn.BatchNorm2D(96)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2D(256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=2)
        self.linear1 = nn.Linear(512*4*4, 512)
        self.linear2 = nn.Linear(512, 512)
        self.branches = [nn.Linear(512, 2) for _ in range(K)]
    
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

class ResNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        self.model.fc = torch.nn.Linear(512, 1)
    def __call__(self, x):
        output = self.model(x)
        return output

if __name__ == "__main__":
    mdnet = tinyMDNet()
    resnet = ResNet()
    x = Tensor.ones(1,3,256,256)
    out = mdnet(x)
    x = torch.rand(1,3,255,255)
    out_res = resnet(x)

    print(out.shape, out_res.size())
