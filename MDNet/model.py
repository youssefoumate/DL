""" Model Architecture Module
"""
"""
org implementation in torch
class MDNet(nn.Module):
    def __init__(self, model_path=None, K=1):
        super(MDNet, self).__init__()
        self.K = K
        self.layers = nn.Sequential(OrderedDict([
                ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                        nn.ReLU(inplace=True),
                                        nn.LocalResponseNorm(2),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                        nn.ReLU(inplace=True),
                                        nn.LocalResponseNorm(2),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                        nn.ReLU(inplace=True))),
                ('fc4',   nn.Sequential(nn.Linear(512 * 3 * 3, 512),
                                        nn.ReLU(inplace=True))),
                ('fc5',   nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(512, 512),
                                        nn.ReLU(inplace=True)))]))

        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),
                                                     nn.Linear(512, 2)) for _ in range(K)])
"""
from tinygrad.tensor import Tensor
import tinygrad.nn as nn

class tinyMDNet():
    def __init__(self, K=1):
        super(tinyMDNet, self).__init__()
        self.K = K
        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=2)
        #self.linear1 = nn.Linear(512*3*3, 512)
        #self.linear2 = nn.Linear(512, 512)
        #self.branches = [nn.Linear(512, 2) for _ in range(K)]
    
    def forward(self, x):
        x = self.conv1(x)
        # add relu, norm and max pool
        x = self.conv2(x)
        # addd relu, norm and max pool
        x = self.conv3(x)
        # addd relu, norm and max pool
        #x = self.linear1(x.reshape())
        return x

    def __call__(self, x):
        return self.forward(x)

if __name__ == "__main__":
    model = tinyMDNet()
    x = Tensor.ones(1,3,256,256)
    out = model(x)
    print(out.shape)

