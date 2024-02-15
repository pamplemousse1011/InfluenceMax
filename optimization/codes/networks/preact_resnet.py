## https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable  
import sys

def preact_cfg(depth):
    depth_lst = [18, 34, 50, 101, 152]
    assert (depth in depth_lst), "Error : Resnet depth should be either 18, 34, 50, 101, 152"
    cf_dict = {
        '18': (PreActBlock, [2,2,2,2]),
        '34': (PreActBlock, [3,4,6,3]),
        '50': (PreActBottleneck, [3,4,6,3]),
        '101':(PreActBottleneck, [3,4,23,3]),
        '152':(PreActBottleneck, [3,8,36,3]),
    }

    return cf_dict[str(depth)]


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out
    
class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out

class PreActResNet(nn.Module):
    def __init__(self, depth:int, dropout_rate:float, num_classes:int, num_input_channels:int=3):
        super().__init__()
        in_planes = 16
        
        self.featemb = PreActResNetEmbedding(in_planes, depth, num_input_channels)
        self.fclayer = nn.Sequential(
            nn.Dropout(p=dropout_rate), 
            nn.Linear(self.featemb.n_features, num_classes)
        )
     
    def forward(self, x:Tensor):
        # feature embedding
        out = self.featemb(x) 
        # final layer
        out = self.fclayer(out)  
        return out

class PreActResNetEmbedding(nn.Module):
    def __init__(self, in_planes:int=16, depth:int=18, num_input_channels:int=3, **kwargs):
        super().__init__()
        self.in_planes = in_planes
        block, num_blocks = preact_cfg(depth)

        self.conv1 = nn.Conv2d(num_input_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.n_features = 64*block.expansion

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x:Tensor):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.shape[-1])
        out = out.view(out.size(0), -1)  

        return out

if __name__ == '__main__':
    net = PreActResNet(50, 0., 10)
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())
 