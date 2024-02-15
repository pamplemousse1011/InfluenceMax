import torch
from torch import Tensor
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
 
def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform(m.weight, gain=np.sqrt(2))
        nn.init.constant(m.bias, 0)

def cfg(depth):
    depth_lst = [11, 13, 16, 19]
    assert (depth in depth_lst), "Error : VGGnet depth should be either 11, 13, 16, 19"
    cf_dict = {
        '11': [
            64, 'mp',
            128, 'mp',
            256, 256, 'mp',
            512, 512, 'mp',
            512, 512, 'mp'],
        '13': [
            64, 64, 'mp',
            128, 128, 'mp',
            256, 256, 'mp',
            512, 512, 'mp',
            512, 512, 'mp'
            ],
        '16': [
            64, 64, 'mp',
            128, 128, 'mp',
            256, 256, 256, 'mp',
            512, 512, 512, 'mp',
            512, 512, 512, 'mp'
            ],
        '19': [
            64, 64, 'mp',
            128, 128, 'mp',
            256, 256, 256, 256, 'mp',
            512, 512, 512, 512, 'mp',
            512, 512, 512, 512, 'mp'
            ],
    }

    return cf_dict[str(depth)]

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

class VGG(nn.Module):
    def __init__(self, depth:int, dropout_rate:float, num_classes:int, num_input_channels:int=3):
        super().__init__()
        
        self.featemb = VGGEmbedding(depth, num_input_channels)
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

class VGGEmbedding(nn.Module):
    def __init__(self, depth:int, num_input_channels:int=3, **kwargs):
        super().__init__()
        self.num_input_channels=num_input_channels
        self.features = self._make_layers(cfg(depth))
        
        self.n_features = 512

    def forward(self, x:Tensor):
        out = self.features(x)
        out = out.view(out.size(0), -1) 

        return out

    def _make_layers(self, cfg):
        layers = [] 
        in_planes = self.num_input_channels
        for x in cfg:
            if x == 'mp':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [conv3x3(in_planes, x), 
                           nn.BatchNorm2d(x), 
                           nn.ReLU(inplace=True)]
                in_planes = x

        # After cfg convolution
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

if __name__ == "__main__":
    net = VGG(16, 10)
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size()) 