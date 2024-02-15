from torch import nn as nn
from torch.nn import functional as F
from torch import Tensor, flatten
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1) 
       

class NormalNet(nn.Module):
    def __init__(self, in_features:int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 392)
        self.fc2 = nn.Linear(392, 128)
        self.fc3 = nn.Linear(128, out_features)

    def forward(self, input: Tensor):
        input = flatten(input, start_dim=1)
        input = F.relu(self.fc1(input))
        input = F.relu(self.fc2(input))
        input = self.fc3(input)
        return input
