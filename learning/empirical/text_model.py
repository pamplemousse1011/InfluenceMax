
from torch import nn as nn
from torch.nn import functional as F
from torch import Tensor 


class NormalNet(nn.Module):
    def __init__(self, in_features:int, out_features: int, hidden_features: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, out_features)
    
    def forward(self, input: Tensor):
        input = F.relu(self.fc1(input))
        input = F.relu(self.fc2(input))
        input = self.fc3(input)
        return input
