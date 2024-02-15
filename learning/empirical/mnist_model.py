from torch import nn as nn
from torch.nn import functional as F
       
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1) # return torch.flatten(x, start_dim=1)
       
class NormalNet(nn.Module):
    def __init__(self, in_features:int, out_features: int):
        super().__init__()

        self.net = nn.Sequential(
            Flatten(),
            nn.Linear(in_features, 392),
            nn.ReLU(),
            nn.Linear(392, 128),
            nn.ReLU(),
            nn.Linear(128, out_features)
        )

    def forward(self, x):
        return self.net(x)

