import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform(m.weight, gain=np.sqrt(2))
        nn.init.constant(m.bias, 0)

class LeNet(nn.Module):
    def __init__(self, dropout_rate:float, num_classes:int, num_input_channels:int=3):
        super().__init__()
        self.featemb = LeNetEmbedding(num_input_channels)
        self.fclayer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.featemb.n_features, num_classes)
        )

    def forward(self, x):
        # feature embedding
        out = self.featemb(x)
        # final layer
        out = self.fclayer(out)

        return(out)
    
class LeNetEmbedding(nn.Module):
    def __init__(self, num_input_channels:int=3, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(num_input_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)  
        
        self.n_features = 84

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, out.shape[-1])
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, out.shape[-1])
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return(out)