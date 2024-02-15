import torch.nn as nn
from torch import Tensor

class CNN(nn.Module):
    def __init__(self, kernel_size:int=3, dropout_rate:float=0.1, num_classes:int=10, num_input_channels:int=3):
        super().__init__()
        self.featemb = CNNEmbedding(kernel_size, num_input_channels)
        self.fclayer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate), 
            nn.Linear(self.featemb.n_features, num_classes)
        )
 
    def forward(self, x:Tensor):
        # feature embedding
        out = self.featemb(x)
        # final layer
        out = self.fclayer(out) 
        return out
    
class CNNEmbedding(nn.Module):
    def __init__(self, kernel_size:int=3, num_input_channels:int=3, **kwargs):
        super().__init__()
        self.conv_layer = nn.Sequential( 
            # Conv Layer block 1
            nn.Conv2d(in_channels=num_input_channels, out_channels=16, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 2
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 3
            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=kernel_size, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=kernel_size, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128), 
        )
        self.n_features = 128
 
    def forward(self, x:Tensor):
        # conv layers
        x = self.conv_layer(x) 
        # flatten
        x = x.view(x.size(0), -1) 
        # fc layer
        x = self.fc_layer(x) 
        return x