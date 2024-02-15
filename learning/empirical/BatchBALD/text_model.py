from torch import nn as nn
from torch.nn import functional as F
from torch import Tensor, flatten

import mc_dropout

  
class BayesianNet(mc_dropout.BayesianModule):
    def __init__(self, num_classes, input_features, hidden_features):
        super().__init__(num_classes)

        self.fc1 = nn.Linear(input_features, hidden_features)
        self.fc1_drop = mc_dropout.MCDropout()
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc2_drop = mc_dropout.MCDropout()
        self.fc3 = nn.Linear(hidden_features, num_classes)

    def mc_forward_impl(self, input: Tensor):
        input = flatten(input, start_dim=1)
        input = F.relu(self.fc1_drop(self.fc1(input)))
        input = F.relu(self.fc2_drop(self.fc2(input)))
        input = self.fc3(input)
        input = F.log_softmax(input, dim=1)

        return input

class NormalNet(nn.Module):
    def __init__(self, num_classes, input_features, hidden_features):
        super().__init__()

        self.fc1 = nn.Linear(input_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, num_classes)

    def forward(self, input: Tensor):
        input = flatten(input, start_dim=1)
        input = F.relu(self.fc1(input))
        input = F.relu(self.fc2(input))
        input = self.fc3(input)
        input = F.log_softmax(input, dim=1)

        return input
