import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import torch
import torch.nn as nn

from typing import Callable

"""
Here we can train multiple nets at the same time.
At test/validation step, we only look at the first net
"""

class MakeModule(nn.Module):
    def __init__(self, modulelist: nn.ModuleList):
        super().__init__()
        self.nets = modulelist

    def forward(self, x):
        out = torch.stack([net(x) for net in self.nets], dim = 0)
        return out

class MultiNet(nn.Module):
    def __init__(self, net_factory: Callable[[], nn.Module], num_models):
        super().__init__()
        self.nets = nn.ModuleList(
            [net_factory() for _ in range(num_models)]
        )

    def forward(self, x):
        out = torch.stack([net(x) for net in self.nets], dim = 0)
        return out


class LitModel(pl.LightningModule):
    def __init__(self, net, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.net           = net     
        self.acc_class     = torchmetrics.Accuracy()
        
    # define the loss function
    def comp_loss(self, output, target, reduction='mean'):
        return F.cross_entropy(output, target,reduction=reduction)
    def comp_acc(self, output, target):
        return self.acc_class(F.softmax(output, dim=-1), target)

    def training_step(self, batch, batch_idx):
        """
        - indicator: MASK of size (batch_size, num_models)
        - x: data of size (batch_size, 28, 28)
        - y: targets of size (batch_size, )
        - yhat: predictions of size (num_models, batch_size, num_classes)
        """
        
        indicator, (x, y) = batch 
        y_hat = self.net(x) 
        num_models = len(y_hat)
        loss = torch.stack([
            self.comp_loss(y_hat[i][indicator[:,i]], y[indicator[:,i]])
            for i in range(num_models)
        ]).mean() 
        
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return {'loss':  loss}
    

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)
        loss = torch.stack([self.comp_loss(item, y) for item in y_hat]).mean()
        acc = torch.stack([self.comp_acc(item, y) for item in y_hat]).mean()
        
        return {'loss': loss, 'acc':acc}
    
    def validation_epoch_end(self, outputs):
        avg_acc  = torch.stack([x['acc'] for x in outputs]).mean()
        
        self.log('val_acc', avg_acc, prog_bar = True)

    def test_step(self, batch, batch_idx):
        x, y  = batch
        y_hat = self.net.nets[0](x) # batch_size x num_classes
        loss  = self.comp_loss(y_hat, y)
        acc   = self.comp_acc(y_hat, y)
        return {'acc': acc, 'loss': loss}

    def test_epoch_end(self, outputs):
        avg_acc  = torch.stack([x['acc'] for x in outputs]).mean()
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        metrics = {'acc': avg_acc, 'loss': avg_loss}
        self.log_dict(metrics)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        return {"optimizer": optimizer}


