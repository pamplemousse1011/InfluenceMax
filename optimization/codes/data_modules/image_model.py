 

import torch.optim as optim
import torch.nn as nn
import torch 
from torch import Tensor 

import lightning.pytorch as pl

from codes.data_modules.model_utils import get_imagemodel_hyperparameters, get_imgemb_module
 
class ImageModelModule(pl.LightningModule):
    def __init__(
        self, 
        net_type:int,
        num_classes:int,
        rgb:bool=False,
        dropout_rate:float=0.,
        learning_rate:float=0.0001, 
        weight_decay:float=0.01,
        momentum:float=0.01,
        max_epochs:int=1000, 
        save_logpath:str=None
    ):
        super().__init__()   
        self.save_hyperparameters()
        self.dropout_rate  = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay  = weight_decay 
        self.momentum      = momentum
        self.max_epochs    = max_epochs 
        self.save_logpath  = save_logpath

        self.train_step_numsamp = []
        self.train_step_sumloss = []
        self.train_epoch_outputs = []
        # self.val_step_numsamp = []
        # self.val_step_sumloss = []
        # self.val_epoch_outputs = []
        # self.test_loss  = [] 

        
        ## Get image model
        net_name, depth, width_factor = get_imagemodel_hyperparameters(net_type)
        self.featemb = get_imgemb_module(net_name)(
            depth=depth, 
            num_input_channels=3 if rgb else 1,
            width_factor=width_factor
        )
        self.fclayer = nn.Sequential(
            nn.Dropout(p=dropout_rate), 
            nn.Linear(self.featemb.n_features, num_classes)
        )
    
    def forward(self, x:Tensor) -> Tensor: 
        # feature embedding
        out = self.featemb(x)
        # final layer
        out = self.fclayer(out)
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch 
        # get predictions
        logits = self(x)
        # compute loss
        loss = nn.functional.cross_entropy(logits, y)

        # logging to TensorBoard by default
        self.log('loss', loss, on_step=False, on_epoch=True)

        if self.save_logpath is not None:
            self.train_step_sumloss.append(loss*x.shape[0])
            self.train_step_numsamp.append(x.shape[0])
        return {'loss':loss}
    
    def on_train_epoch_end(self): 
        if self.save_logpath is not None:
            avg_loss = torch.stack(self.train_step_sumloss).sum(0) / sum(self.train_step_numsamp)
            self.train_epoch_outputs.append((self.current_epoch, avg_loss))

            # free memory
            self.train_step_sumloss.clear()  
            self.train_step_numsamp.clear()  

    def on_train_end(self):
        if self.save_logpath is not None:
            num_epochs = len(self.train_epoch_outputs)

            with open(self.save_logpath, 'w') as f:
                for i in range(num_epochs):
                    epoch, train_loss = self.train_epoch_outputs[i]
                    f.write(f"{epoch}, {train_loss}\n")
                    # val_loss = self.val_epoch_outputs[i]
                    # f.write(f"{epoch}, {train_loss}, {val_loss}\n")

    # def validation_step(self, batch, batch_idx):
    #     """
    #     model.eval() and torch.no_grad() are called automatically for validation
    #     """
    #     x, y = batch 
    #     # get predictions
    #     logits = self(x)
    #     # compute loss
    #     loss = nn.functional.cross_entropy(logits, y)
    #     # Logging to TensorBoard by default
    #     self.log('val_loss', loss, on_step=False, on_epoch=True)
    #     self.val_step_sumloss.append(loss*x.shape[0])
    #     self.val_step_numsamp.append(x.shape[0])  
    #     return {'val_loss':loss}

    # def on_validation_epoch_end(self):
    #     avg_loss = torch.stack(self.val_step_sumloss).sum(0) / sum(self.val_step_numsamp) 
    #     self.val_epoch_outputs.append(avg_loss)
    #     # free memory
    #     self.val_step_sumloss.clear()  
    #     self.val_step_numsamp.clear()   
    
    # def test_step(self, batch, batch_idx):
    #     """ 
    #     model.eval() and torch.no_grad() are called automatically for validation
    #     - x: data of size (batch_size, 28, 28)
    #     - y: targets of size (batch_size, )
    #     - yhat: predictions of size (num_models, batch_size, num_classes)
    #     """
    #     x, y = batch 
    #     logits = self(x)
    #     test_loss = nn.functional.cross_entropy(logits, y, reduction="none")
        
    #     self.test_loss.append(test_loss)
    #     return {'test_loss': test_loss.mean()}
    
    # def on_test_epoch_end(self):
    #     loss = torch.concatenate(self.test_loss)
    #     self.log("test_loss", loss.mean())
    #     # free memory
    #     self.test_loss.clear()  
    
    def on_test_epoch_start(self):
        self.base_x_embedding = [[],[]]

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch 
        self.base_x_embedding[dataloader_idx].append(self.featemb(x))
    
    def on_test_epoch_end(self):
        for idx in range(len(self.base_x_embedding)):
            self.base_x_embedding[idx] = torch.concatenate(self.base_x_embedding[idx], dim=0) 

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch 
        logits = self(x)
        return nn.functional.cross_entropy(logits, y, reduction="none")

    def configure_optimizers(self): 
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.max_epochs, 
            eta_min=1e-5,  
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler} 

