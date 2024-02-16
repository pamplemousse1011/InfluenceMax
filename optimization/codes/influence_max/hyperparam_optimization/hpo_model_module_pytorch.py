from typing import Callable, Tuple

import torch.optim as optim
import torch.nn.functional as F
import lightning.pytorch as pl
import torch
import torch.nn as nn  
from torch import Tensor, vmap

from collections import OrderedDict

from codes.utils import vectorize

class LatentEmbedding(object):
    """
    Process params into embeddings
    - Do nothing
    - Rescale all the input domain to be [0, 1]
    - Transform with adially symmetric fields
    
    Arguments
    ----------
    search_domain : numpy array (d,2)
    n_rad : int
        number of centers on each dimension 
    x : torch tensor (..., d) 
        n number of d-dimensional features to be transformed 
    
    Returns
    ----------
    x : torch tensor (..., nrad**d)
        transformed features
    """
    def __init__(
        self, 
        search_domain: Tensor,
        method       : str="none",
        n_rad        : int=50, 
        dtype        : torch.dtype=torch.float32,  
    ):   
        self.search_domain = search_domain 
        self.n_rad = n_rad # nrad for each dimension  
        self.dtype = dtype
        self.d = search_domain.shape[0]
        
        self.n_features = self.d 
        self.method = method
        if self.method == "rbf":
            # compute the parameters for radial basis function transform
            self._get_rbf_params() 
        elif self.method == "zon":
            # compute the parameters for standardization
            self._get_zon_params()
        elif self.method == "none":
            self.mu = 0.
            self.gamma = 1.
        else:
            raise ValueError(f"Transformer method {self.method} is not implemented.")  
            
        self._TRANSFORMER = {
            "none": lambda x: x,
            "zon" : self._zon,
            "rbf" : self._rbf  
        }
        
    def _get_zon_params(self): 
        """
        For x in range [a, b], we standardize x by 
        (x - a) / (b - a)
        """
        self.mu = self.search_domain[:,0]
        self.gamma = self.search_domain[:,1]-self.search_domain[:,0]  
    
    def _zon(self, x:Tensor) -> Tensor:
        """Max-Min standardization

        Arguments:
        ===================
        x: torch tensor (...,d)
           Features to be standardized in each dimension
        m: torch tensor (d,)
           Min value in each dimension 
        s: torch tensor (d,)
           Range in each dimension

        Returns:
        ===================
        x: torch tensor (...,d)
           Standardized features
        """
        return (x - self.mu.to(x.device)) / self.gamma.to(x.device)
    
    def _get_rbf_params(self):
        """
        here we assume all have been standardized 
        and x values are all within [0,1]
        
        Arguments:
        ===================
        prodsum: bool=False
            Whether to use product sum formulation; 
            otherwise, just the union sum formulation
            If used, n_rad**d centers in total; if not, n_rad*d.

        Returns:
        ===================
        mu: (n_rad, d) 
        gamma: float
        """  
         
        gamma = torch.stack([(self.n_rad/(xmax - xmin))**2
                            for (xmin, xmax) in self.search_domain], dim=0)
        mu = torch.stack([torch.linspace(start=xmin+(xmax-xmin)/(10*self.n_rad), 
                                        end=xmax-(xmax-xmin)/(10*self.n_rad), 
                                        steps=self.n_rad)
                        for (xmin, xmax) in self.search_domain], dim=-1)
    
        self.mu    = mu.to(self.dtype)
        self.gamma = gamma.to(self.dtype)

        self.n_features = self.n_rad ** self.d
         
    def _map_rbf_func(self, x:Tensor, mu:Tensor, gamma:Tensor) -> Tensor:
        """ 
        param x: (...,d)
            Feature value
        param mu: (n_rad,d)
            Radial basis centers
        param gamma (d,)
        returns gamma * ||x - mu||^2: (n, d, n_rad) 
        """
        return gamma * (x.unsqueeze(-1) - mu.unsqueeze(0)) ** 2 # (n, d, n_rad) 
    
    def _rbf(self, x:Tensor) -> Tensor:
        """Radial basis function transform

        Arguments:
        ===================
        x: (...,d)
            Feature value to be transformed
        mu: (n_rad,d)
            Radial basis centers
        gamma: (d,)

        Returns:
        ===================
        x: (..., n_rad**d)  
           Transformed features 
        """ 
        
        self.mu    = self.mu.to(x.device)
        self.gamma = self.gamma.to(x.device)

        out = vmap(self._map_rbf_func, in_dims=-1, out_dims=1)(x, self.mu, self.gamma) # (n, d, n_rad)
        out = vmap(lambda y: torch.cartesian_prod(*y).sum(-1), in_dims=0)(out) # (n, n_rad**d)
         
        if x.dim() == 1:
            out = out.squeeze(0)
        return torch.exp(-out)           
        
    def __call__(self, x:Tensor) -> Tensor:
        """
        Input x : torch tensor (..., d)
            -- n number of d-dimensional features to be transformed 
        Output x : torch tensor (..., n_features)
            -- n_features depends on which transform method is used
        """ 
        return self._TRANSFORMER[self.method](x) 

class StoBlock(nn.Module):
    """A stochastic layer.

    Args:
        in_dim (int): input dimension 
        out_dim (int): output dimension
        noise_dim (int, optional): noise dimension. Defaults to 100.
    """
    def __init__(self, in_dim, out_dim, no_batch_norm:bool=False, noise_dim:int=100, noise_std:float=1.):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.noise_dim = noise_dim
        self.noise_std = noise_std
        if not no_batch_norm: 
            layer = [
                nn.Linear(in_dim + noise_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.SiLU(inplace=True),
            ]
        else:
            layer = [
                nn.Linear(in_dim + noise_dim, out_dim), 
                nn.SiLU(inplace=True),
            ]
        self.layer = nn.Sequential(*layer)
    
    def forward(self, x: Tensor):
        if self.noise_dim > 0:
            eps = torch.normal(0, self.noise_std, size=(x.size(0), self.noise_dim), device=x.device) 
            x = torch.cat([x, eps], dim=-1)
        return self.layer(x)

class StoModel(pl.LightningModule):
    def __init__(
        self, 
        *n_hidden,
        base_x_embedding_fn:nn.Module=None,
        base_x_embedding_dim:int=512,
        n_model:int=1, 
        no_batch_norm:bool=False,
        n_noise:int=100,
        noise_std:float=1.,
        search_domain:Tensor=None,
        trans_method:str="none",
        trans_rbf_nrad:int=50,
        use_double:bool=False, 
        learning_rate:float=0.001, 
        weight_decay:float=0.01,
        gamma:float=0.1,  
        dropout_rate:float=0,
        use_pretrained_featemb:bool=False,
        disable_base_x_embedding_training:bool=False,
        **kwargs
    ):
        super().__init__()   
        if use_double:
            dtype = torch.float64
        else: 
            dtype = torch.float32

        ## Params embeddings (no trainable parameters)
        self.latent_embedding_fn = LatentEmbedding(
            search_domain, 
            trans_method, 
            trans_rbf_nrad,
            dtype
        )
        
        self.use_pretrained_featemb = use_pretrained_featemb
        self.disable_base_x_embedding_training = disable_base_x_embedding_training
        base_x_embedding_dim = base_x_embedding_dim
        ## Base data feature embeddings (containing trainable model parameters if True)
        if self.disable_base_x_embedding_training:
            print(f"disable_base_x_embedding_training={str(self.disable_base_x_embedding_training)[0]}")
            assert self.use_pretrained_featemb, "use_pretrained_featemb is not TRUE!"
        else:
            if self.use_pretrained_featemb:
                self.base_x_embedding_fn = nn.Linear(base_x_embedding_dim, base_x_embedding_dim)
                print(f"train base_x_embedding with use_pretrained_featemb=T")
            else:
                self.base_x_embedding_fn = base_x_embedding_fn
                print(f"train base_x_embedding from scratch (use_pretrained_featemb=F)")
        
        ## Create the model 
        n_in = self.latent_embedding_fn.n_features + base_x_embedding_dim
        self.dropout_rate = dropout_rate

        self.nets = nn.ModuleList([
            self.create_net(dtype, n_in, n_noise, noise_std, no_batch_norm, *n_hidden) 
            for _ in range(n_model)
        ]) 
        self.n_model   = n_model
        self.n_noise   = n_noise 
        self.noise_std = noise_std

        ## Denote other hyperparemeters used for training
        self.learning_rate = learning_rate 
        self.weight_decay  = weight_decay
        self.gamma         = gamma 
        self.milestones    = [1000]
    
    def create_net(
            self, 
            dtype: torch.dtype, 
            n_in:int, 
            n_noise:int, 
            noise_std:float, 
            no_batch_norm:bool, 
            *n_hidden
        ):
        Layers = []
        Layers.append(StoBlock(in_dim=n_in, out_dim=n_hidden[0], no_batch_norm=no_batch_norm, noise_dim=n_noise, noise_std=noise_std))        
        for i in range(1, len(n_hidden)):    
            Layers.append(StoBlock(in_dim=n_hidden[i-1], out_dim=n_hidden[i], no_batch_norm=no_batch_norm, noise_dim=n_noise, noise_std=noise_std)) 
        Layers.append(nn.Linear(in_features=n_hidden[-1], out_features=1))
        return nn.Sequential(*Layers).to(dtype) 
    
    def forward(self, base_x: Tensor, x:Tensor) -> Tensor: 
        """
        INPUT:
        ###################
        base_x : images of shape (..., C, H, W) or embedding of shape (..., d_base)
        x      : hyperparameters of shape (..., d)  
        
        RETURN:
        ###################
        y_hat  : (..., outdim=1)
        """
        if self.disable_base_x_embedding_training: 
            base_x_embedding = base_x.reshape(-1, *base_x.shape[-1:])

        else:
            if self.use_pretrained_featemb:
                # Get base x embedding: (b, d_1)
                base_x_embedding = self.base_x_embedding_fn(
                    base_x.reshape(-1, *base_x.shape[-1:])
                ) 
            else:
                # Reshape (b, C, H, W)
                base_x_re = base_x.reshape(-1, *base_x.shape[-3:])
                # Get base x embedding: (b, d_1)
                base_x_embedding = self.base_x_embedding_fn(base_x_re)                  

        # Reshape (b, d)
        x_re = x.reshape(-1, *x.shape[-1:]) 
        
        # Get x embeddings: (b, d_2)
        latent_embedding = self.latent_embedding_fn(x_re)                       

        # Concatenate the embeddings: (b, d_1 + d_2)
        emb = torch.concatenate([base_x_embedding, latent_embedding], dim=-1)

        out = torch.stack([net(emb) for net in self.nets], dim = 0)
        return out # (n_model, b, out_dim=1) 
         
    def training_step(self, batch:Tuple[Tensor, ...], batch_idx):
        base_x, x, y = batch

        if self.n_noise > 0:
            ## Model prediction
            y_hat_1 = self(base_x, x)             # (n_model, b, out_dim=1) 
            y_hat_2 = self(base_x, x)             # (n_model, b, out_dim=1) 
            
            ## Compute loss 
            loss = energy_loss_two_sample(
                x_true=y.unsqueeze(-1),           # (n_model, b, out_dim=1)
                x_perb_1=y_hat_1,                 # (n_model, b, out_dim=1)
                x_perb_2=y_hat_2,                 # (n_model, b, out_dim=1)
                multichannel=True, 
                reduction="mean"
            )                                     # (n_model, )  

            loss = loss.mean()                    # (,) 
                
        else:
            ## Model prediction
            y_hat = self(base_x, x).squeeze(-1)                 # (n_model, b) 
            ## Compute loss
            loss = F.mse_loss(
                input     = y_hat,                              # (n_model, b) 
                target    = torch.tile(y, (self.n_model, 1)),   # (n_model, b) 
                reduction = 'mean')                             # (,)
            
        ## Logging to TensorBoard by default
        self.log('loss', loss)
        return {'loss':  loss}
    
    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self.test_output_list = []
        self.test_batchsize_list = []
        return
    
    def test_step(self, batch:Tuple[Tensor, ...], batch_idx):
        base_x, x, y = batch
        if self.n_noise > 0:
            ## Model prediction
            y_hat_1 = self(base_x, x)             # (n_model, b, out_dim=1) 
            y_hat_2 = self(base_x, x)             # (n_model, b, out_dim=1) 
            
            ## Compute loss 
            loss = energy_loss_two_sample(
                x_true=y.unsqueeze(-1),           # (n_model, b, out_dim=1)
                x_perb_1=y_hat_1,                 # (n_model, b, out_dim=1)
                x_perb_2=y_hat_2,                 # (n_model, b, out_dim=1)
                multichannel=True, 
                reduction='sum'                   # (n_model, )   
            )            

        else:
            ## Model prediction
            y_hat = self(base_x, x).squeeze(-1)   # (n_model, b) 
            ## Compute loss
            loss = F.mse_loss(
                input     = y_hat,                             # (n_model, b) 
                target    = torch.tile(y, (self.n_model, 1)),  # (n_model, b) 
                reduction = 'none')                            # (n_model, b)
              
            loss = loss.sum(-1)                   # (n_model, )
            
        self.test_output_list.append(loss)
        self.test_batchsize_list.append(batch[0].shape[0])
    
    def on_test_epoch_end(self):
        n_train  = sum(self.test_batchsize_list) 
        losses   = torch.stack(self.test_output_list, dim=0).sum(dim=0) / n_train # (n_model,)
        for i in range(self.n_model):
            self.log('_'.join(['MLP', str(i)]), losses[i])

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.nets.parameters() if self.disable_base_x_embedding_training else self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        return [optimizer]#, [scheduler]
      
class RntBlock(nn.Module):
    """A ResNet block.

    Args:
        in_dim (int): input dimension 
        out_dim (int): output dimension
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim  
        self.layer = nn.Sequential(
            OrderedDict([
                # Define the first linear layer and batch normalization
                ('Dense_1', nn.Linear(in_dim, out_dim)),
                ('BatchNorm_1', nn.BatchNorm1d(out_dim)),
                ('SiLU', nn.SiLU(inplace=True)),
                # Define the second linear layer and batch normalization
                ('Dense_2', nn.Linear(out_dim, out_dim)),
                ('BatchNorm_2', nn.BatchNorm1d(out_dim))
            ])
        )
        
        # Residual connection
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x: Tensor):
        identity = self.residual(x)
        out = self.layer(x)
        out += identity
        out = F.silu(out)
        return out

class RntModel(pl.LightningModule):
    def __init__(
        self, 
        *n_hidden,
        base_x_embedding_fn:nn.Module=None,
        base_x_embedding_dim:int=512,
        n_model:int=1, 
        search_domain:Tensor=None,
        trans_method:str="none",
        trans_rbf_nrad:int=50,
        use_double:bool=False, 
        learning_rate:float=0.001, 
        weight_decay:float=0.01,
        gamma:float=0.1,  
        dropout_rate:float=0,
        use_pretrained_featemb:bool=False,
        disable_base_x_embedding_training:bool=False,
        save_logpath:str=None,
        **kwargs
    ):
        super().__init__()   
        if use_double:
            dtype = torch.float64
        else: 
            dtype = torch.float32

        ## Params embeddings (no trainable parameters)
        self.latent_embedding_fn = LatentEmbedding(
            search_domain, 
            trans_method, 
            trans_rbf_nrad,
            dtype
        )
        self.use_pretrained_featemb = use_pretrained_featemb
        self.disable_base_x_embedding_training = disable_base_x_embedding_training
        base_x_embedding_dim = base_x_embedding_dim
        ## Base data feature embeddings (containing trainable model parameters if True)
        if self.disable_base_x_embedding_training:
            print(f"disable_base_x_embedding_training={str(self.disable_base_x_embedding_training)[0]}")
            assert self.use_pretrained_featemb, "use_pretrained_featemb is not TRUE!"
        else:
            if self.use_pretrained_featemb:
                self.base_x_embedding_fn = nn.Linear(base_x_embedding_dim, base_x_embedding_dim)
                print(f"train base_x_embedding with use_pretrained_featemb=T")
            else:
                self.base_x_embedding_fn = base_x_embedding_fn
                print(f"train base_x_embedding from scratch (use_pretrained_featemb=F)")
                
        ## Create the model 
        n_in = self.latent_embedding_fn.n_features + base_x_embedding_dim
        self.dropout_rate = dropout_rate

        self.nets = nn.ModuleList([
            self.create_net(dtype, n_in, *n_hidden) 
            for _ in range(n_model)
        ]) 
        self.n_model   = n_model

        ## Denote other hyperparemeters used for training
        self.learning_rate = learning_rate 
        self.weight_decay  = weight_decay
        self.gamma         = gamma 
        self.milestones    = [1000]

        self.save_logpath = save_logpath
        self.train_step_numsamp = []
        self.train_step_sumloss = []
        self.train_epoch_outputs = []
        self.val_step_numsamp = []
        self.val_step_sumloss = []
        self.val_epoch_outputs = []
    
    def create_net(
            self, 
            dtype: torch.dtype, 
            n_in:int, 
            *n_hidden
        ):
        Layers = []
        Layers.append(RntBlock(in_dim=n_in, out_dim=n_hidden[0]))        
        for i in range(1, len(n_hidden)):    
            # Layers.append(nn.Dropout(0.5))
            Layers.append(RntBlock(in_dim=n_hidden[i-1], out_dim=n_hidden[i])) 
        # Layers.append(nn.Dropout(0.7))
        Layers.append(nn.Linear(in_features=n_hidden[-1], out_features=1))
        return nn.Sequential(*Layers).to(dtype) 
    
    def forward(self, base_x: Tensor, x:Tensor) -> Tensor: 
        """
        INPUT:
        ###################
        base_x : images of shape (..., C, H, W) or embedding of shape (..., d_base)
        x      : hyperparameters of shape (..., d)  
        
        RETURN:
        ###################
        y_hat  : (..., outdim=1)
        """
        if self.disable_base_x_embedding_training: 
            base_x_embedding = base_x.reshape(-1, *base_x.shape[-1:])

        else:
            if self.use_pretrained_featemb:
                # Get base x embedding: (b, d_1)
                base_x_embedding = self.base_x_embedding_fn(
                    base_x.reshape(-1, *base_x.shape[-1:])
                ) 
            else:
                # Reshape (b, C, H, W)
                base_x_re = base_x.reshape(-1, *base_x.shape[-3:])
                # Get base x embedding: (b, d_1)
                base_x_embedding = self.base_x_embedding_fn(base_x_re)                  

        # Reshape (b, d)
        x_re = x.reshape(-1, *x.shape[-1:]) 
        
        # Get x embeddings: (b, d_2)
        latent_embedding = self.latent_embedding_fn(x_re)                       

        # Concatenate the embeddings: (b, d_1 + d_2)
        emb = torch.concatenate([base_x_embedding, latent_embedding], dim=-1)

        out = torch.stack([net(emb) for net in self.nets], dim = 0)
        return out # (n_model, b, out_dim=1) 
         
    def training_step(self, batch:Tuple[Tensor, Tensor, Tensor], batch_idx:int):
        base_x, x, y = batch

        ## Model prediction
        y_hat = self(base_x, x).squeeze(-1)                 # (n_model, b) 
        ## Compute loss
        loss = F.mse_loss(
            input     = y_hat,                              # (n_model, b) 
            target    = torch.tile(y, (self.n_model, 1)),   # (n_model, b) 
            reduction = 'mean')                             # (,)
        
        ## Logging to TensorBoard by default
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.save_logpath is not None:
            self.train_step_sumloss.append(loss.detach()*x.shape[0])
            self.train_step_numsamp.append(x.shape[0])
        return {'loss':  loss}
    
    def on_train_epoch_end(self): 
        if self.save_logpath is not None:
            avg_loss = torch.stack(self.train_step_sumloss).sum(0) / sum(self.train_step_numsamp)
            self.train_epoch_outputs.append((self.current_epoch, avg_loss.item()))

            # free memory
            self.train_step_sumloss.clear()  
            self.train_step_numsamp.clear()  

    def on_train_end(self):
        if self.save_logpath is not None:
            train_num_epochs = len(self.train_epoch_outputs)
            with open(self.save_logpath[0], 'a') as f:
                f.write(f"epoch, train_loss\n")
                for i in range(train_num_epochs):
                    epoch, train_loss = self.train_epoch_outputs[i]
                    f.write(f"{epoch}, {train_loss}\n") 
         
            val_loss_name = [f'val_loss_{i}' for i in range(self.n_model)]
            val_num_epochs = len(self.val_epoch_outputs)
            with open(self.save_logpath[1], 'a') as f:
                f.write(f"epoch, {val_loss_name}\n")
                for i in range(val_num_epochs): 
                    epoch, val_loss = self.val_epoch_outputs[i]
                    f.write(f"{epoch}, {val_loss}\n")


    def validation_step(self, batch:Tuple[Tensor, Tensor, Tensor], batch_idx):
        base_x, x, y = batch
        ## Model prediction
        y_hat = self(base_x, x).squeeze(-1)   # (n_model, b) 
        ## Compute loss
        loss = F.mse_loss(
            input     = y_hat,                             # (n_model, b) 
            target    = torch.tile(y, (self.n_model, 1)),  # (n_model, b) 
            reduction = 'none')                            # (n_model, b)
            
        loss = loss.sum(-1)                   # (n_model, )
        
        self.log('val_loss', loss.mean()/x.shape[0], on_step=False, on_epoch=True, prog_bar=True)
        if self.save_logpath is not None:
            self.val_step_sumloss.append(loss)
            self.val_step_numsamp.append(x.shape[0])

        return {'val_loss': loss.mean()}

    def on_validation_epoch_end(self): 
        if self.save_logpath is not None:
            avg_loss = torch.stack(self.val_step_sumloss).sum(0) / sum(self.val_step_numsamp)
            self.val_epoch_outputs.append((self.current_epoch, avg_loss.tolist()))
            # free memory
            self.val_step_sumloss.clear()  
            self.val_step_numsamp.clear()  
    
    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self.test_output_list = []
        self.test_batchsize_list = []
        return
    
    def test_step(self, batch:Tuple[Tensor, Tensor, Tensor], batch_idx):
        base_x, x, y = batch
        ## Model prediction
        y_hat = self(base_x, x).squeeze(-1)   # (n_model, b) 
        ## Compute loss
        loss = F.mse_loss(
            input     = y_hat,                             # (n_model, b) 
            target    = torch.tile(y, (self.n_model, 1)),  # (n_model, b) 
            reduction = 'none')                            # (n_model, b)
            
        loss = loss.sum(-1)                   # (n_model, )
        
        self.test_output_list.append(loss)
        self.test_batchsize_list.append(batch[0].shape[0])
    
    def on_test_epoch_end(self):
        n_train  = sum(self.test_batchsize_list) 
        losses   = torch.stack(self.test_output_list, dim=0).sum(dim=0) / n_train # (n_model,)
        for i in range(self.n_model):
            self.log('_'.join(['MLP', str(i)]), losses[i])

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.nets.parameters() if self.disable_base_x_embedding_training else self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        return [optimizer]#, [scheduler]

def energy_loss_two_sample(
        x_true       : Tensor, 
        x_perb_1     : Tensor, 
        x_perb_2     : Tensor, 
        beta         : float=1,
        multichannel : bool=False,  
        verbose      : bool=False,
        reduction    : str ="mean") -> Tensor:
    """Loss function based on the energy score (estimated based on two samples).
    
    Args:
        x_true       : iid samples from the true distribution.
                       (...,d) 
        x_perb_1     : iid samples from the estimated distribution.
                       (...,d)
        x_perb_2     : iid samples from the estimated distribution.
                       (...,d)
        beta         : power parameter in the energy score.
        multichannel : input of vectorize.
        verbose      : whether to return two terms of the loss.
        reduction    : "mean" or "sum"
    
    Returns:
        loss    : energy loss.
    """
    x_true   = vectorize(x_true,   multichannel) # (n_sample, n_dim) or (n_channel, n_sample, n_dim)
    x_perb_1 = vectorize(x_perb_1, multichannel) # (n_sample, n_dim) or (n_channel, n_sample, n_dim)
    x_perb_2 = vectorize(x_perb_2, multichannel) # (n_sample, n_dim) or (n_channel, n_sample, n_dim)
    
    s1 = ((torch.norm(x_perb_1 - x_true, 2, dim=-1).pow(beta)) / 2       # (n_sample,) or (n_channel,n_sample)
            + (torch.norm(x_perb_2 - x_true, 2, dim=-1).pow(beta)) / 2)  # (n_sample,) or (n_channel,n_sample)
    s2 = (torch.norm(x_perb_1 - x_perb_2, 2, dim=-1).pow(beta))          # (n_sample,) or (n_channel,n_sample)
    if reduction == "mean": 
        s1_o = s1.mean(dim=-1)                   # () or (n_channel,)
        s2_o = s2.mean(dim=-1)                   # () or (n_channel,)
    else: 
        s1_o = s1.sum(dim=-1)                    # () or (n_channel,)
        s2_o = s2.sum(dim=-1)                    # () or (n_channel,)
    if verbose:
        return torch.stack([(s1_o - s2_o/2), s1_o, s2_o], dim=0)  # (3,) or (3,n_channel)
    else:
        return (s1_o - s2_o/2)                                    # ()   or (n_channel,)
    
class StoModelNonPL(nn.Module):
    def __init__(
            self, 
            *n_hidden,  
            base_x_embedding_dim ,
            n_model , 
            no_batch_norm , 
            n_noise ,
            noise_std ,
            search_domain ,
            trans_method , 
            trans_rbf_nrad , 
            use_double):
        super().__init__()   
        
        if use_double:
            dtype = torch.float64
        else:
            dtype = torch.float32

        self.latent_embedding_fn = LatentEmbedding(search_domain, trans_method, trans_rbf_nrad, dtype)
        n_in = self.latent_embedding_fn.n_features + base_x_embedding_dim
        self.nets = nn.ModuleList([
            self.create_net(dtype, n_in, n_noise, noise_std , no_batch_norm, *n_hidden) 
            for _ in range(n_model)
        ])
        self.n_noise = n_noise
    
    def forward(self, base_x_embedding:Tensor, x:Tensor):
        """
        base_x_embedding: (...,d_base)
        x: (1,d) or (d,)
        """
        base_x_embedding = base_x_embedding.reshape(-1, base_x_embedding.shape[-1])
        b = base_x_embedding.shape[0]
        # Reshape (b, d)
        
        # Get x embeddings: (b, d_2)
        latent_embedding = self.latent_embedding_fn(x.reshape(-1))                       
         
        # Concatenate the embeddings: (b, d_1 + d_2)
        emb = torch.concatenate([base_x_embedding, torch.tile(latent_embedding, (b,1))], dim=-1)

        out = torch.stack([net(emb) for net in self.nets], dim = 0)
        return out # (n_model, b, out_dim=1) 
    
    def create_net( 
            self,
            dtype: torch.dtype, 
            n_in:int, 
            n_noise:int, 
            noise_std:float, 
            no_batch_norm:bool, 
            *n_hidden
        ):
        Layers = []
        Layers.append(StoBlock(in_dim=n_in, out_dim=n_hidden[0], no_batch_norm=no_batch_norm, noise_dim=n_noise, noise_std=noise_std))        
        for i in range(1, len(n_hidden)):    
            Layers.append(StoBlock(in_dim=n_hidden[i-1], out_dim=n_hidden[i], no_batch_norm=no_batch_norm, noise_dim=n_noise, noise_std=noise_std)) 
        Layers.append(nn.Linear(in_features=n_hidden[-1], out_features=1))
        return nn.Sequential(*Layers).to(dtype) 

    def loss_fn(self, base_x, x, y):
        if self.n_noise > 0:
            ## Model prediction
            y_hat_1 = self(base_x, x)             # (n_model, b, out_dim=1) 
            y_hat_2 = self(base_x, x)             # (n_model, b, out_dim=1) 
            
            ## Compute loss 
            loss = energy_loss_two_sample(
                x_true=y.unsqueeze(-1),           # (n_model, b, out_dim=1)
                x_perb_1=y_hat_1,                 # (n_model, b, out_dim=1)
                x_perb_2=y_hat_2,                 # (n_model, b, out_dim=1)
                multichannel=True, 
                reduction="mean"
            )                                     # (n_model, )  

            loss = loss.mean()                    # (,) 
                
        else:
            ## Model prediction
            y_hat = self(base_x, x).squeeze(-1)                 # (n_model, b) 
            ## Compute loss
            loss = nn.functional.mse_loss(
                input     = y_hat,                              # (n_model, b) 
                target    = torch.tile(y, (len(self.nets), 1)),   # (n_model, b) 
                reduction = 'mean')                             # (,)
        return loss

class RntModelNonPL(nn.Module):
    def __init__(
            self, 
            *n_hidden,  
            base_x_embedding_dim:int=512,
            n_model:int=10, 
            search_domain:Tensor=None,
            trans_method:str="rbf", 
            trans_rbf_nrad:int=5, 
            use_double:bool=False,
            **kwargs):
        super().__init__()   
        
        if use_double:
            dtype = torch.float64
        else:
            dtype = torch.float32

        self.latent_embedding_fn = LatentEmbedding(search_domain, trans_method, trans_rbf_nrad, dtype)
        n_in = self.latent_embedding_fn.n_features + base_x_embedding_dim
        self.nets = nn.ModuleList([
            self.create_net(dtype, n_in, *n_hidden) 
            for _ in range(n_model)
        ])
    
    def forward(self, base_x_embedding:Tensor, x:Tensor):
        """
        base_x_embedding: (...,d_base)
        x: (1,d) or (d,)
        """
        base_x_embedding = base_x_embedding.reshape(-1, base_x_embedding.shape[-1])
        b = base_x_embedding.shape[0]
        # Reshape (b, d)
        
        # Get x embeddings: (b, d_2)
        latent_embedding = self.latent_embedding_fn(x.reshape(-1))                       
         
        # Concatenate the embeddings: (b, d_1 + d_2)
        emb = torch.concatenate([base_x_embedding, torch.tile(latent_embedding, (b,1))], dim=-1)

        out = torch.stack([net(emb) for net in self.nets], dim = 0)
        return out # (n_model, b, out_dim=1) 
    
    def create_net( 
            self,
            dtype: torch.dtype, 
            n_in:int, 
            *n_hidden
        ):
        Layers = []
        Layers.append(RntBlock(in_dim=n_in, out_dim=n_hidden[0]))        
        for i in range(1, len(n_hidden)):    
            Layers.append(RntBlock(in_dim=n_hidden[i-1], out_dim=n_hidden[i])) 
        Layers.append(nn.Linear(in_features=n_hidden[-1], out_features=1))
        return nn.Sequential(*Layers).to(dtype) 

    def loss_fn(self, base_x, x, y):
        
        ## Model prediction
        y_hat = self(base_x, x).squeeze(-1)                 # (n_model, b) 
        ## Compute loss
        loss = nn.functional.mse_loss(
            input     = y_hat,                              # (n_model, b) 
            target    = torch.tile(y, (len(self.nets), 1)),   # (n_model, b) 
            reduction = 'mean')                             # (,)
        return loss


