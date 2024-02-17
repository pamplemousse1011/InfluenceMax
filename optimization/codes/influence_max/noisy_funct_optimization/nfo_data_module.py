import torch 
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import Tensor

import lightning.pytorch as pl

from codes.influence_max.utils import zero_mean_unit_var_normalization

class ZipDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)   

class OptDataModule(pl.LightningDataModule):
    def __init__(self, 
        train_x:Tensor,
        train_y:Tensor,  
        do_normalize_y:bool=True,
        n_candidate_model:int=1,  
        n_ensemble_model:int=1,  
        leave_one_out:bool=True, 
        batch_size:int=32,  
        **kwargs
    ):
        super().__init__() 
        ## Normalization statistics 
        train_y_normalized, self.ymean, self.ystd = self.get_y_stats(
            train_y, do_normalize_y)


        self.leave_one_out  = leave_one_out 
        self.batch_size     = batch_size 
        self.kwargs         = kwargs
        
        data_train      = TensorDataset(train_x, train_y_normalized)
        indicator_jn    = self.generate_indicator_jn(n_candidate_model, n_ensemble_model, data_train)
        self.data_train = ZipDataset(indicator_jn, data_train) 
        self.data_test  = data_train
        self.n_train    = len(self.data_train) 
    
    def get_y_stats(self, y:Tensor, do_normalize_y:bool=False):
        """
        y: (n,1) or (n,)
        """
        if do_normalize_y: 
            return zero_mean_unit_var_normalization(y)
        else:
            return y, torch.zeros_like(y.mean()), torch.ones_like(y.std())
        
    def generate_indicator_jn(self, n_candidate_model:int, n_ensemble_model:int, data:TensorDataset):
        n_train = len(data) 
        indicator_jn = torch.full((n_train, n_candidate_model+n_ensemble_model), True) 
        if self.leave_one_out:
            if n_ensemble_model > n_train:
                n_ensemble_model = n_train
            
            idx_jn = torch.randperm(n_train)[:n_ensemble_model]
            for i in range(n_ensemble_model):
                indicator_jn[idx_jn[i]][n_candidate_model+i] = False 
        return indicator_jn

    def compute_inverse_distance_weights(self, data: TensorDataset):
        _, y = data[:]
        # standardize 
        ytilde = (y - y.min()) / (y.max()-y.min())
        # compute interse distance weights 
        w = 1 / torch.clamp((ytilde-ytilde.min())**3, 1e-7)
        return w
    
    def generate_indicator_ws(self, y:Tensor): 
        cutoff = torch.quantile(y, 0.4, interpolation='nearest')
        mask = torch.full(y.shape, False) 
        mask[y<cutoff] = True 
        return mask 

    def train_dataloader(self):
        loader_data = DataLoader(
            self.data_train, 
            batch_size=min(self.batch_size, self.n_train), 
            drop_last=True,
            **self.kwargs
        ) 
        return loader_data 

    def val_dataloader(self):
        loader_val = DataLoader(
            self.data_test, 
            batch_size=min(self.batch_size, self.n_train),
            **self.kwargs
        )
        return loader_val

    def test_dataloader(self):
        loader_test = DataLoader(
            self.data_test, 
            batch_size=min(self.batch_size, self.n_train),
            **self.kwargs
        )
        return loader_test
        
