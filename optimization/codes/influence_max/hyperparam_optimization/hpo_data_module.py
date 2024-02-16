import os 
from typing import Callable

import torch 
from torch.utils.data import DataLoader 
from torch import Tensor
# from functorch import vmap 

import numpy as np

import lightning.pytorch as pl 
from sklearn.model_selection import train_test_split

from codes.influence_max.hyperparam_optimization.hpo_data_utils import CustomConcat, MapDataset, collate_wrapper

class OptDataModule(pl.LightningDataModule):
    def __init__(
        self,
        x:Tensor, 
        base_dm:pl.LightningDataModule,
        n_select_base:int=None,  
        y_fn:Callable[[np.ndarray], np.ndarray]=None,
        y_savedir:str=None,    
        do_normalize_y:bool=False,
        n_model:int=1,
        batch_size:int=64,
        use_pretrained_featemb:bool=False,
        differ_sample_for_each_model:bool=False
    ):
        """
        x: (n, d)
        base_dm: base data module to train the model 
            - n_base_train
            - base_train.data
            - base_train_y
            - train_indices 
            - small_train_x
            - small_train_i
            - small_targt_x
            - small_targt_i 
            - num_workers
            - pin_memory
        n_select_base: int 
            total number of base data samples. For each set of x, 
            it selects the same subset of (base_x, base_y) of size 
            int(n_select_base/n) to obtain the corresponding y.
        y_fn:  
            callable function to give y that takes (x, base_x) as inputs.
            If y_savedir is None, y_fn has to be given.
        y_savedir: 
            directory where the precomputed y are saved for all pairs of (x, base_x).
            It checks whether y_savedir is None first. If is None, y_fn has to be given.
        y_mchoise: 
            model choice to compute y. Can be either 'last' or 'best'. 
        do_normalize_y: bool
            whether to normalize y. This is the input of function get_y_stats, 
            which sets attributes ymean and ystd. 
            If False then ymean=0, ystd=1.
        n_trainset: int 
            number of training sets of ((base_x, x), y)
        """
        super().__init__() 
        self.base_dm = base_dm 
        self.x = x
        self.n = x.shape[0]
        self.y_savedir = y_savedir 
        self.y_fn = y_fn 
        self.n_model = n_model
        assert y_fn is not None or y_savedir is not None, "both y_fn and y_savedir is None!"
           
        if n_select_base is None:
            n_select_base = base_dm.n_base_train 
        self.n_select_base = max(n_select_base, 500)
        
        self.batch_size = batch_size
        self.use_pretrained_featemb = use_pretrained_featemb
        self.differ_sample_for_each_model = differ_sample_for_each_model
        
        ## Get normalization statistics
        self.get_y_stats(do_normalize_y)

    def get_y_stats(self, do_normalize_y:bool=False):
        self.ymean = 0.
        self.ystd  = 1.
        if do_normalize_y:
            ymean = []
            yvar  = [] 
            if self.y_savedir is not None:
                for ii in range(self.n):
                    output_filename = 'i{:04d}_output-train.pt'.format(ii)
                    output_filepath = os.path.join(self.y_savedir, output_filename)
                    ymean.append(torch.load(output_filepath)['ymean'])
                    yvar.append(torch.load(output_filepath)['yvar'])
            else: 
                for ii in range(self.n):
                    # the invidual training loss for a set of hyperparameters
                    y = self.y_fn(x=self.x[ii])         
                    ymean.append(y.mean())
                    yvar.append(y.var())
            self.ymean = torch.stack(ymean).sum()/self.n
            self.ystd  = torch.sqrt(torch.stack(yvar).sum()/self.n)

    def get_y(self, selected_base_indices:torch.LongTensor=None):
        ## Obtain the corresponding y values 
        selected_y=[]
        if (self.y_savedir is not None) and (selected_base_indices is not None):
            for ii in range(self.n):
                output_filename = 'i{:04d}_output-train.pt'.format(ii)
                output_filepath = os.path.join(self.y_savedir, output_filename)
                yy = torch.load(output_filepath)['y'][selected_base_indices]
                selected_y.append(yy)
        else: 
            for ii, xx in enumerate(self.x):
                # the invidual training loss for a set of hyperparameters
                yy = self.y_fn(x=xx)
                selected_y.append(yy)
        selected_y = torch.stack(selected_y, dim=0) # (n, n_select)

        # do_y_normalization 
        selected_y = (selected_y - self.ymean) / self.ystd
        return selected_y # (n, n_select)
    
    def get_precomputed_base_x_embedding(self, selected_base_indices:torch.LongTensor=None, data:str="train"):
        ## Obtain the corresponding precomputed base_x_embedding values 
        # if (self.y_savedir is not None) and (selected_base_indices is not None):
        #     output_filepath = os.path.join(self.y_savedir, f'precomputed_featemb-{data}.pt')
        #     assert os.path.isfile(output_filepath), f"Pretrained feature embedding cannot be found in path {output_filepath}"
        #     selected_base_x_embedding = torch.load(output_filepath)[selected_base_indices]
        # else: 
        #     raise ValueError(f"When precomputed_base_x_embedding = True, cannot find it in y_savedir={self.y_savedir}")
        output_filepath = os.path.join(self.y_savedir, f'precomputed_featemb-{data}.pt')
        selected_base_x_embedding = torch.load(output_filepath)[selected_base_indices]
        return selected_base_x_embedding # (n_select, d_base)

    def get_train_dataset(self, n_select_base_per_x:int):
        ## Select n_select_base number of base samples in total
        if n_select_base_per_x < self.base_dm.n_base_train:
            selected_base_indices, _ = train_test_split(
                torch.arange(self.base_dm.n_base_train),
                train_size=n_select_base_per_x, 
                shuffle=True, 
                stratify=self.base_dm.base_train_y
            )
        else:
            selected_base_indices = torch.arange(self.base_dm.n_base_train)
        ## Get selected y
        selected_y = self.get_y(selected_base_indices)

        if self.use_pretrained_featemb:
            ## Get precomputed selected base_x_embedding
            selected_base_x_embedding = self.get_precomputed_base_x_embedding(selected_base_indices, "train")
             
            return MapDataset(
                selected_base_x_embedding, 
                self.x, 
                selected_y)   
        else:
            return MapDataset(
                self.base_dm.base_train_x[selected_base_indices], 
                self.x, 
                selected_y)   
    
    def setup(self, stage=None):
        """
        base_dm.setup() only has been defined for two stages, "fit", "test" and "predict".

        The test_dataloader does not rely on the stage="test", 
        since it works on small_base_train_x and small_base_train_i, attributes always carried by base_dm.
        """
        if stage == "fit":
            self.base_dm.break_down_dataset = True
            self.base_dm.setup(stage)

    def train_dataloader(self):  
        if self.differ_sample_for_each_model:
            datasets = CustomConcat([self.get_train_dataset(self.n_select_base)
                                    for _ in range(self.n_model)])
            return DataLoader(
                datasets,
                batch_size=min(self.batch_size, len(datasets)), 
                collate_fn=collate_wrapper,
                shuffle=True,
                drop_last=True,
                num_workers=self.base_dm.num_workers,
                pin_memory=self.base_dm.pin_memory)  
        else:  
            datasets = self.get_train_dataset(self.n_select_base)

            return DataLoader(
                datasets,
                batch_size=min(self.batch_size, len(datasets)), 
                shuffle=True,
                drop_last=True,
                num_workers=self.base_dm.num_workers,
                pin_memory=self.base_dm.pin_memory)     
        
    def test_dataloader(self):
        small_base_train_y = self.get_y(self.base_dm.small_base_train_i)
        if self.use_pretrained_featemb:
            ## Get precomputed selected base_x_embedding
            selected_base_x_embedding = self.get_precomputed_base_x_embedding(
                self.base_dm.small_base_train_i, "train")
            
            datasets = MapDataset(selected_base_x_embedding, self.x, small_base_train_y)
        else:
            datasets = MapDataset(self.base_dm.small_base_train_x, self.x, small_base_train_y)
        
        return DataLoader(
            datasets,
            batch_size=min(self.batch_size, len(datasets)), 
            shuffle=False,
            drop_last=True,
            num_workers=self.base_dm.num_workers,
            pin_memory=self.base_dm.pin_memory)  
      
    def val_dataloader(self):
        small_base_train_y = self.get_y(self.base_dm.small_base_train_i)
        if self.use_pretrained_featemb:
            ## Get precomputed selected base_x_embedding
            selected_base_x_embedding = self.get_precomputed_base_x_embedding(
                self.base_dm.small_base_train_i, "train")
            
            datasets = MapDataset(selected_base_x_embedding, self.x, small_base_train_y)
        else:
            datasets = MapDataset(self.base_dm.small_base_train_x, self.x, small_base_train_y)
        
        return DataLoader(
            datasets,
            batch_size=min(self.batch_size, len(datasets)), 
            shuffle=False,
            drop_last=True,
            num_workers=self.base_dm.num_workers,
            pin_memory=self.base_dm.pin_memory)    


