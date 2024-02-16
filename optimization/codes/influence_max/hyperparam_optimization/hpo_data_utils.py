import os 
from typing import Callable, List 

import torch
from torch import Tensor 
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split

from types import MethodType

"""
Two methods can be used to bind a standalone function 
to an instance of a class in Python:
- Using MethodType from the types module.
    ```
    from types import MethodType

    class FIRST:
        def __init__(self, a, b):
            self.a = a
            self.b = b

    def new_fn1(self, c):
        self.c = c

    first_fn = FIRST(a=1, b=2)
    first_fn.new_fn1 = MethodType(new_fn1, first_fn)
    first_fn.new_fn1(3)  # Sets first_fn.c to 3
    ```
- Using the __get__ method directly on the function object.
    ```
    class FIRST:
        def __init__(self, a, b):
            self.a = a
            self.b = b

    def new_fn1(self, c):
        self.c = c

    first_fn = FIRST(a=1, b=2)
    first_fn.new_fn1 = new_fn1.__get__(first_fn, FIRST)
    first_fn.new_fn1(3)  # Sets first_fn.c to 3
    ```

 Both methods achieve the same result - they bind a standalone 
 function to an instance of a class, allowing it to be called 
 as if it were a method of that instance. 
 Using MethodType is generally more conventional and 
 might be preferred for clarity and readability, 
 but using __get__ is a perfectly valid alternative, 
 especially in situations where you might want to 
 leverage more of Python's dynamic and introspective capabilities.
"""
# def get_y_stats(self, do_normalize_y:bool=False):
#     self.ymean = 0.
#     self.ystd  = 1.
#     if do_normalize_y:
#         ymean = []
#         yvar  = []
#         if self.y_savedir is not None:
#             for ii in range(self.n):
#                 output_filename = 'i{:04d}_output-train.pt'.format(ii)
#                 output_filepath = os.path.join(self.y_savedir, output_filename)
#                 ymean.append(torch.load(output_filepath)['ymean'])
#                 yvar.append(torch.load(output_filepath)['yvar'])
#         else: 
#             for ii in range(self.n):
#                 # the invidual training loss for a set of hyperparameters
#                 y = self.y_fn(x=self.x[ii])         
#                 ymean.append(y.mean())
#                 yvar.append(y.var())
#         self.ymean = torch.stack(ymean).sum()/self.n
#         self.ystd  = torch.sqrt(torch.stack(yvar).sum()/self.n)


# def get_y(self, selected_base_indices:torch.LongTensor=None):
#     ## Obtain the corresponding y values 
#     selected_y=[]
#     if (self.y_savedir is not None) and (selected_base_indices is not None):
#         for ii in range(self.n):
#             output_filename = 'i{:04d}_output-train.pt'.format(ii)
#             output_filepath = os.path.join(self.y_savedir, output_filename)
#             yy = torch.load(output_filepath)['y'][selected_base_indices]
#             selected_y.append(yy)
#     else: 
#         for ii, xx in enumerate(self.x):
#             # the invidual training loss for a set of hyperparameters
#             yy = self.y_fn(x=xx)
#             selected_y.append(yy)
#     selected_y = torch.stack(selected_y, dim=0) # (n, n_select_base_per_x)

#     # do_y_normalization 
#     selected_y = (selected_y - self.ymean) / self.ystd
#     return selected_y

# def get_dataset(self, n_select_base_per_x:int):
#     ## Select n_select_base number of base samples in total
#     if n_select_base_per_x < self.n_base_train:
#         selected_base_indices, _ = train_test_split(
#             torch.arange(self.n_base_train),
#             train_size=n_select_base_per_x, 
#             shuffle=True, 
#             stratify=self.base_train_y
#         )
#     else:
#         selected_base_indices = torch.arange(self.n_base_train)
#     ## Get selected y
#     selected_y = self.get_y(selected_base_indices)

#     return MapDataset(self.base_train_x[selected_base_indices], 
#                       self.x, 
#                       selected_y)

# def update_train_dataloader(self):  
#     datasets = CustomConcat([self.get_dataset(self.n_select_base_per_x)
#                                 for _ in range(self.n_model)])
#     return DataLoader(
#         datasets,
#         batch_size=min(self.batch_size, len(datasets)), 
#         collate_fn=collate_wrapper,
#         shuffle=True,
#         drop_last=True,
#         num_workers=self.num_workers,
#         pin_memory=self.pin_memory)     

# def update_test_dataloader(self):
#     small_base_train_y = self.get_y(self.small_base_train_i)
#     datasets = MapDataset(self.small_base_train_x, self.x, small_base_train_y)
    
#     return DataLoader(
#         datasets,
#         batch_size=min(self.batch_size, len(datasets)), 
#         collate_fn=collate_wrapper,
#         shuffle=False,
#         drop_last=True,
#         num_workers=self.num_workers,
#         pin_memory=self.pin_memory)     
    
# def get_infmax_datamodule(
#         base_dm, 
#         x:Tensor, 
#         n_select_base:int=None,
#         y_savedir:str=None, 
#         y_fn:Callable[[Tensor],Tensor]=None,
#         do_normalize_y:bool=False,
#         n_model:int=1,
#     ):
#     assert y_fn is not None or y_savedir is not None, "both y_fn and y_savedir is None!"

#     base_dm.x = x
#     base_dm.n = x.shape[0]
#     base_dm.y_savedir = y_savedir 
#     base_dm.y_fn = y_fn 
#     base_dm.n_model = n_model
 
#     if n_select_base is None:
#         n_select_base = base_dm.n_base_train 
#     base_dm.n_select_base_per_x = max(int(n_select_base/base_dm.n), 500)

#     # Create a bound method from the standalone function get_y_stats, get_y and get_dataset, in order. 
#     base_dm.get_y_stats = MethodType(get_y_stats, base_dm)
#     base_dm.get_y_stats(do_normalize_y)
#     assert hasattr(base_dm, "ymean"), f"{type(base_dm).__name__} does not have attribute 'ymean'!"
#     assert hasattr(base_dm, "ystd") , f"{type(base_dm).__name__} does not have attribute 'ystd'!"
#     base_dm.get_y       = MethodType(get_y, base_dm)
#     base_dm.get_dataset = MethodType(get_dataset, base_dm)

#     # Create a bound method from the standalone function update_train_loader and update_test_loader. 
#     base_dm.break_down_dataset = True 
#     base_dm.train_dataloader   = MethodType(update_train_dataloader, base_dm)
#     base_dm.test_dataloader    = MethodType(update_test_dataloader,  base_dm)
#     return base_dm

####################################################################
####################################################################
########################### HELPER #################################
####################################################################
####################################################################
class MapDataset(Dataset):
    def __init__(self, base_x:Tensor, x:Tensor, y:Tensor):
        """
        base_x or base_x_embedding: (n_base, ...)
        x: (n, d)
        y: (n, n_base)
        """
        self.base_x = base_x
        self.x = x
        self.y = y
        self.n, self.n_base = y.shape[0], y.shape[1]
        
        assert self.x.shape[0] == self.y.shape[0], \
            "number of samples in x does not match up with the ones in y (first dimension)!"
        
        assert self.base_x.shape[0] == self.y.shape[1], \
            "number of samples in base_x does not match up with the ones in y (second dimension)!"
        
    def __getitem__(self, i:int):
        row = i % self.n    # Modulo operation to find the row
        col = i // self.n   # Integer division to find the colomn
        return self.base_x[col], self.x[row], self.y[row, col]
    
    def __len__(self):
        return self.n * self.n_base

class CustomConcat(Dataset):
    def __init__(self, datasets: List[Dataset]):
        """
        n_model number of datasets 
        """
        self.datasets = datasets
    
    def __getitem__(self, i):
        """
        the i-th dataset
        """
        base_x = torch.stack([dataset[i][0] for dataset in self.datasets], dim=0)
        x      = torch.stack([dataset[i][1] for dataset in self.datasets], dim=0)
        y      = torch.stack([dataset[i][2] for dataset in self.datasets], dim=0)
        return base_x, x, y
    
    def __len__(self):
        """
        number of data points in each dataset
        """
        return len(self.datasets[0])

class CustomBatch:
    def __init__(self, data):
        """
        b number of data 
        """  
        base_x, x, y = zip(*data) 
        self.base_x = torch.stack(base_x, 1)     # (n_nets, b, C, H, W) or (n_nets, b, d_base)
        self.x = torch.stack(x, 1)               # (n_sets, b, d) 
        self.y = torch.stack(y, 1)               # (n_sets, b) 

    def __call__(self):
        return self.base_x, self.x, self.y
    
    # custom memory pinning method on custom type
    def pin_memory(self): 
        self.base_x = self.base_x.pin_memory()
        self.x = self.x.pin_memory()
        self.y = self.y.pin_memory()
        return self.base_x, self.x, self.y
        
def collate_wrapper(batch):
    return CustomBatch(batch)


 