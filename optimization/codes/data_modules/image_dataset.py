import os
from typing import Union
from sklearn.model_selection import train_test_split

import numpy as np 
 
import torch
from torch import Tensor, LongTensor
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST, SVHN, CIFAR10
from torchvision import transforms
 
import lightning.pytorch as pl

from codes.data_modules.data_utils import get_data_indices, compute_mean_std

'''Train CIFAR10 with PyTorch.'''
# https://shonit2096.medium.com/cnn-on-cifar10-data-set-using-pytorch-34be87e09844
# https://github.com/kuangliu/pytorch-cifar/blob/49b7aa97b0c12fe0d4054e670403a16b6b834ddd/main.py#L86
# https://github.com/wujian16/Cornell-MOE/blob/df299d1be882d2af9796d7a68b3f9505cac7a53e/examples/real_functions.py

class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size:int, 
        n_small_data:int=0,
        dtype:torch.dtype=torch.float32,
        data_dir:str=os.environ.get("PATH_DATASETS", "."), 
        seed:int=0, 
        use_validation: bool=False, 
        break_down_dataset:bool=False,
        num_workers:int=1, 
        pin_memory:bool=False,
        **kwargs
    ):
        super().__init__()
        self.n_small_data = n_small_data
        self.dtype = dtype
        self.data_dir = data_dir
        self.batch_size = batch_size  
        self.predict_data = ['targt']
        
        # set the seed 
        self.seed = seed
        self.use_validation = use_validation
        self.break_down_dataset = break_down_dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.base_train = None 
        self.base_targt = None 
    
    def get_small_base(
            self, 
            full_base_train_x : Union[np.ndarray, Tensor], 
            full_base_train_y : Union[np.ndarray, Tensor], 
            full_base_targt_x : Union[np.ndarray, Tensor], 
            train_indices     : Tensor,
            targt_indices     : Tensor,
            n_small_data      : int=5000,
            seed              : int=None
        ):
        """
        INPUT:
        #####################################
        full_base_train_x : np.ndarray (n_full_base_train, H, W, C)
        full_base_train_y : np.ndarray (n_full_base_train, )
        full_base_targt_x : np.ndarray (n_full_base_x, H, W, C)
        train_indices     : LongTensor (n_base_train, )
        targt_indices     : LongTensor (n_base_targt, )
        n_small_data : int

        OUTPUT:
        #####################################
        small_base_train_x : torch.Tensor     (n_small_data, C, H, W)
        small_base_train_i : torch.LongTensor (n_small_data, )
        small_base_targt_x : torch.Tensor     (n_small_data, C, H, W)
        """
        base_train_x = full_base_train_x[train_indices].to(self.dtype)
        base_targt_x = full_base_targt_x[targt_indices].to(self.dtype)
        base_train_y = full_base_train_y[train_indices].to(torch.int64)
        
        # Get indices for the small train set and those for the small targt set 
        n_base_train = base_train_x.shape[0]

        if n_small_data < n_base_train:
            small_train_indices, _ = train_test_split(
                torch.arange(n_base_train), 
                train_size=n_small_data, 
                stratify=base_train_y,
                random_state=seed)
        else:
            small_train_indices = torch.arange(n_base_train)
        
        n_base_targt = base_targt_x.shape[0]
        if n_small_data < n_base_targt:
            small_targt_indices, _ = train_test_split(
                torch.arange(n_base_targt), 
                train_size=n_small_data,
                random_state=seed)
        else:
            small_targt_indices = torch.arange(n_base_targt)
        
        # Construct small train set 
        small_base_train_x = base_train_x[small_train_indices]
        # Construct small targt set 
        small_base_targt_x  = base_targt_x[small_targt_indices]

        return (small_base_train_x, small_train_indices, small_base_targt_x, small_targt_indices)
    
    def train_dataloader(self):
        return DataLoader(self.base_train, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers, 
                          pin_memory=self.pin_memory)
    
    def report_dataloader(self):
        return DataLoader(self.base_train, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers, 
                          pin_memory=self.pin_memory,
                          persistent_workers=True)
    
    def predict_dataloader(self):
        self._data=['train', 'targt']
        return [self.report_dataloader(), 
                DataLoader(self.base_targt, 
                           batch_size=self.batch_size, 
                           shuffle=False, 
                           num_workers=self.num_workers, 
                           pin_memory=self.pin_memory,
                           persistent_workers=True)]
     
## Real data 
class CIFAR10DataModule(ImageDataModule):
    def __init__(
            self, 
            batch_size:int, 
            n_small_data:int=0,
            dtype:torch.dtype=torch.float32,
            data_dir:str=os.environ.get("PATH_DATASETS", "."), 
            seed:int=0, 
            use_validation: bool=False,
            break_down_dataset:bool=False,
            num_workers:int=1, 
            pin_memory:bool=False, 
            ntrain:int=None,
            shift:bool=False, 
            targets_to_shift:list=None,
            shrink_to_proportion:float=1.,
            in_distribution:bool=False,
            **kwargs
        ):
        super().__init__(batch_size, n_small_data, dtype, data_dir,
                         seed, use_validation, break_down_dataset, 
                         num_workers, pin_memory)

        # additional shift 
        self.ntrain = ntrain
        self.classes = np.arange(10)

        self.shift = shift
        self.targets_to_shift=targets_to_shift
        self.shrink_to_proportion=shrink_to_proportion
        self.in_distribution=in_distribution

        self.get_carrying_around_values()
    
    def get_carrying_around_values(self):
        """
        Part 1. Get train_indices and targt_indices
        """
        ## Load unnormalized full train dataset and full targt dataset
        unormalized_transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        unormalized_transform_targt = transforms.Compose([
            transforms.ToTensor(),
        ])
         
        unormalized_full_train = CIFAR10(
            root=self.data_dir, train=True,  download=True, transform=unormalized_transform_train)
        unormalized_full_targt = CIFAR10(
            root=self.data_dir, train=False, download=True, transform=unormalized_transform_targt)
        
        if self.ntrain is None:
            self.ntrain = len(unormalized_full_train)  
        ## Get training data indices
        train_indices_saved_path = os.path.join(
            self.data_dir,
            'cifar10_shift{}_train-n{}_target{}_prop{}_seed{}.npy'.format(
                str(self.shift)[0],
                self.ntrain,
                '-'.join(map(str, self.targets_to_shift)) if self.targets_to_shift is not None else 'All',
                self.shrink_to_proportion,
                self.seed
            )
        )
        # set the proportion of targets that are in the targets_to_shift to be shrink_to_proportion
        self.train_indices = torch.from_numpy(get_data_indices(
            unormalized_full_train,  
            self.ntrain,  
            self.shift,
            self.targets_to_shift,
            self.shrink_to_proportion, 
            train_indices_saved_path,
            self.seed
        )).to(torch.int64)

        ## Get targt data indices
        targt_indices_saved_path = os.path.join(
            self.data_dir,
            'cifar10_shift{}_test-{}_target{}_prop{}_seed{}.npy'.format(
                str(self.shift)[0],
                'in' if self.in_distribution else 'ood',
                '-'.join(map(str, self.targets_to_shift)) if self.targets_to_shift is not None else 'All',
                self.shrink_to_proportion,
                self.seed
            )
        )
        if self.in_distribution:
            # set the proportion of targets that are in the targets_to_shift to be shrink_to_proportion
            targt_targets_to_shift = self.targets_to_shift
            targt_shrink_to_proportion = self.shrink_to_proportion
        else:
            # set the proportion of targets that are not in the targets_to_shift to be zero 
            # create a boolean array
            mask = ~np.isin(self.classes, self.targets_to_shift)
            # subset classes using the mask
            targt_targets_to_shift = self.classes[mask].tolist()
            targt_shrink_to_proportion = 0.
        
        self.targt_indices = torch.from_numpy(get_data_indices(
            unormalized_full_targt,  
            None,  
            self.shift,
            targt_targets_to_shift,
            targt_shrink_to_proportion, 
            targt_indices_saved_path,
            self.seed
        )).to(torch.int64)

        print("20:40th indices to make base_train", self.train_indices[20:40])

        # if self.use_validation:
            # self.train_indices, self.val_indices = train_test_split(self.train_indices, test_size=5000, random_state=self.seed)
            # print("First five indices to make valset",   self.val_indices[:5])
    
        print("First twenty indices to make base_targt" , self.targt_indices[:20])
        
        """
        Part 2. Get transform_train and transform_targt
        """
        ## Compute the normalizing statistics
        train_mean, train_std = compute_mean_std(Subset(unormalized_full_train, self.train_indices))
        targt_mean, targt_std = compute_mean_std(Subset(unormalized_full_targt, self.targt_indices))

        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ])
        self.transform_targt  = transforms.Compose([ 
            transforms.ToTensor(),
            transforms.Normalize(targt_mean, targt_std),
        ])

        """
        Part 3. Get the indices of a BALANCED small base_train (both data and targets) and a RANDOMIZED small base_targt (only data)
        """
        if self.n_small_data > 0:
            full_base_train = CIFAR10(root=self.data_dir, train=True,  download=True, transform=self.transform_train)
            full_base_targt = CIFAR10(root=self.data_dir, train=False, download=True, transform=self.transform_targt)
            
            full_base_train_x, full_base_train_y = next(iter(
                DataLoader(full_base_train, 
                           batch_size=len(full_base_train))))
            full_base_targt_x, _ = next(iter(
                DataLoader(full_base_targt, 
                           batch_size=len(full_base_targt)))
            )

            (self.small_base_train_x, self.small_base_train_i, self.small_base_targt_x, self.small_base_targt_i) = self.get_small_base(
                full_base_train_x = full_base_train_x, 
                full_base_train_y = full_base_train_y, 
                full_base_targt_x = full_base_targt_x, 
                train_indices     = self.train_indices,
                targt_indices     = self.targt_indices,
                n_small_data      = self.n_small_data,
                seed              = self.seed
            )
            del full_base_train, full_base_targt
        self.n_base_train = len(self.train_indices)
            
    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage == "predict" or stage == "test":
            # Define the transform with the computed mean and standard deviation
            full_base_train = CIFAR10(root=self.data_dir, train=True,  download=True, transform=self.transform_train)
            # Split into train and val 
            self.base_train = Subset(full_base_train, self.train_indices)
            # if self.use_validation:
            #     self.valset = Subset(base_train, self.val_indices)

            if self.break_down_dataset:
                # Obtain n_base_train and base_train_y for getting model output 
                # self.base_train_x = self.get_ready_data(
                #     full_base_train.data, 
                #     self.train_indices, 
                #     channel_permute=True, 
                #     dtype=self.dtype)
                self.base_train_y = torch.from_numpy(
                    np.array(full_base_train.targets)
                )[self.train_indices].to(torch.int64)
                
                
        # Assign test dataset for use in dataloader(s)
        if stage == "predict" or stage == "test":
            full_base_targt = CIFAR10(root=self.data_dir, train=False, download=True, transform=self.transform_targt)
            self.base_targt = Subset(full_base_targt, self.targt_indices)

class MNISTFRSDataModule(ImageDataModule):
    def __init__(
            self, 
            batch_size:int, 
            n_small_data:int=0,
            dtype:torch.dtype=torch.float32,
            data_dir:str=os.environ.get("PATH_DATASETS", "."), 
            seed:int=0,
            use_validation: bool=False, 
            break_down_dataset: bool=False,
            num_workers:int=1, 
            pin_memory:bool=False,
            **kwargs
        ):
        super().__init__(batch_size, n_small_data, dtype, data_dir,
                         seed, use_validation, break_down_dataset, 
                         num_workers, pin_memory)
        self.get_carrying_around_values()

    def get_carrying_around_values(self):
        """
        Part 1. Get train_indices and targt_indices
        """
        ## SVHN 
        # Load the entire training dataset without normalization
        unnormalized_transform_train = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])
        unnormalized_train = SVHN(
            root=self.data_dir,
            split='train', 
            download=True, 
            transform=unnormalized_transform_train
        )
        # Compute the mean and standard deviation for the training dataset 
        train_mean, train_std = compute_mean_std(unnormalized_train, 1)
        
        self.train_indices = torch.arange(len(unnormalized_train))
        # if self.use_validation:
        #     self.train_indices, self.val_indices = train_test_split(self.train_indices, test_size=5000, random_state=self.seed)
        #     print("First five indices to make valset",  self.val_indices[:5]) 
        
        print("20:40th indices to make base_train", self.train_indices[20:40])

        """
        Part 2. Get transform_train and transform_targt
        """
        self.transform_train = transforms.Compose([
            transforms.Grayscale(num_output_channels=1), # output dimension (1,28,28)
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std)
        ])   

        ## MNIST
        self.transform_targt = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1325,), (0.3105,))])
        
        """
        Part 3. Get a BALANCED small base_train (both data and targets) and a RANDOMIZED small base_targt (only data)
        """
        if self.n_small_data > 0:
            full_base_train = SVHN(root=self.data_dir, split='train', download=True, transform=self.transform_train)  
            full_base_targt = MNIST(root=self.data_dir, train=False, download=True, transform=self.transform_targt)
            
            full_base_train_x, full_base_train_y = next(iter(
                DataLoader(full_base_train, 
                           batch_size=len(full_base_train))))
            full_base_targt_x, _ = next(iter(
                DataLoader(full_base_targt, 
                           batch_size=len(full_base_targt))))
            
            (self.small_base_train_x, self.small_base_train_i, self.small_base_targt_x, self.small_base_targt_i) = self.get_small_base(
                full_base_train_x = full_base_train_x, 
                full_base_train_y = full_base_train_y, 
                full_base_targt_x =  full_base_targt_x, 
                train_indices     = self.train_indices,
                targt_indices     = torch.arange(len(full_base_targt)),
                n_small_data      = self.n_small_data,
                seed              = self.seed
            )
            del full_base_train, full_base_targt
        self.n_base_train = len(self.train_indices)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage == "predict" or stage == "test":
            # Define the transform with the computed mean and standard deviation
            full_base_train = SVHN(root=self.data_dir, split='train', download=True, transform=self.transform_train)          
            # Split into train and val 
            self.base_train = Subset(full_base_train, self.train_indices)
            # if self.use_validation:
            #     self.valset = Subset(base_train, self.val_indices)

            if self.break_down_dataset:
                # Obtain n_base_train and base_train_y for getting model output 
                # self.base_train_x = self.get_ready_data(
                #     full_base_train.data, 
                #     self.train_indices, 
                #     channel_permute=True, 
                #     dtype=self.dtype)
                self.base_train_y = torch.from_numpy(full_base_train.labels)[self.train_indices].to(torch.int64)
                
        # Assign test dataset for use in dataloader(s)
        if stage == "predict" or stage == "test":
            self.base_targt = MNIST(root=self.data_dir, train=False, download=True, transform=self.transform_targt)
    
class SVHNFRMDataModule(ImageDataModule):
    def __init__(
            self, 
            batch_size:int, 
            n_small_data:int=0,
            dtype:torch.dtype=torch.float32,
            data_dir:str=os.environ.get("PATH_DATASETS", "."), 
            seed:int=0,
            use_validation: bool=False, 
            break_down_dataset: bool=False,
            num_workers:int=1, 
            pin_memory:bool=False,
            **kwargs
        ):
        super().__init__(batch_size, n_small_data, dtype, data_dir,
                         seed, use_validation, break_down_dataset, 
                         num_workers, pin_memory)
        self.get_carrying_around_values()

    def get_carrying_around_values(self):
        """
        Part 1. Get train_indices and targt_indices
        """
        ## MNIST
        # Load the entire training dataset without normalization
        self.transform_train = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1325,), (0.3105,))])
        
        self.train_indices = torch.arange(60000)
        # if self.use_validation:
        #     self.train_indices, self.val_indices = train_test_split(self.train_indices, test_size=5000, random_state=self.seed)
        #     print("First five indices to make valset",  self.val_indices[:5]) 
        
        print("20:40th indices to make base_train", self.train_indices[20:40])

        """
        Part 2. Get transform_train and transform_targt
        """
        ## SVHN 
        # Load the entire training dataset without normalization
        unnormalized_transform_targt = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])
        unnormalized_targt = SVHN(
            root=self.data_dir,
            split='train', 
            download=True, 
            transform=unnormalized_transform_targt
        )
        # Compute the mean and standard deviation for the training dataset 
        targt_mean, targt_std = compute_mean_std(unnormalized_targt, 1)
        
        self.transform_targt = transforms.Compose([
            transforms.Grayscale(num_output_channels=1), # output dimension (1,28,28)
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(targt_mean, targt_std)
        ])   

        """
        Part 3. Get a BALANCED small base_train (both data and targets) and a RANDOMIZED small base_targt (only data)
        """
        if self.n_small_data > 0:
            full_base_train = MNIST(root=self.data_dir, 
                                    train=True, 
                                    download=True, 
                                    transform=self.transform_train)
            full_base_targt = SVHN(root=self.data_dir, 
                                   split='test', 
                                   download=True, 
                                   transform=self.transform_targt)  
            
            full_base_train_x, full_base_train_y = next(iter(
                DataLoader(full_base_train, 
                           batch_size=len(full_base_train))))
            full_base_targt_x, _ = next(iter(
                DataLoader(full_base_targt, 
                           batch_size=len(full_base_targt))))
            
            (self.small_base_train_x, self.small_base_train_i, self.small_base_targt_x, self.small_base_targt_i) = self.get_small_base(
                full_base_train_x = full_base_train_x, 
                full_base_train_y = full_base_train_y, 
                full_base_targt_x = full_base_targt_x, 
                train_indices     = self.train_indices,
                targt_indices     = torch.arange(len(full_base_targt)),
                n_small_data      = self.n_small_data,
                seed              = self.seed
            )
            del full_base_train, full_base_targt
        self.n_base_train = len(self.train_indices)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage == "predict" or stage == "test":
            # Define the transform with the computed mean and standard deviation
            full_base_train = MNIST(root=self.data_dir, 
                                    train=True,
                                    download=True, 
                                    transform=self.transform_train)
            # Split into train and val 
            self.base_train = Subset(full_base_train, self.train_indices)
    
            # if self.use_validation:
            #     self.valset = Subset(base_train, self.val_indices)

            if self.break_down_dataset:
                # Obtain n_base_train and base_train_y for getting model output 
                # self.base_train_x = self.get_ready_data(
                #     full_base_train.data, 
                #     self.train_indices, 
                #     channel_permute=True, 
                #     dtype=self.dtype)
                self.base_train_y = full_base_train.targets[self.train_indices].to(torch.int64)
                
        # Assign test dataset for use in dataloader(s)
        if stage == "predict" or stage == "test":
            self.base_targt = SVHN(root=self.data_dir, 
                                   split='test', 
                                   download=True, 
                                   transform=self.transform_targt)          
            
class MNISTDataModule(ImageDataModule):
    def __init__(
            self, 
            batch_size:int, 
            n_small_data:int=0,
            dtype:torch.dtype=torch.float32,
            data_dir:str=os.environ.get("PATH_DATASETS", "."), 
            seed:int=0,
            use_validation: bool=False, 
            break_down_dataset: bool=False,
            num_workers:int=1, 
            pin_memory:bool=False,
            **kwargs
        ):
        super().__init__(batch_size, n_small_data, dtype, data_dir,
                         seed, use_validation, break_down_dataset, 
                         num_workers, pin_memory)
        self.get_carrying_around_values()

    def get_carrying_around_values(self):
        """
        Part 1. Get train_indices and targt_indices
        """
        ## MNIST
        # Load the entire training dataset without normalization
        self.transform_train = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1325,), (0.3105,))])
        
        self.train_indices = torch.arange(60000)
        # if self.use_validation:
        #     self.train_indices, self.val_indices = train_test_split(self.train_indices, test_size=5000, random_state=self.seed)
        #     print("First five indices to make valset",  self.val_indices[:5]) 
        
        print("20:40th indices to make base_train", self.train_indices[20:40])

        """
        Part 2. Get a BALANCED small base_train (both data and targets) and a RANDOMIZED small base_targt (only data)
        """
        if self.n_small_data > 0:
            full_base_train = MNIST(root=self.data_dir, 
                                    train=True, 
                                    download=True, 
                                    transform=self.transform_train)
            full_base_targt = MNIST(root=self.data_dir, 
                                    train=False, 
                                    download=True, 
                                    transform=self.transform_train)
            
            full_base_train_x, full_base_train_y = next(iter(
                DataLoader(full_base_train, 
                           batch_size=len(full_base_train))))
            full_base_targt_x, _ = next(iter(
                DataLoader(full_base_targt, 
                           batch_size=len(full_base_targt))))
            
            (self.small_base_train_x, self.small_base_train_i, self.small_base_targt_x, self.small_base_targt_i) = self.get_small_base(
                full_base_train_x = full_base_train_x, 
                full_base_train_y = full_base_train_y, 
                full_base_targt_x = full_base_targt_x, 
                train_indices     = self.train_indices,
                targt_indices     = torch.arange(len(full_base_targt)),
                n_small_data      = self.n_small_data,
                seed              = self.seed
            )
            del full_base_train, full_base_targt
        self.n_base_train = len(self.train_indices)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage == "predict" or stage == "test":
            # Define the transform with the computed mean and standard deviation
            full_base_train = MNIST(root=self.data_dir, 
                                    train=True,
                                    download=True, 
                                    transform=self.transform_train)
            # Split into train and val 
            self.base_train = Subset(full_base_train, self.train_indices)
    
            # if self.use_validation:
            #     self.valset = Subset(base_train, self.val_indices)

            if self.break_down_dataset:
                # Obtain n_base_train and base_train_y for getting model output 
                # self.base_train_x = self.get_ready_data(
                #     full_base_train.data, 
                #     self.train_indices, 
                #     channel_permute=True, 
                #     dtype=self.dtype)
                self.base_train_y = full_base_train.targets[self.train_indices].to(torch.int64)
                
        # Assign test dataset for use in dataloader(s)
        if stage == "predict" or stage == "test":
            self.base_targt = MNIST(root=self.data_dir, 
                                    train=False,
                                    download=True, 
                                    transform=self.transform_train)
