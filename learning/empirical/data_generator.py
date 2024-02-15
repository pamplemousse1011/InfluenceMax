# Inspired by https://github.com/BlackHC/BatchBALD/tree/3cb37e9a8b433603fc267c0f1ef6ea3b202cfcb0
import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.sparse import csr_matrix as csrmat
import scipy.linalg
import pickle
import collections
from dataclasses import dataclass
import enum
import itertools
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, random_split
from torchvision import datasets, transforms
import torch
import pickle 
from typing import List
import os
from sklearn.model_selection import train_test_split 


import mnist_model
import text_model
import emnist_model
from active_learning_data import ActiveLearningData
from utils import get_balanced_sample_indices
from train_pl import train_pl_model
import subrange_dataset
from preprocess_text import preprocess_tweet, preprocess_movie



@dataclass
class ExperimentData:
    active_learning_data: ActiveLearningData
    train_dataset: Dataset
    available_dataset: Dataset
    validation_dataset: Dataset
    test_dataset: Dataset
    initial_samples: List[int]
    
    


@dataclass
class DataSource:
    train_dataset: Dataset
    validation_dataset: Dataset = None
    test_dataset: Dataset = None



def get_MNIST(root):
    # num_classes=10, input_size=28
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, transform=transform)

    return DataSource(train_dataset=train_dataset, test_dataset=test_dataset)

def get_EMNIST(root, split="letters", mean=0.1722, std=0.3309):
    transform = transforms.Compose(
        [transforms.ToTensor(), 
        transforms.Normalize((mean,), (std,))]
    )

    if split == "letters":
        def target_transform(x):
            return x-1
    else:
        def target_transform(x):
            return x

    train_dataset = datasets.EMNIST(root=root, split=split, train=True, download=True,
                                    transform=transform, target_transform=target_transform)

    test_dataset = datasets.EMNIST(root=root, split=split, train=False, 
                                    transform=transform, target_transform=target_transform)
    """
        Table II contains a summary of the EMNIST datasets and
        indicates which classes contain a validation subset in the
        training set. In these datasets, the last portion of the training
        set, equal in size to the testing set, is set aside as a validation
        set. Additionally, this subset is also balanced such that it
        contains an equal number of samples for each task. If the
        validation set is not to be used, then the training set can be
        used as one contiguous set.
    """
    
    if split in ("letters", "balanced"):
        # Balanced contains a test set
        split_index = len(train_dataset) - len(test_dataset)
        train_dataset, validation_dataset = subrange_dataset.dataset_subset_split(train_dataset,
                                                                                    split_index)
    else:
        validation_dataset = None
    return DataSource(
        train_dataset=train_dataset, test_dataset=test_dataset, validation_dataset=validation_dataset
    )
  
class ZipDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)    

class MakeDataset(Dataset):
    def __init__(self, data, targets):
        super(MakeDataset).__init__()
        self.data     = data
        self.targets  = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, to_tensor = False):
        if to_tensor:
            x_train = torch.tensor(csrmat.todense(self.data[idx])).float()
            return x_train, self.targets[idx]
        else:
            return self.data[idx], self.targets[idx]


class DatasetEnum(enum.Enum):
    mnist           = "mnist"
    emnist_balanced = "emnist_balanced"
    emnist_bymerge  = "emnist_bymerge"
    emnist_letters  = "emnist_letters"
    movie = "movie"
    tweet = "tweet"

    def get_data_source(self, path_data=None):
        if self == DatasetEnum.mnist:
            return get_MNIST(root=path_data)
        elif self == DatasetEnum.emnist_letters:
            # letters: training: 124800 test: 20800
            return get_EMNIST(root=path_data, split="letters", mean=0.1722, std=0.3309)
        elif self == DatasetEnum.emnist_balanced:
            # balanced: training: 112800 test: 18800
            return get_EMNIST(root=path_data, split="balanced", mean=0.1751, std=0.3332)
            
        elif self == DatasetEnum.emnist_bymerge:
            # num_classes=47, input_size=28,
            # train: 697932 test: 116323
            return get_EMNIST(root=path_data, split="bymerge", mean=0.1736, std=0.3317)
        elif self == DatasetEnum.movie:
            
            sentiment = preprocess_movie(root=path_data)
             
            num_test = 16384
            X_train, X_test, y_train, y_test = train_test_split(
                    sentiment['bag_of_words'],
                sentiment['targets'],
                test_size=num_test, 
                train_size=None, 
                random_state=42
            )

            num_val = 1024
            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=num_val, 
                random_state=42
            )
            return DataSource(
                train_dataset=MakeDataset(X_train.to(torch.float32), y_train.to(torch.int64)), 
                test_dataset=MakeDataset(X_test.to(torch.float32), y_test.to(torch.int64)), 
                validation_dataset=MakeDataset(X_val.to(torch.float32), y_val.to(torch.int64)), 
            )
        elif self == DatasetEnum.tweet:

            sentiment = preprocess_tweet(root=path_data)
            
            tweet_dataset = MakeDataset(
                sentiment['bag_of_words'].to(torch.float32), 
                sentiment['targets'].to(torch.int64)
            )
            
            number_of_testing = 16384
            number_of_training = len(tweet_dataset) - number_of_testing

            train, test = random_split(tweet_dataset, 
                                       [number_of_training , number_of_testing],
                                       generator=torch.Generator().manual_seed(1))

            number_of_val = 1024
            number_of_training = len(train) - number_of_val
            train, val = random_split(train, 
                                     [number_of_training , number_of_val],
                                     generator=torch.Generator().manual_seed(1))

            return DataSource(
                train_dataset=train, 
                test_dataset=test, 
                validation_dataset=val, 
            )
        else:
            raise NotImplementedError(f"Unknown dataset {self}!")

    @property
    def num_classes(self):
        if self == DatasetEnum.mnist:
            return 10
        elif self in (DatasetEnum.emnist_balanced, DatasetEnum.emnist_bymerge):
            return 47
        elif self == DatasetEnum.emnist_letters:
            return 26
        elif self in (DatasetEnum.movie,
                      DatasetEnum.tweet):
            return 2 
        else:
            raise NotImplementedError(f"Unknown dataset {self}!")
    def create_net_factory(self):
        num_classes = self.num_classes
        if self == DatasetEnum.mnist:
            return mnist_model.NormalNet(
                in_features=784, 
                out_features=num_classes
            )
        elif self in (
            DatasetEnum.emnist_balanced, 
            DatasetEnum.emnist_bymerge, 
            DatasetEnum.emnist_letters
        ):
            return emnist_model.NormalNet(
                in_features=784, 
                out_features=num_classes
            )
        elif self == DatasetEnum.movie:
            return text_model.NormalNet(
                in_features=7004,
                out_features=num_classes
            )
        elif self == DatasetEnum.tweet:
            return text_model.NormalNet(
                in_features=4504, 
                out_features=num_classes,
                hidden_features=128
            )
        else:
            raise NotImplementedError(f"Unknown dataset {self}!")

    def train_model(
        self,
        num_models: int,
        train_loader: DataLoader,
        num_gpus: int,
        args,
        test_loader: DataLoader=None,
        validation_loader: DataLoader=None,
    ):    
        model, test_metrics = train_pl_model(
            self.create_net_factory,
            num_models,
            train_loader,
            num_gpus,
            args,
            test_loader,
            validation_loader,
        )
        return model, test_metrics


def get_experiment_data(
        data_source,
        num_classes,
        initial_samples,
        initial_samples_per_class,
        validation_set_size,
        use_validation_set=True, 
):

    train_dataset, test_dataset, validation_dataset = (
        data_source.train_dataset,
        data_source.test_dataset,
        data_source.validation_dataset,
    )

    active_learning_data = ActiveLearningData(train_dataset)
    
    if initial_samples is None:
        initial_samples = list(
            itertools.chain.from_iterable(
                get_balanced_sample_indices(
                    get_targets(train_dataset), num_classes=num_classes, n_per_digit=initial_samples_per_class
                ).values()
            )
        )

    # Split off the validation dataset after acquiring the initial samples.
    active_learning_data.acquire(initial_samples)
    
    if use_validation_set:
        if validation_dataset is None:
            print("Acquiring validation set from training set.")
            if not validation_set_size:
                validation_set_size = len(test_dataset)

            validation_dataset = active_learning_data.extract_dataset(validation_set_size)
            
        else:
            if validation_set_size == 0:
                print("Using provided validation set.")
                validation_set_size = len(validation_dataset)
            if validation_set_size < len(validation_dataset):
                print("Shrinking provided validation set.")
                validation_dataset = Subset(
                    validation_dataset, torch.randperm(len(validation_dataset))[:validation_set_size].tolist()
                )
                

    show_class_frequencies = False
    if show_class_frequencies:
        print("Distribution of training set classes:")
        classes = get_target_bins(train_dataset)
        print(classes)
        
        if use_validation_set:
            print("Distribution of validation set classes:")
            classes = get_target_bins(validation_dataset)
            print(classes)

        print("Distribution of test set classes:")
        classes = get_target_bins(test_dataset)
        print(classes)

        print("Distribution of pool classes:")
        classes = get_target_bins(active_learning_data.available_dataset)
        print(classes)

        print("Distribution of active set classes:")
        classes = get_target_bins(active_learning_data.active_dataset)
        print(classes)

    print(f"Dataset info:")
    num_active = len(active_learning_data.active_dataset)
    print(f"\t{num_active} active samples")
    num_available = len(active_learning_data.available_dataset)
    print(f"\t{num_available} available samples")
    if use_validation_set:
        print(f"\t{len(validation_dataset)} validation samples")
    print(f"\t{len(test_dataset)} test samples")

    return ExperimentData(
        active_learning_data=active_learning_data,
        train_dataset=active_learning_data.active_dataset,
        available_dataset=active_learning_data.available_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        initial_samples=initial_samples,
    )





def get_target_bins(dataset):
    classes = collections.Counter(int(target) for target in get_targets(dataset))
    return classes



def get_targets(dataset):
    """Get the targets of a dataset without any target target transforms(!)."""
    if isinstance(dataset, Subset):
        targets = get_targets(dataset.dataset)
        return torch.as_tensor(targets)[dataset.indices]
    if isinstance(dataset, ConcatDataset):
        return torch.cat([get_targets(sub_dataset) for sub_dataset in dataset.datasets])
    if isinstance(
        dataset, (datasets.EMNIST)
    ):  
        return torch.as_tensor(dataset.target_transform(dataset.targets))
    if isinstance(
        dataset, (datasets.MNIST)
    ):
        return torch.as_tensor(dataset.targets)
    if isinstance(dataset, MakeDataset):
        return dataset.targets

    
    raise NotImplementedError(f"Unknown dataset {dataset}!")

def get_inputs(dataset):
    """Get the data of a dataset without any data transforms(!)."""
    if isinstance(dataset, Subset):
        inputs = get_inputs(dataset.dataset)
        return torch.as_tensor(inputs)[dataset.indices]
    if isinstance(dataset, ConcatDataset):
        return torch.cat([get_inputs(sub_dataset) for sub_dataset in dataset.datasets])
    if isinstance(
        dataset, (datasets.EMNIST)
    ):  
        out = torch.as_tensor(dataset.data)
        if dataset.split == "letters":
            return (out/255 - 0.1722)/0.3309
        elif dataset.split == "balanced":
            return (out/255 - 0.1751)/0.3332
    if isinstance(
        dataset, (datasets.MNIST)
    ):  
        out = torch.as_tensor(dataset.data)
        return (out/255 - 0.1307)/0.3081
    if isinstance(dataset, MakeDataset):
        return dataset.data

    raise NotImplementedError(f"Unknown dataset {dataset}!")




def update_data_loader(
    dataset, num_models=1, leave_one_out=False, **kwargs
):
    num_train = len(dataset)
    indicator_jn = torch.full((num_train, num_models), True)

    if leave_one_out: 
        idx_jn = torch.randperm(num_train)[:num_models]
        for i in range(1, num_models):
            indicator_jn[idx_jn[i]][i] = False

    dataset = ZipDataset(indicator_jn, dataset)

        
    return DataLoader(dataset, **kwargs)

    
