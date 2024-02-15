# See https://github.com/BlackHC/BatchBALD/tree/3cb37e9a8b433603fc267c0f1ef6ea3b202cfcb0
from torch.utils.data import Subset
import numpy as np


def SubrangeDataset(dataset, begin, end):
    if end > len(dataset):
        end = len(dataset)
    return Subset(dataset, range(begin, end))


def dataset_subset_split(dataset, indices):
    if isinstance(indices, int):
        indices = [indices]

    datasets = []

    last_index = 0
    for index in indices:
        datasets.append(SubrangeDataset(dataset, last_index, index))
        last_index = index
    datasets.append(SubrangeDataset(dataset, last_index, len(dataset)))

    return datasets
