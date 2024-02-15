# See https://github.com/BlackHC/BatchBALD/tree/3cb37e9a8b433603fc267c0f1ef6ea3b202cfcb0
import collections
import typing
import torch
from torch.utils.data import Subset, Dataset
import gc

def gc_cuda():
    gc.collect()
    torch.cuda.empty_cache()


def get_balanced_sample_indices(target_classes: typing.List, num_classes, n_per_digit=2) -> typing.Dict[int, list]:
    permed_indices = torch.randperm(len(target_classes))

    initial_samples_by_class = collections.defaultdict(list)
    if n_per_digit == 0:
        return initial_samples_by_class

    finished_classes = 0
    for i in range(len(permed_indices)):
        permed_index = int(permed_indices[i])
        index, target = permed_index, int(target_classes[permed_index])

        target_indices = initial_samples_by_class[target]
        if len(target_indices) == n_per_digit:
            continue

        target_indices.append(index)
        if len(target_indices) == n_per_digit:
            finished_classes += 1

        if finished_classes == num_classes:
            break

    return dict(initial_samples_by_class)


def get_subset_base_indices(dataset: Subset, indices: typing.List[int]):
    return [int(dataset.indices[index]) for index in indices]


def get_base_indices(dataset: Dataset, indices: typing.List[int]):
    if isinstance(dataset, Subset):
        return get_base_indices(dataset.dataset, get_subset_base_indices(dataset, indices))
    return indices
