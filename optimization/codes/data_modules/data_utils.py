from torchvision import datasets  
from torch import Tensor
from torch.utils.data import DataLoader
import numpy as np
import numpy.random as npr

import os
from collections import Counter
from sklearn.model_selection import train_test_split



def scale_func(x, bounds):
    """
    x: (...,d)
    bounds: (d,2)
    """
    xmin = np.min(x, axis=0)
    xmax = np.max(x, axis=0)
    
    # scale between 0 and 1
    x_scale = (x-xmin)/(xmax-xmin)
    # scale between lower bounds and upper bounds (in each dimension) 
    x_scale = x_scale*(bounds[:,1]-bounds[:,0])+bounds[:,0]
    return x_scale

def lowrank_perb(n_sample, low_dim, high_dim, 
                 bounds=np.array([0,1]), 
                 rng:npr.Generator=npr.default_rng(),
                 ):
    """Low-rank perturbation
        
    Apply PPCA on the randomly generated vectors of dimension low_dim
    scale the results to match with bounds
    """ 
    if rng is None:
        rng = npr.default_rng()
    # low-rank perturbation
    U      = rng.uniform(size = (n_sample, low_dim)) 
    theta  = rng.normal(size = (low_dim, high_dim))
    x_perb = U.dot(theta)
    return scale_func(x_perb, bounds) 

def compute_mean_std(dataset:datasets, num_channel:int=3):
    """
    datasets: of shape [N, C, H, W]
    """
    loader = DataLoader(dataset, 
                        batch_size=len(dataset), 
                        shuffle=False)
    imgs, _ = next(iter(loader))
    if num_channel == 3:
        return imgs.mean([0,2,3]).tolist(), imgs.std([0,2,3]).tolist()
    else: 
        return imgs.mean().reshape(-1).tolist(), imgs.std().reshape(-1).tolist()
    
def get_indices_shift(data:datasets,
        targets_to_shrink:list=[1,2,7], 
        shrink_to_proportion:float=0.1,
        rng:npr.Generator=npr.default_rng(),    
    ):
    ind_indices = np.full(len(data), True)

    for target_curr in targets_to_shrink:
        indices_curr = [i for i, (_, target) in enumerate(data) if target == target_curr]
        # number of indices in the current class
        n_curr = len(indices_curr)
        # number of indices to be leftout
        n_to_leftout = int(n_curr*(1-shrink_to_proportion))
        # choice the indices to be leftout
        sub_lefout = rng.choice(n_curr, n_to_leftout, replace=False)
        # note down the indices to be leftout
        ind_indices[np.array(indices_curr)[sub_lefout]] = False
 
    indices = np.where(ind_indices)[0]
    return indices 

def get_data_indices( 
    dataset              : datasets,  
    ndata                : int=None,
    shift                : bool=False, 
    targets_to_shift     : list=[1,2,7],
    shrink_to_proportion : float=0.2,
    indices_saved_path   : str=None,
    seed                 : int=101) -> np.ndarray:
    
    if ndata is None:
        ndata = len(dataset)
    
    if os.path.isfile(indices_saved_path): 
        print(f"Found {indices_saved_path}.")
        print(f"Retrieving results from {indices_saved_path}...")
        loaded_dict = np.load(indices_saved_path, allow_pickle=True).item()
        kept_indices = loaded_dict['indices']
        label_counts = loaded_dict['label_counts'] 
    else: 
        print(f"Could not find {indices_saved_path}.")
        print(f"Creating {indices_saved_path}...")

        if shift: 
            generator = npr.default_rng(seed)

            ## only keep shrink_to_proportion of data points if their labels are in targets_to_shift 
            kept_indices = get_indices_shift(
                    dataset,
                    targets_to_shift, 
                    shrink_to_proportion,
                    generator
            )
        else:
            kept_indices = np.arange(len(dataset))
        
        ## shorten the data if asked to
        if ndata < len(kept_indices): 
            kept_indices, _ = train_test_split(kept_indices, train_size=ndata, random_state=seed)
        
        if isinstance(dataset.targets[0], Tensor):
            label_counts = Counter([dataset.targets[i].item() for i in kept_indices]) 
        else:
            label_counts = Counter([dataset.targets[i] for i in kept_indices]) 
        save_dict = {'label_counts': label_counts, 'indices': kept_indices}
    
        np.save(indices_saved_path, save_dict)
        print(f"Saved the indices for dataset of size {ndata} (out of {len(dataset)}) in '{indices_saved_path}'.")
     
    print("label_counts:", dict(label_counts)) 
    return kept_indices 

