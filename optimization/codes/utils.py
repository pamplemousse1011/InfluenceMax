import torch
from torch import Tensor
import gc

import numpy as np 
import numpy.random as npr
from scipy.stats import gamma

import os, sys
import datetime

class SuppressPrints:
    def __init__(self, suppress=True):
        self.suppress = suppress

    def __enter__(self):
        if self.suppress:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.suppress:
            sys.stdout.close()
            sys.stdout = self._original_stdout

def gc_cuda():
    gc.collect()
    torch.cuda.empty_cache()

def eta_func(i, emin, emax, T):
    if i < T:
        return emax-i*(emax-emin)/T
    else:
        return emin

def print_x(x:np.ndarray, decimal:int=3):
    format_s = '{:.'+str(decimal)+'f}'
    d=x.shape[-1]
    x = x.reshape(-1, d)
    return ','.join(('['+','.join([format_s]*d)+']').format(*k) for k in x)

def print_progress(nsample:int,
                   tol:float, 
                   train_tol:float,
                   xmin:np.ndarray):
    print("Status: nsamples={:d}, curr_tol={:.3f} (training curr_tol={:.3f}), xmin={:s}".format(
        nsample, 
        tol,
        train_tol, 
        print_x(xmin)
    ))
     
def vectorize(x:Tensor, multichannel=False):
    """Vectorize data in any shape.

    Args:
        x (torch.Tensor): input data
        multichannel (bool, optional): whether to keep the multiple channels (in the second dimension). Defaults to False.

    Returns:
        torch.Tensor: data of shape (sample_size, dimension) or (sample_size, num_channel, dimension) if multichannel is True.
    """
    if len(x.shape) == 1:
        return x.unsqueeze(1)
    if len(x.shape) == 2:
        return x
    else:
        if not multichannel: # one channel
            return x.reshape(x.shape[0], -1)
        else: # multi-channel
            return x.reshape(x.shape[0], x.shape[1], -1)
        
def generate_seed_according_to_time(n:int=1):    
    # Getting todays date and time using now() of datetime class
    current_date = datetime.datetime.now()

    # Using the strftime() of datetime class
    # which takes the components of date as parameter
    # %Y - year
    # %m - month
    # %d - day
    # %H - Hours
    # %M - Minutes
    # %S - Seconds 
    addon = np.random.choice(50000, size=(n,), replace=False)
    out = int(current_date.strftime("%m%d%H%M%S")) + addon 
    return out.tolist()

def zero_one_denormalization(X_normalized, lower, upper):
    return lower + (upper - lower) * X_normalized

def zero_mean_unit_var_denormalization(X_normalized, mean, std):
    return X_normalized * std + mean

def ignore_criterion(samples_x, new_x=None, threshold:float=0.05, search_domain=None):
    """To decide whether we should ignore/not acquire new_x 
    
    If new_x is not None and there exists (or exist more than) one sample from samples_x, 
    the distance between the two points in every dimension is smaller than 
    the threshold (scaled w.r.t. corresponding search domain width), 
    then new_x should be ignored.

    Arguments
    ################################
    samples_x      : of shape (n_sample, n_dim)
    new_x         : of shape (..., n_dim)
    threshold     : similarity defined w.r.t. search domain being [0,1]
        Threshold values should be defined w.r.t. each i, i=1,...,n_dim. 
        For features that have a wider search domain, 
        the similarity threshold should be larger. 
    search_domain : of shape (n_dim, 2)
    
    Returns
    #################################
    bool: True -- too similar to one of the samples_x, should ignore; 
          else not similar to any one of the samples_x, should Not ignore.
    """
    if new_x is not None:
        scaled_threshold = threshold * (search_domain[:,1] - search_domain[:,0]) # (n_dim,)
        # Whether for all dimensions, each falls into the corresponding threshold
        res = np.all(np.abs(new_x - samples_x) < scaled_threshold, axis=-1) # (n_sample,)
        # Decide if it is True for any of the samples_x
        return np.any(res)
    else:
        return True

def perturb(x:np.ndarray, threshold:float=0.05, search_domain=None, offset:float=0.1, only_one=True):
    """To perturb x on one randomly chosen dimension

    Given every dimension of x fall within the threshold (scaled w.r.t. corresponding 
    search domain width), randomly pick one dimension to perturb, using a right-skewed 
    gamma distribution with density peaking around threshold.

    Arguments
    ################################
    x             : of shape (..., n_dim)
    threshold     : similarity defined w.r.t. search domain being [0,1]
    search_domain : of shape (n_dim, 2)
    offset        : sample values starting at threshold*(1-offset)
    
    Returns
    #################################
    perturbed x on one randomly chosen dimension 
    """
    n_dim = x.shape[-1]
    dtype = x.dtype
     
    gamma_shape = 3 # A smaller shape parameter will increase the right skewness
    gamma_scale = threshold * 0.4 / gamma_shape
    
    if only_one:
        # to perturb one randomly chosen dimension 
        to_perturb = npr.choice(n_dim, (1,)) 
        
    else: 
        
        # every index can be perturbed with 0.5 probability; n_perturb = sum(ind)
        indicator = npr.choice(2, n_dim).astype(bool)
        # make sure at least one dimension is perturbed
        while sum(indicator) < 1:
            indicator = npr.choice(2, n_dim).astype(bool)

        to_perturb = np.where(indicator)[0]

    for i in to_perturb:
        # perturb the target dimensions 
        val = gamma.rvs(a=gamma_shape, scale=gamma_scale) + (1 - offset) * threshold
        # compute the absolute scaled values
        scaled_val = val * (search_domain[i,1] - search_domain[i,0])
        # randomized the sign
        sign = npr.choice(2) * 2 - 1 
        x_perturb_i = x[...,i] + sign * scaled_val 
        while (x_perturb_i < search_domain[i,0]) or (x_perturb_i > search_domain[i,1]): 
            # sample a different perturbed values
            val = gamma.rvs(a=gamma_shape, scale=gamma_scale) + (1 - offset) * threshold
            # compute the absolute scaled values
            scaled_val = val * (search_domain[i,1] - search_domain[i,0])
            # randomized the sign
            sign = npr.choice(2) * 2 - 1 
            x_perturb_i = x[...,i] + sign * scaled_val
        # finally we have the perturbed x falls into the search domain    
        x[...,i] = x_perturb_i
    return x.astype(dtype)

