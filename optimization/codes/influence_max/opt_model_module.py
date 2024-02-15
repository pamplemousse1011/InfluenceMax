import torch  
from jax import vmap, jit, grad, tree_map, random, lax
import jax.numpy as jnp 
from jax.tree_util import Partial 
from jax.flatten_util import ravel_pytree

# !pip install --upgrade -q git+https://github.com/google/flax.git
from flax.core import freeze 
from flax.core.frozen_dict import FrozenDict
from flax import linen as nn

from typing import (
    Any,
    List, 
    Callable,
    Optional,
    Sequence,
    Tuple, 
    Dict
)
PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
 
from codes.utils import generate_seed_according_to_time
from codes.utils import zero_mean_unit_var_denormalization

from codes.influence_max.utils import value_and_jacfwd 

def get_zon_params(search_domain:jnp.ndarray, n_rad:Optional[int]=None) -> Tuple[jnp.ndarray, jnp.ndarray]: 
    """
    For x in range [a, b], we standardize x by 
    (x - a) / (b - a)
    """
    mu    = search_domain[:,0]
    gamma = search_domain[:,1] - search_domain[:,0]  
    return mu, gamma

def get_rbf_params(search_domain:jnp.ndarray, n_rad:int=3) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    We use product sum formulation (not the union sum formulation). 
    This leads to n_rad**d centers in total (n_rad*d for union sum).
    
    Arguments:
    ===================
    n_rad: number of centers in each dimension

    Returns:
    ===================
    mu: (n_rad, d) 
    gamma: (d, )
    """ 
     
    gamma = jnp.stack(
        [(n_rad/(xmax - xmin))**2 for (xmin, xmax) in search_domain], 
        axis=0
    )
    mu = jnp.stack(
        [jnp.linspace(start=xmin+(xmax-xmin)/(10*n_rad), 
                    stop=xmax-(xmax-xmin)/(10*n_rad), 
                    num=n_rad,
                    endpoint=True
        ) for (xmin, xmax) in search_domain], 
        axis=-1
    )
    return mu, gamma

def map_rbf_func(x:jnp.ndarray, mu:jnp.ndarray, gamma:jnp.ndarray) -> jnp.ndarray:
    """ 
    param x: (...,d)
        Feature value
    param mu: (n_rad,d)
        Radial basis centers
    param gamma (d,)
    returns gamma * ||x - mu||^2: (n, d, n_rad) 
    """
    return gamma * (jnp.expand_dims(x, -1) - jnp.expand_dims(mu, 0)) ** 2 # (n, d, n_rad) 

def rbf(x:jnp.ndarray, mu:jnp.ndarray, gamma:jnp.ndarray) -> jnp.ndarray:
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
    out = vmap(map_rbf_func, in_axes=-1, out_axes=1)(x, mu, gamma) # (n, d, n_rad)
    
    # out = vmap(lambda y: jnp.array(list(itertools.product(*y))).sum(-1), in_axes=0)(out) # (n, n_rad**d)
    out = vmap(lambda y: jnp.stack(jnp.meshgrid(*y), -1).sum(-1).reshape(-1), in_axes=0)(out) # (n, n_rad**d)
    
    if x.ndim == 1:
        out = out.reshape(-1)
    return jnp.exp(-out) 

def zon(x:jnp.ndarray, mu:jnp.ndarray, gamma:jnp.ndarray) -> jnp.ndarray:
    """Max-Min standardization

    Arguments:
    ===================
    x: jnp.ndarray (...,d)
        Features to be standardized in each dimension
    mu: jnp.ndarray (d,)
        Min value in each dimension 
    gamma: jnp.ndarray (d,)
        Range in each dimension

    Returns:
    ===================
    x: jnp.ndarray (...,d)
        Standardized features
    """ 
        
    return (x - mu) / gamma

def preprocess(mu, gamma, method='rbf') -> Callable[[jnp.ndarray], jnp.ndarray]:
    func = {
        "none": lambda x:x,
        "rbf": rbf,
        "zon": zon
    }
    return Partial(func[method], mu=mu, gamma=gamma) 

##################################################################################### 

def process_in_batches(fn: Callable[[jnp.ndarray], Any], input:jnp.ndarray, n_batch:int=2, reduction:str="none" ):
    d_input = input.shape[-1]
    input   = input.reshape(-1, d_input)
    n_input = input.shape[0] 
    
    if reduction == "none":
        ret = []
        for batch in jnp.array_split(input, n_batch):
            batch_result = vmap(fn)(batch)  
            ret.append(batch_result)
        ret = jnp.concatenate(ret)
    elif reduction == "sum":
        ret = 0.
        for batch in jnp.array_split(input, n_batch):
            batch_result = vmap(fn)(batch)
            ret += batch_result.sum(0)
    elif reduction == "mean":
        ret = 0.
        for batch in jnp.array_split(input, n_batch):
            batch_result = vmap(fn)(batch)
            ret += batch_result.sum(0)
        ret /= n_input
    else:
        raise ValueError(f"Received reduction={reduction}. Only accepted 'none', 'sum' or 'mean'.")
    
    return ret
        
def compute_ens_from_embedding(x:jnp.ndarray, MLPs:nn.Module, stochastic:bool, resample_size:int) -> jnp.ndarray:
    """
    x: embedding (d,)
    
    if stochastic:
    - First sample new response data, x_rep, by creating identical copies, 
        of shape (n_base * resample_size, n_dim). 
    - Then pass the response data into model and obtain samples, where
        samples[n_base*(i-1):n_sample*i,:] contains one sample for each data point, 
        for i = 1, ..., resample_size
    """
    n_model = len(MLPs)

    if stochastic > 0:
        ## Sample new response data
        emb_rep = jnp.tile(x, (resample_size, 1))                    # (resample_size, d)

        ## Pass into model and average over n_model
        output = jnp.stack(
            [MLPs[str(ii)](emb_rep) 
            for ii in range(n_model)],
            axis = 0
        ).mean(0)                                                    # (resample_size, out_dim=1)
    
    else: 
        output = jnp.stack(
            [MLPs[str(ii)](x) 
            for ii in range(n_model)],
            axis = 0
        ).mean(0)                                                    # (1, out_dim=1) 
        
    ## Average the resamples
    return output.mean(-2).squeeze(-1)                               # (,)

class StoJBlock(nn.Module):
    feature       : int
    no_batch_norm : bool = False
    n_noise       : int = 100
    dtype         : Dtype = jnp.float32
    noise_init    : Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal()
    key           : PRNGKey = random.PRNGKey(42)
 
    @nn.compact    
    def __call__(self, x:jnp.ndarray):
        x = x.reshape(-1, x.shape[-1])
        if self.n_noise > 0:
            eps = self.noise_init(
                self.key, 
                (x.shape[0], self.n_noise), 
                dtype = self.dtype
            ) 
            x = jnp.concatenate([x, eps], axis=-1)
        x = nn.Dense(self.feature, name='Dense')(x)
        if not self.no_batch_norm:
            x = nn.BatchNorm(use_running_average=True, name='BatchNorm')(x)
        x = nn.silu(x)
        return x

class StoJMLPBatch(nn.Module):
    n_hidden             : Sequence[int]
    latent_embedding_fn : Callable[[jnp.ndarray], jnp.ndarray]
    ymean                : float = 0.
    ystd                 : float = 1.
    no_batch_norm        : bool  = False
    n_noise              : int   = 100
    noise_std            : float = 1.0
    resample_size        : int   = 100 
    dtype                : Dtype = jnp.float32 
    key                  : int   = generate_seed_according_to_time(1)[0]
    
    def setup(self): 
        key_chain  = random.split(random.PRNGKey(self.key), len(self.n_hidden)) 
        noise_init = (nn.initializers.normal(stddev=self.noise_std, dtype=self.dtype) 
                      if self.n_noise > 0 else None)
        Layers = []
        for i in range(len(self.n_hidden)):
            Layers.append(
                StoJBlock(
                    self.n_hidden[i], 
                    self.no_batch_norm,
                    self.n_noise, 
                    self.dtype, 
                    noise_init, 
                    key_chain[i])
            )
            
        self.featurizer = nn.Sequential(Layers)
        self.targetizer = nn.Dense(1)
    
    def compute_from_embedding(self, base_x_embedding:jnp.ndarray, latent_embedding:jnp.ndarray) -> jnp.ndarray:
        """
        base_x_embedding: (d_1,)
        latent_embedding: (d_2,)
        """
        # Concatenate the embeddings
        emb_rep = jnp.concatenate([base_x_embedding, latent_embedding], axis=-1)    # (d_1+d_2,)

        if self.n_noise > 0:
            ## Sample new response data
            emb_rep = jnp.tile(emb_rep, (self.resample_size, 1))                    # (resample_size, d_1+d_2)
             
            ## Pass into model 
            emb_rep = self.featurizer(emb_rep)
            output  = self.targetizer(emb_rep)                                      # (resample_size, out_dim=1)
        else: 
            emb_rep = self.featurizer(emb_rep)                                      # (1, out_dim=1)
            output  = self.targetizer(emb_rep)                                      # (1, out_dim=1)
        
        ## Average the resamples
        output = output.mean(-2).squeeze(-1)                                       # (,)
 
        return output
    
    def __call__(self, base_x_embedding:jnp.ndarray, x:jnp.ndarray, n_batch=1, reduction="mean"):
        """
        INPUT 
        base_x: (..., n_base_dim)  
        x : (n_dim,) or (1, n_dim)

        RETURN
        (n_base,) if individual else ()

        - First sample new response data, x_rep, by creating identical copies, 
          of shape (n_base * resample_size, n_dim). 
        - Then pass the response data into model and obtain samples, where
          samples[n_base*(i-1):n_sample*i,:] contains one sample for each data point, 
          for i = 1, ..., resample_size
        """ 

        ## Get x embeddings, (d_2,)
        latent_embedding = self.latent_embedding_fn(x.reshape(-1))    
        
        ## Get model output ()
        output = process_in_batches(
            Partial(self.compute_from_embedding, latent_embedding=latent_embedding), 
            base_x_embedding, 
            n_batch=n_batch, 
            reduction=reduction
        )
            
        ## Denormalize
        output = zero_mean_unit_var_denormalization(output, self.ymean, self.ystd)
        return output

class StoJMLPSingle(nn.Module):
    n_hidden             : Sequence[int]
    latent_embedding_fn : Callable[[jnp.ndarray], jnp.ndarray]
    ymean                : float = 0.
    ystd                 : float = 1.
    no_batch_norm        : bool  = False
    n_noise              : int   = 100
    noise_std            : float = 1.0
    resample_size        : int   = 100 
    dtype                : Dtype = jnp.float32 
    key                  : int   = generate_seed_according_to_time(1)[0]
    
    def setup(self): 
        key_chain  = random.split(random.PRNGKey(self.key), len(self.n_hidden)) 
        noise_init = (nn.initializers.normal(stddev=self.noise_std, dtype=self.dtype) 
                      if self.n_noise > 0 else None)
        Layers = []
        for i in range(len(self.n_hidden)):
            Layers.append(
                StoJBlock(
                    self.n_hidden[i], 
                    self.no_batch_norm,
                    self.n_noise, 
                    self.dtype, 
                    noise_init, 
                    key_chain[i])
            )
            
        self.featurizer = nn.Sequential(Layers)
        self.targetizer = nn.Dense(1)
    
    def compute_from_embedding(self, base_x_embedding:jnp.ndarray, latent_embedding:jnp.ndarray) -> jnp.ndarray:
        """
        base_x_embedding: (d_1,)
        latent_embedding: (d_2,)
        """
        # Concatenate the embeddings
        emb_rep = jnp.concatenate([base_x_embedding, latent_embedding], axis=-1)    # (d_1+d_2,)

        if self.n_noise > 0:
            ## Sample new response data
            emb_rep = jnp.tile(emb_rep, (self.resample_size, 1))                    # (resample_size, d_1+d_2)
             
            ## Pass into model 
            emb_rep = self.featurizer(emb_rep)
            output  = self.targetizer(emb_rep)                                      # (resample_size, out_dim=1)
        else: 
            emb_rep = self.featurizer(emb_rep)                                      # (1, out_dim=1)
            output  = self.targetizer(emb_rep)                                      # (1, out_dim=1)
        
        ## Average the resamples
        output = output.mean(-2).squeeze(-1)                                       # (,)
 
        return output
    
    def __call__(self, base_x_embedding:jnp.ndarray, x:jnp.ndarray):
        """
        INPUT 
        base_x: (..., n_base_dim)  
        x : (n_dim,) or (1, n_dim)

        RETURN
        (n_base,) if individual else ()

        - First sample new response data, x_rep, by creating identical copies, 
          of shape (n_base * resample_size, n_dim). 
        - Then pass the response data into model and obtain samples, where
          samples[n_base*(i-1):n_sample*i,:] contains one sample for each data point, 
          for i = 1, ..., resample_size
        """ 

        ## Get x embeddings, (d_2,)
        latent_embedding = self.latent_embedding_fn(x.reshape(-1))    
        
        ## Get model output ()
        output = self.compute_from_embedding(base_x_embedding, latent_embedding=latent_embedding) 
        
        ## Denormalize
        output = zero_mean_unit_var_denormalization(output, self.ymean, self.ystd)
        return output

class StoJMLPwBase(nn.Module):
    n_hidden             : Sequence[int]
    latent_embedding_fn : Callable[[jnp.ndarray], jnp.ndarray]
    base_x_embedding     : jnp.ndarray
    ymean                : float = 0.
    ystd                 : float = 1.
    no_batch_norm        : bool  = False
    n_noise              : int   = 100
    noise_std            : float = 1.0
    resample_size        : int   = 100
    dtype                : Dtype = jnp.float32 
    key                  : int   = generate_seed_according_to_time(1)[0]
    
    def setup(self): 
        key_chain  = random.split(random.PRNGKey(self.key), len(self.n_hidden)) 
        noise_init = (nn.initializers.normal(stddev=self.noise_std, dtype=self.dtype) 
                      if self.n_noise > 0 else None)
        Layers = []
        for i in range(len(self.n_hidden)):
            Layers.append(
                StoJBlock(
                    self.n_hidden[i], 
                    self.no_batch_norm,
                    self.n_noise, 
                    self.dtype, 
                    noise_init, 
                    key_chain[i])
            )
            
        self.featurizer = nn.Sequential(Layers)
        self.targetizer = nn.Dense(1)

    def compute_from_embedding(self, base_x_embedding:jnp.ndarray, latent_embedding:jnp.ndarray) -> jnp.ndarray:
        """
        base_x_embedding: (d_1,)
        latent_embedding: (d_2,)
        """
        # Concatenate the embeddings
        emb_rep = jnp.concatenate([base_x_embedding, latent_embedding], axis=-1)    # (d_1+d_2,)

        if self.n_noise > 0:
            ## Sample new response data
            emb_rep = jnp.tile(emb_rep, (self.resample_size, 1))                    # (resample_size, d_1+d_2)
             
            ## Pass into model 
            emb_rep = self.featurizer(emb_rep)
            output  = self.targetizer(emb_rep)                                      # (resample_size, out_dim=1)
        else: 
            emb_rep = self.featurizer(emb_rep)                                      # (1, out_dim=1)
            output  = self.targetizer(emb_rep)                                      # (1, out_dim=1)
        
        ## Average the resamples
        output = output.mean(-2).squeeze(-1)                                       # (,)
 
        return output
    
    def __call__(self, x:jnp.ndarray, n_batch:int=1, reduction="mean"):
        """
        INPUT 
        base_x: (..., n_base_dim) 
        x : (n_dim,) or (1, n_dim)

        RETURN
        (n_base,) if individual else (,)

        - First sample new response data, x_rep, by creating identical copies, 
          of shape (n_base * resample_size, n_dim). 
        - Then pass the response data into model and obtain samples, where
          samples[n_base*(i-1):n_sample*i,:] contains one sample for each data point, 
          for i = 1, ..., resample_size
        """ 

        ## Get x embeddings, (d_2,)
        latent_embedding = self.latent_embedding_fn(x.reshape(-1))    
    
        ## Get model output (n_base,) or ()
        output = process_in_batches(
            fn=Partial(self.compute_from_embedding, latent_embedding=latent_embedding), 
            input=self.base_x_embedding.reshape(-1, self.base_x_embedding.shape[-1]), 
            n_batch=n_batch,
            reduction=reduction
        )
        
        ## Denormalize
        output = zero_mean_unit_var_denormalization(output, self.ymean, self.ystd)

        return output

class StoJSIN(nn.Module):
    n_hidden      : Sequence[int]
    no_batch_norm : bool  = False
    n_noise       : int   = 100
    noise_std     : float = 1.0 
    dtype         : Dtype = jnp.float32
    key           : int   = generate_seed_according_to_time(1)[0]
    
    def setup(self): 
        key_chain  = random.split(random.PRNGKey(self.key), len(self.n_hidden)) 
        
        noise_init = (nn.initializers.normal(stddev=self.noise_std, dtype=self.dtype)
                      if self.n_noise > 0 else None)
        
        Layers = []
        for i in range(len(self.n_hidden)):
            Layers.append(
                StoJBlock(
                    self.n_hidden[i], 
                    self.no_batch_norm,
                    self.n_noise,  
                    self.dtype, 
                    noise_init, 
                    key_chain[i])
            )
            
        self.featurizer = nn.Sequential(Layers)
        self.targetizer = nn.Dense(1)

    def __call__(self, x:jnp.ndarray):
        """
        INPUT 
        x : transformed x (m, d)

        RETURN
        (m,)
        """
        x = self.featurizer(x)
        return self.targetizer(x)
    
class StoJENSSingle(nn.Module):
    n_model              : int 
    n_hidden             : Sequence[int] 
    latent_embedding_fn : Callable[[jnp.ndarray], jnp.ndarray] 
    ymean                : float = 0.
    ystd                 : float = 1.
    no_batch_norm        : bool  = False
    n_noise              : int   = 100
    noise_std            : float = 1.0
    resample_size        : int   = 100
    dtype                : Dtype = jnp.float32 

    def setup(self): 
        keys  = generate_seed_according_to_time(self.n_model)
        modulelist = {} 
        for ii in range(self.n_model):
            modulelist[str(ii)] = StoJSIN(
                n_hidden        = self.n_hidden, 
                no_batch_norm   = self.no_batch_norm,
                n_noise         = self.n_noise,
                noise_std       = self.noise_std, 
                dtype           = self.dtype,
                key             = keys[ii]
            ) 
        self.MLPs = modulelist
    
     
    def __call__(self, base_x_embedding:jnp.ndarray, x:jnp.ndarray):
        """
        INPUT 
        base_x: (d_base,) or (1, d_base) 
        x : (d,) or (1, d)

        RETURN
        (,)
        """
        ## Get x embeddings, (d_2,)
        latent_embedding = self.latent_embedding_fn(x.reshape(-1))            
        # Concatenate the embeddings
        emb = jnp.concatenate([base_x_embedding.reshape(-1), latent_embedding], axis=-1)    # (d_1+d_2,)
        ## Get model output, (,)
        output = compute_ens_from_embedding(
            emb, 
            self.MLPs,
            (self.n_noise>0), 
            self.resample_size 
        )

        ## Denormalize
        output = zero_mean_unit_var_denormalization(output, self.ymean, self.ystd)

        return output
   
class StoJJACSingle(nn.Module):
    """Compute bias-corrected jackknife estimate of μ
    μ_{all,j}   : Esimate of μ with all samples
    μ_{loo,(i)} : LEAVE-ONE-OUT estimate of μ without i-th sample 
    The bias-corrected jackknife estimate of μ
        μ{Jack} = n/m Σ_{j=1}^m μ_{all,j} - (n-1)/n Σ_{i=1}^n μ_{loo,(i)}   
    We compute the following
        term1 = 1/m * Σ_{j=1}^m μ_{all,j}
        term2 = 1/n * Σ_{i=1}^n μ_{loo,(i)} 
    return 
        n*term1 - (n-1)*term2
    """
    n_model_all          : int 
    n_model_loo          : int
    n_hidden             : Sequence[int] 
    latent_embedding_fn : Callable[[jnp.ndarray], jnp.ndarray]
    base_x_embedding     : jnp.ndarray
    ymean                : float = 0.
    ystd                 : float = 1.
    no_batch_norm        : bool  = False
    n_noise              : int   = 100
    noise_std            : float = 1.0
    resample_size        : int   = 100
    dtype                : Dtype = jnp.float32 

    def setup(self): 
        self.n_model = self.n_model_all+self.n_model_loo
        keys  = generate_seed_according_to_time(self.n_model)
        modulelist = {} 
        for ii in range(self.n_model):
            modulelist[str(ii)] = StoJSIN(
                n_hidden        = self.n_hidden, 
                no_batch_norm   = self.no_batch_norm, 
                n_noise         = self.n_noise,
                noise_std       = self.noise_std, 
                dtype           = self.dtype,
                key             = keys[ii]
            ) 
        self.MLPs = modulelist

    def compute_each_base_x(self, base_x_embedding:jnp.ndarray, latent_embedding:jnp.ndarray) -> jnp.ndarray:
        """
        base_x_embedding: (d_1,)
        latent_embedding: (d_2,)
        """
        # Concatenate the embeddings
        emb = jnp.concatenate([base_x_embedding, latent_embedding], axis=-1)    # (d_1+d_2,)
        output = compute_ens_from_embedding(emb, self.MLPs, (self.n_noise>0), self.resample_size)
        
        ## Compute bias-corrected Jackknife estimate
        term1 = output[:self.n_model_all].mean(0)
        term2 = output[self.n_model_all:].mean(0)
        output = self.n_model_loo*term1 - (self.n_model_loo-1)*term2
        
        return output
       
      
    def __call__(self, base_x_embedding:jnp.ndarray, x:jnp.ndarray):
        """
        INPUT 
        base_x: (d_base,) or (1, d_base) 
        x : (d,) or (1, d)

        RETURN
        (,)
        """
        ## Get x embeddings, (..., d_2)
        latent_embedding = self.latent_embedding_fn(x.reshape(-1))            
         
        ## Get model output
        output = self.compute_each_base_x(base_x_embedding.reshape(-1), latent_embedding)

        ## Denormalize
        output = zero_mean_unit_var_denormalization(output, self.ymean, self.ystd)

        return output

class StoJMLP(nn.Module):
    n_hidden             : Sequence[int]
    latent_embedding_fn : Callable[[jnp.ndarray], jnp.ndarray]
    ymean                : float = 0.
    ystd                 : float = 1.
    no_batch_norm        : bool  = False
    n_noise              : int   = 100
    noise_std            : float = 1.0
    resample_size        : int   = 100 
    dtype                : Dtype = jnp.float32 
    key                  : int   = generate_seed_according_to_time(1)[0]
    
    def setup(self): 
        key_chain  = random.split(random.PRNGKey(self.key), len(self.n_hidden)) 
        noise_init = (nn.initializers.normal(stddev=self.noise_std, dtype=self.dtype) 
                      if self.n_noise > 0 else None)
        Layers = []
        for i in range(len(self.n_hidden)):
            Layers.append(
                StoJBlock(
                    self.n_hidden[i], 
                    self.no_batch_norm,
                    self.n_noise, 
                    self.dtype, 
                    noise_init, 
                    key_chain[i])
            )
            
        self.featurizer = nn.Sequential(Layers)
        self.targetizer = nn.Dense(1)
    
    def compute_output(self, x:jnp.ndarray) -> jnp.ndarray:
        """
        x: (d,) 
        """
        if self.n_noise > 0:
            ## Sample new response data
            emb_rep = jnp.tile(x, (self.resample_size, 1))                    # (resample_size, d_1+d_2)
             
            ## Pass into model 
            emb_rep = self.featurizer(emb_rep)
            output  = self.targetizer(emb_rep)                                      # (resample_size, out_dim=1)
        else: 
            emb_rep = self.featurizer(x)                                      # (1, out_dim=1)
            output  = self.targetizer(emb_rep)                                      # (1, out_dim=1)
        
        ## Average the resamples
        output = output.mean(-2).squeeze(-1)                                       # (,)
 
        return output
    
    def __call__(self, x:jnp.ndarray):
        """
        INPUT 
        x : (n_dim,) or (1, n_dim)

        RETURN
        (,)

        - First sample new response data, x_rep, by creating identical copies, 
          of shape (n_base * resample_size, n_dim). 
        - Then pass the response data into model and obtain samples, where
          samples[n_base*(i-1):n_sample*i,:] contains one sample for each data point, 
          for i = 1, ..., resample_size
        """ 

        ## Get x embeddings, (d_2,)
        emb = self.latent_embedding_fn(x.reshape(-1))    
        
        ## Get model output ()
        output = self.compute_output(emb) 
        
        ## Denormalize
        output = zero_mean_unit_var_denormalization(output, self.ymean, self.ystd)
        return output

class StoJENS(nn.Module):
    n_model              : int 
    n_hidden             : Sequence[int] 
    latent_embedding_fn : Callable[[jnp.ndarray], jnp.ndarray] 
    ymean                : float = 0.
    ystd                 : float = 1.
    no_batch_norm        : bool  = False
    n_noise              : int   = 100
    noise_std            : float = 1.0
    resample_size        : int   = 100
    dtype                : Dtype = jnp.float32 

    def setup(self): 
        keys  = generate_seed_according_to_time(self.n_model)
        modulelist = {} 
        for ii in range(self.n_model):
            modulelist[str(ii)] = StoJSIN(
                n_hidden        = self.n_hidden, 
                no_batch_norm   = self.no_batch_norm,
                n_noise         = self.n_noise,
                noise_std       = self.noise_std, 
                dtype           = self.dtype,
                key             = keys[ii]
            ) 
        self.MLPs = modulelist
    
    def __call__(self, x:jnp.ndarray):
        """
        INPUT  
        x : (d,) or (1, d)

        RETURN
        (,)
        """

        ## Get x embeddings, (d_2,)
        emb = self.latent_embedding_fn(x.reshape(-1))  

        ## Get model output, (,)
        output = compute_ens_from_embedding(
            emb, 
            self.MLPs,
            (self.n_noise>0), 
            self.resample_size 
        )

        ## Denormalize
        output = zero_mean_unit_var_denormalization(output, self.ymean, self.ystd)

        return output
   
class StoJJAC(nn.Module):
    """Compute bias-corrected jackknife estimate of μ
    μ_{all,j}   : Esimate of μ with all samples
    μ_{loo,(i)} : LEAVE-ONE-OUT estimate of μ without i-th sample 
    The bias-corrected jackknife estimate of μ
        μ{Jack} = n/m Σ_{j=1}^m μ_{all,j} - (n-1)/n Σ_{i=1}^n μ_{loo,(i)}   
    We compute the following
        term1 = 1/m * Σ_{j=1}^m μ_{all,j}
        term2 = 1/n * Σ_{i=1}^n μ_{loo,(i)} 
    return 
        n*term1 - (n-1)*term2
    """
    n_model_all          : int 
    n_model_loo          : int
    n_hidden             : Sequence[int] 
    latent_embedding_fn : Callable[[jnp.ndarray], jnp.ndarray]
    base_x_embedding     : jnp.ndarray
    ymean                : float = 0.
    ystd                 : float = 1.
    no_batch_norm        : bool  = False
    n_noise              : int   = 100
    noise_std            : float = 1.0
    resample_size        : int   = 100
    dtype                : Dtype = jnp.float32 

    def setup(self): 
        self.n_model = self.n_model_all+self.n_model_loo
        keys  = generate_seed_according_to_time(self.n_model)
        modulelist = {} 
        for ii in range(self.n_model):
            modulelist[str(ii)] = StoJSIN(
                n_hidden        = self.n_hidden, 
                no_batch_norm   = self.no_batch_norm, 
                n_noise         = self.n_noise,
                noise_std       = self.noise_std, 
                dtype           = self.dtype,
                key             = keys[ii]
            ) 
        self.MLPs = modulelist

    def compute_output(self, emb: jnp.ndarray) -> jnp.ndarray:
        """
        emb: (d,)
        """
        # Concatenate the embeddings
        output = compute_ens_from_embedding(emb, self.MLPs, (self.n_noise>0), self.resample_size)
        
        ## Compute bias-corrected Jackknife estimate
        term1 = output[:self.n_model_all].mean(0)
        term2 = output[self.n_model_all:].mean(0)
        output = self.n_model_loo*term1 - (self.n_model_loo-1)*term2
        
        return output
       
      
    def __call__(self, x:jnp.ndarray):
        """
        INPUT  
        x : (d,) or (1, d)

        RETURN
        (,)
        """
        ## Get x embeddings 
        emb = self.latent_embedding_fn(x.reshape(-1))            
         
        ## Get model output
        output = self.compute_output(emb)

        ## Denormalize
        output = zero_mean_unit_var_denormalization(output, self.ymean, self.ystd)

        return output

class RntJBlock(nn.Module): 
    feature       : int
    dtype         : Dtype = jnp.float32
    
    def setup(self):
        self.layer = nn.Sequential(
            [nn.Dense(self.feature, name='Dense_1'),
            nn.BatchNorm(use_running_average=True, name='BatchNorm_1'),
            nn.silu,
            nn.Dense(self.feature, name='Dense_2'),
            nn.BatchNorm(use_running_average=True, name='BatchNorm_2')]
        )
        self.residual = nn.Dense(self.feature, name='Dense_3')

    def __call__(self, x:jnp.ndarray):
        identity = self.residual(x) 
        out = self.layer(x) 
        out += identity
        out = nn.silu(out)
        return out
    
class RntJMLPSingle(nn.Module):
    n_hidden             : Sequence[int]
    latent_embedding_fn  : Callable[[jnp.ndarray], jnp.ndarray]
    ymean                : float = 0.
    ystd                 : float = 1.
    dtype                : Dtype = jnp.float32 
    no_batch_norm        : bool  = False
    n_noise              : int   = 100
    noise_std            : float = 1.0
    resample_size        : int   = 100  
    key                  : int   = generate_seed_according_to_time(1)[0]
    
    def setup(self): 
        Layers = []
        for i in range(len(self.n_hidden)):
            Layers.append(RntJBlock(self.n_hidden[i], self.dtype))
            
        self.featurizer = nn.Sequential(Layers)
        self.targetizer = nn.Dense(1)
    
    def compute_from_embedding(self, base_x_embedding:jnp.ndarray, latent_embedding:jnp.ndarray) -> jnp.ndarray:
        """
        base_x_embedding: (d_1,)
        latent_embedding: (d_2,)
        """
        # Concatenate the embeddings
        emb_rep = jnp.concatenate([base_x_embedding, latent_embedding], axis=-1)  # (..., d_1+d_2)

        emb_rep = self.featurizer(emb_rep)                                        # (..., out_dim=1)
        output  = self.targetizer(emb_rep)                                        # (..., out_dim=1)
        ## Average the resamples
        output = output.squeeze(-1)                                               # (...,)
 
        return output
    
    def __call__(self, base_x_embedding:jnp.ndarray, x:jnp.ndarray):
        """
        INPUT 
        base_x: (..., n_base_dim)  
        x : (n_dim,) or (1, n_dim)

        RETURN
        (n_base,) if individual else ()

        - First sample new response data, x_rep, by creating identical copies, 
          of shape (n_base * resample_size, n_dim). 
        - Then pass the response data into model and obtain samples, where
          samples[n_base*(i-1):n_sample*i,:] contains one sample for each data point, 
          for i = 1, ..., resample_size
        """ 

        ## Get x embeddings, (d_2,)
        latent_embedding = self.latent_embedding_fn(x.reshape(-1))    
        
        ## Get model output ()
        output = self.compute_from_embedding(base_x_embedding, latent_embedding=latent_embedding) 
        
        ## Denormalize
        output = zero_mean_unit_var_denormalization(output, self.ymean, self.ystd)
        return output

class RntJMLPBatch(nn.Module):
    n_hidden             : Sequence[int]
    latent_embedding_fn : Callable[[jnp.ndarray], jnp.ndarray]
    ymean                : float = 0.
    ystd                 : float = 1.
    dtype                : Dtype = jnp.float32 
    no_batch_norm        : bool  = False
    n_noise              : int   = 100
    noise_std            : float = 1.0
    resample_size        : int   = 100  
    key                  : int   = generate_seed_according_to_time(1)[0]

    def setup(self): 
        Layers = []
        for i in range(len(self.n_hidden)):
            Layers.append(RntJBlock(self.n_hidden[i], self.dtype))
            
        self.featurizer = nn.Sequential(Layers)
        self.targetizer = nn.Dense(1)
    
    def compute_from_embedding(self, base_x_embedding:jnp.ndarray, latent_embedding:jnp.ndarray) -> jnp.ndarray:
        """
        base_x_embedding: (d_1,)
        latent_embedding: (d_2,)
        """
        # Concatenate the embeddings
        emb_rep = jnp.concatenate([base_x_embedding, latent_embedding], axis=-1)    # (d_1+d_2,)

        emb_rep = self.featurizer(emb_rep)                                      # (1, out_dim=1)
        output  = self.targetizer(emb_rep)                                      # (1, out_dim=1)
        
        ## Average the resamples
        output = output.squeeze(-1)                                       # (,)
 
        return output
    
    def __call__(self, base_x_embedding:jnp.ndarray, x:jnp.ndarray, n_batch=1, reduction="mean"):
        """
        INPUT 
        base_x: (..., n_base_dim)  
        x : (n_dim,) or (1, n_dim)

        RETURN
        (n_base,) if individual else ()

        - First sample new response data, x_rep, by creating identical copies, 
          of shape (n_base * resample_size, n_dim). 
        - Then pass the response data into model and obtain samples, where
          samples[n_base*(i-1):n_sample*i,:] contains one sample for each data point, 
          for i = 1, ..., resample_size
        """ 

        ## Get x embeddings, (d_2,)
        latent_embedding = self.latent_embedding_fn(x.reshape(-1))    
        
        ## Get model output ()
        output = process_in_batches(
            Partial(self.compute_from_embedding, latent_embedding=latent_embedding), 
            base_x_embedding, 
            n_batch=n_batch, 
            reduction=reduction
        )
            
        ## Denormalize
        output = zero_mean_unit_var_denormalization(output, self.ymean, self.ystd)
        return output

class RntJSIN(nn.Module):
    n_hidden      : Sequence[int] 
    dtype         : Dtype = jnp.float32 
    
    def setup(self): 
         
        Layers = []
        for i in range(len(self.n_hidden)):
            Layers.append(
                RntJBlock(
                    self.n_hidden[i],  
                    self.dtype)
            )
            
        self.featurizer = nn.Sequential(Layers)
        self.targetizer = nn.Dense(1)

    def __call__(self, x:jnp.ndarray):
        """
        INPUT 
        x : transformed x (m, d)

        RETURN
        (m,)
        """
        x = self.featurizer(x)
        return self.targetizer(x)
    
class RntJMLP(nn.Module):
    n_hidden             : Sequence[int]
    latent_embedding_fn  : Callable[[jnp.ndarray], jnp.ndarray]
    ymean                : float = 0.
    ystd                 : float = 1.
    dtype                : Dtype = jnp.float32 
    no_batch_norm        : bool  = False
    n_noise              : int   = 100
    noise_std            : float = 1.0
    resample_size        : int   = 100  
    key                  : int   = generate_seed_according_to_time(1)[0]
    
    def setup(self): 
        Layers = []
        for i in range(len(self.n_hidden)):
            Layers.append(RntJBlock(self.n_hidden[i], self.dtype))
            
        self.featurizer = nn.Sequential(Layers)
        self.targetizer = nn.Dense(1)
    
    def compute_output(self, x:jnp.ndarray) -> jnp.ndarray:
        """
        x: (d,) 
        """
        
        emb_rep = self.featurizer(x)            # (..., out_dim=1)
        output  = self.targetizer(emb_rep)      # (..., out_dim=1)
        ## Average the resamples
        output = output.squeeze(-1)             # (...,)
 
        return output
    
    def __call__(self, x:jnp.ndarray):
        """
        INPUT  
        x : (n_dim,) or (1, n_dim)

        RETURN
        (,)

        - First sample new response data, x_rep, by creating identical copies, 
          of shape (n_base * resample_size, n_dim). 
        - Then pass the response data into model and obtain samples, where
          samples[n_base*(i-1):n_sample*i,:] contains one sample for each data point, 
          for i = 1, ..., resample_size
        """ 

        ## Get x embeddings, (d_2,)
        latent_embedding = self.latent_embedding_fn(x.reshape(-1))    
        
        ## Get model output ()
        output = self.compute_output(latent_embedding) 
        
        ## Denormalize
        output = zero_mean_unit_var_denormalization(output, self.ymean, self.ystd)
        return output

class RntJENS(nn.Module):
    n_model              : int 
    n_hidden             : Sequence[int] 
    latent_embedding_fn : Callable[[jnp.ndarray], jnp.ndarray] 
    ymean                : float = 0.
    ystd                 : float = 1.
    no_batch_norm        : bool  = False
    n_noise              : int   = 100
    noise_std            : float = 1.0
    resample_size        : int   = 100
    dtype                : Dtype = jnp.float32 
    
    def setup(self): 
        modulelist = {} 
        for ii in range(self.n_model):
            modulelist[str(ii)] = RntJSIN(
                n_hidden        = self.n_hidden,  
                dtype           = self.dtype
            ) 
        self.MLPs = modulelist
    
    def __call__(self, x:jnp.ndarray):
        """
        INPUT  
        x : (d,) or (1, d)

        RETURN
        (,)
        """

        ## Get x embeddings, (d_2,)
        emb = self.latent_embedding_fn(x.reshape(-1))  

        ## Get model output, (,)
        output = compute_ens_from_embedding(
            emb, 
            self.MLPs,
            (self.n_noise>0), 
            self.resample_size 
        )

        ## Denormalize
        output = zero_mean_unit_var_denormalization(output, self.ymean, self.ystd)

        return output

def sto_parameter_reconstruct(nets: torch.nn.ModuleList) -> FrozenDict:
    output = {'params':{}, 'batch_stats':{}} # flax.core.frozen_dict.unfreeze(model_params) # 
    for i in range(len(nets)):
        group_name = '_'.join(['MLP', str(i)])
        output['params'][group_name] = {'featurizer':{}, 'targetizer':{}}
        output['batch_stats'][group_name] = {'featurizer':{}}
        
        # the last layer is the target layer
        n_layer = len(nets[i])
        for j in range(n_layer-1):
            layer_name = '_'.join(['layers', str(j)])

            for layer_fn in nets[i][j].layer:
                if isinstance(layer_fn, torch.nn.Linear):
                    output['params'][group_name]['featurizer'][layer_name] = {
                        'Dense': {
                            'kernel': jnp.transpose(jnp.array(layer_fn.weight.detach().cpu()), (1, 0)),
                            'bias'  : jnp.array(layer_fn.bias.detach().cpu())
                        }
                    }
                elif isinstance(layer_fn, torch.nn.BatchNorm1d):
                    output['params'][group_name]['featurizer'][layer_name]['BatchNorm'] = {
                        'bias'  : jnp.array(layer_fn.bias.detach().cpu()),
                        'scale' : jnp.array(layer_fn.weight.detach().cpu()),
                    }
                    output['batch_stats'][group_name]['featurizer'][layer_name] = {
                        'BatchNorm' : {
                            'mean'  : jnp.array(layer_fn.running_mean.detach().cpu()),
                            'var'   : jnp.array(layer_fn.running_var.detach().cpu()),
                        },
                    }
                elif isinstance(layer_fn, torch.nn.SiLU):
                    pass 
                else:
                    raise ValueError(f"Unidentified layer. Received {layer_fn} while expecting torch.nn.Linear, torch.nn.BatchNorm1d or torch.nn.SiLU.")
        
        output['params'][group_name]['targetizer'] = {
            'kernel'  : jnp.transpose(jnp.array(nets[i][-1].weight.detach().cpu()), (1, 0)),
            'bias'    : jnp.array(nets[i][-1].bias.detach().cpu())
        }

    return freeze(output)

def rnt_parameter_reconstruct(nets: torch.nn.ModuleList) -> FrozenDict:
    output = {'params':{}, 'batch_stats':{}} # flax.core.frozen_dict.unfreeze(model_params) # 
    for i in range(len(nets)):
        group_name = '_'.join(['MLP', str(i)])
        output['params'][group_name] = {'featurizer':{}, 'targetizer':{}}
        output['batch_stats'][group_name] = {'featurizer':{}}
        
        n_layer = len(nets[i])
        
        k = 0
        # the last layer is the target layer
        for j in range(n_layer-1):
            if not isinstance(nets[i][j], torch.nn.Dropout):
                layer_name = '_'.join(['layers', str(k)])
                output['params'][group_name]['featurizer'][layer_name] = {}
                output['batch_stats'][group_name]['featurizer'][layer_name] = {}

                # add the fully connected layers in the resnet block 
                for dictname in ['Dense_1', 'Dense_2']:
                    output['params'][group_name]['featurizer'][layer_name][dictname] = {
                        'kernel': jnp.transpose(jnp.array(nets[i][j].layer._modules[dictname].weight.detach().cpu()), (1, 0)),
                        'bias'  : jnp.array(nets[i][j].layer._modules[dictname].bias.detach().cpu())
                    } 
                 
                # add the batchnorm layers in the resnet block 
                for dictname in ['BatchNorm_1', 'BatchNorm_2']:
                    output['params'][group_name]['featurizer'][layer_name][dictname] = {
                        'bias'  : jnp.array(nets[i][j].layer._modules[dictname].bias.detach().cpu()),
                        'scale' : jnp.array(nets[i][j].layer._modules[dictname].weight.detach().cpu()),
                    } 
                    output['batch_stats'][group_name]['featurizer'][layer_name][dictname] = {
                        'mean'  : jnp.array(nets[i][j].layer._modules[dictname].running_mean.detach().cpu()),
                        'var'   : jnp.array(nets[i][j].layer._modules[dictname].running_var.detach().cpu()),
                    }

                # add the residual layer in the resnet block 
                if  isinstance(nets[i][j].residual, torch.nn.Identity):
                    output['params'][group_name]['featurizer'][layer_name]['Dense_3'] = {
                        'kernel': jnp.ones_like(output['params'][group_name]['featurizer'][layer_name]['Dense_1']['kernel']),
                        'bias'  : jnp.zeros_like(output['params'][group_name]['featurizer'][layer_name]['Dense_1']['bias']),
                    } 
                else:
                    output['params'][group_name]['featurizer'][layer_name]['Dense_3'] = {
                        'kernel': jnp.transpose(jnp.array(nets[i][j].residual.weight.detach().cpu()), (1, 0)),
                        'bias'  : jnp.array(nets[i][j].residual.bias.detach().cpu())
                    } 
                k += 1
        # the last layer is the target layer
        output['params'][group_name]['targetizer'] = {
            'kernel'  : jnp.transpose(jnp.array(nets[i][-1].weight.detach().cpu()), (1, 0)),
            'bias'    : jnp.array(nets[i][-1].bias.detach().cpu())
        }

    return freeze(output)

def mean_output(model_fn:Callable[[Any],jnp.ndarray], *args, **kwargs):
    """Compute mean output from model_fn with model_variables on inputs"""
    return model_fn(*args, **kwargs).mean()

def compute_enspred_grad_on_x_BATCH(
        model_fn:Callable[[FrozenDict,jnp.ndarray],jnp.ndarray], 
        model_vars:FrozenDict,  
        base_x_embedding:jnp.ndarray,
        x:jnp.ndarray) -> jnp.ndarray:
    """Compute gradient of x on model_fn with model_variables taking x and base_x_embedding as inputs 
    INPUT: 
    ######################
    base_x_embedding: (d_base,)
    x: (d,)

    OUTPUT:
    ######################
    grad_x: (d,)
    """
    output = jnp.vstack(
        tree_map(
            lambda j: grad(model_fn, 2)(
                freeze({
                    'params'      : model_vars['params']['MLP_'+str(j)],
                    'batch_stats' : model_vars['batch_stats']['MLP_'+str(j)]
                }), 
                base_x_embedding,
                x, 
            ), 
            list(range(len(model_vars['params'])))
        )      # (n_model, d)
    ).mean(0)  # (d,)
    return output

def compute_enspred_grad_on_x_SINGLE(
        model_fn:Callable[[FrozenDict,jnp.ndarray],jnp.ndarray], 
        model_vars:FrozenDict,  
        base_x_embedding:jnp.ndarray,
        x:jnp.ndarray) -> jnp.ndarray:
    """Compute gradient of x on model_fn with model_variables taking x and base_x_embedding as inputs 
    INPUT: 
    ######################
    base_x_embedding: (d_base,)
    x: (d,)

    OUTPUT:
    ######################
    grad_x: (d,)
    """
    output = jnp.vstack(
        tree_map(
            lambda j: grad(model_fn, 2)(
                freeze({
                    'params'      : model_vars['params']['MLP_'+str(j)],
                    'batch_stats' : model_vars['batch_stats']['MLP_'+str(j)]
                }), 
                base_x_embedding.reshape(-1),
                x, 
            ), 
            list(range(len(model_vars['params'])))
        )      # (n_model, d)
    ).mean(0)  # (d,)
    return output

def model_fn_targetizer(
        model_fn               : Callable[[FrozenDict,Any],jnp.ndarray], 
        batch_stats            : FrozenDict, 
        featurizer             : FrozenDict, 
        targetizer_vector_list : jnp.ndarray, 
        targetizer_structure, 
        *args, 
        **kwargs
    ):
    return mean_output(
        model_fn,
        freeze({
            'params': {'featurizer': featurizer, 
                       'targetizer': targetizer_structure(targetizer_vector_list)},
            'batch_stats': batch_stats
        }), 
        *args,
        **kwargs
    )

def intermediate_grad_fn(
        model_fn    : Callable[[FrozenDict,Any],jnp.ndarray], 
        batch_stats : FrozenDict, 
        featurizer  : FrozenDict, 
        targetizer  : FrozenDict,
        *args,
        **kwargs
    ): 
    targetizer_vector_list, targetizer_structure = ravel_pytree(targetizer)
    return grad(model_fn_targetizer, argnums=3)(
        model_fn,
        batch_stats, 
        featurizer, 
        targetizer_vector_list, 
        targetizer_structure,
        *args,
        **kwargs
    )

def compute_enspred(
        model_fn:Callable[[FrozenDict,jnp.ndarray],jnp.ndarray], 
        model_vars:FrozenDict,  
        *args,
        **kwargs 
    ) -> jnp.ndarray:
    # if 'MLP_0' in model_vars['params']:
    output = jnp.vstack(
        tree_map(
            lambda j: model_fn(
                freeze({
                    'params'      : model_vars['params']['MLP_'+str(j)],
                    'batch_stats' : model_vars['batch_stats']['MLP_'+str(j)]
                }),
                *args,
                **kwargs
            ), 
            list(range(len(model_vars['params'])))
        )      # (n_model,out_dim=1)
    ).mean()  # (1,)
    return output

def compute_loss(
        input             : jnp.ndarray,
        target            : jnp.ndarray,
        base_x_embedding  : jnp.ndarray, 
        model_fn          : Callable[[Dict, jnp.ndarray], jnp.ndarray],
        batch_stats       : FrozenDict,
        featurizer        : FrozenDict,
        targetizer        : FrozenDict,
    ) -> jnp.ndarray:   
    """
    INPUTS
    input            : (d,) 
    target           : ()
    base_x_embedding : (d_base,)

    RETURNS
    loss   : ()
    """
    pred = model_fn(
        freeze({
            'params': {'featurizer': featurizer, 'targetizer': targetizer},
            'batch_stats' : batch_stats
        }),  
        base_x_embedding,
        input 
    )  # ()
    loss = jnp.square(pred - target) # ()
    return loss.sum()

def compute_loss_vectorize(
        input             : jnp.ndarray,
        target            : jnp.ndarray, 
        base_x_embedding  : jnp.ndarray,
        model_fn          : Callable[[Dict, jnp.ndarray], jnp.ndarray],
        batch_stats       : FrozenDict,
        featurizer        : FrozenDict,
        targetizer_vector : jnp.ndarray, 
        targetizer_structure,
    ) -> jnp.ndarray:   
    """
    INPUT:
    input : (d,)
    target: ()
    base_x_embedding: (d_base,)
    
    RETURN:
    loss  : ()
    """
    targetizer = targetizer_structure(targetizer_vector)
    pred = model_fn(
        freeze({
            'params': {'featurizer': featurizer, 'targetizer': targetizer},
            'batch_stats' : batch_stats
        }),  
        base_x_embedding,
        input,  
    )                               
    return jnp.square(pred - target).sum()  # ()

def compute_loss_grad_and_jac(
        input             : jnp.ndarray,
        target            : jnp.ndarray, 
        base_x_embedding  : jnp.ndarray,
        model_fn          : Callable[[Dict, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        batch_stats       : FrozenDict,
        featurizer        : FrozenDict,
        targetizer_vector : jnp.ndarray, 
        targetizer_structure) -> List[jnp.ndarray]:
    """
    INPUT:
    input : (d,)
    target: ()
    base_x_embedding: (d_base,)
    
    RETURN:
    grad_on_vars      : (d_theta,)
    hessian_vars_and_x: (d_theta,d)
    """
    
    return value_and_jacfwd(compute_loss_grad_on_vars, 0)(
        input, 
        target,
        base_x_embedding,
        model_fn,
        batch_stats,
        featurizer,
        targetizer_vector, 
        targetizer_structure,
    )

def compute_loss_grad_on_vars(
        input             : jnp.ndarray,
        target            : jnp.ndarray, 
        base_x_embedding  : jnp.ndarray,
        model_fn          : Callable[[Dict, jnp.ndarray], jnp.ndarray],
        batch_stats       : FrozenDict,
        featurizer        : FrozenDict,
        targetizer_vector : jnp.ndarray, 
        targetizer_structure) -> jnp.ndarray:
    """
    INPUT:
    input : (d,)
    target: ()
    base_x_embedding: (d_base,)
    
    RETURN:
    grad_on_vars: (d_theta,)
    """
    
    return grad(compute_loss_vectorize, argnums=6)(
        input, 
        target,
        base_x_embedding,
        model_fn,
        batch_stats,
        featurizer,
        targetizer_vector, 
        targetizer_structure,
    )   
    
def compute_loss_vectorize_batch(
        input             : jnp.ndarray,
        targets           : jnp.ndarray, 
        base_x_embedding  : jnp.ndarray,
        model_fn          : Callable[[Dict, jnp.ndarray], jnp.ndarray],
        batch_stats       : FrozenDict,
        featurizer        : FrozenDict,
        targetizer_vector : jnp.ndarray, 
        targetizer_structure,
        n_batch           : int=1,
    ) -> jnp.ndarray:   
    """
    INPUT:
    input : (d,)
    target: (n_base,)
    base_x_embedding: (n_base,d_base)
    
    RETURN:
    loss  : ()
    """
    def model_fn_none(featurizer, targetizer, batch_stats, base_x_embedding, x):
        return model_fn(
                freeze({
                    'params': {'featurizer': featurizer, 'targetizer': targetizer},
                    'batch_stats' : batch_stats
                }),  
                base_x_embedding,
                x,
                n_batch=n_batch,
                reduction="none"
            )  
    targetizer = targetizer_structure(targetizer_vector)
    preds = jit(model_fn_none)(featurizer, targetizer, batch_stats, base_x_embedding, input)
    
    # model_fn(
    #     freeze({
    #         'params': {'featurizer': featurizer, 'targetizer': targetizer},
    #         'batch_stats' : batch_stats
    #     }),  
    #     base_x_embedding,
    #     input
    # )                               
    return jnp.mean(jnp.square(preds - targets))  # ()

def compute_loss_grad_on_vars_batch(
        input             : jnp.ndarray,
        targets           : jnp.ndarray, 
        base_x_embedding  : jnp.ndarray,
        model_fn          : Callable[[Dict, jnp.ndarray], jnp.ndarray],
        batch_stats       : FrozenDict,
        featurizer        : FrozenDict,
        targetizer_vector : jnp.ndarray, 
        targetizer_structure,
        n_batch           : int=1
        ) -> jnp.ndarray:
    """
    INPUT:
    input : (d,)
    target: (n_base,)
    base_x_embedding: (n_base,d_base)
    
    RETURN:
    grad_on_vars: (d_theta,)
    """
    
    return grad(compute_loss_vectorize_batch, argnums=6)(
        input, 
        targets,
        base_x_embedding,
        model_fn,
        batch_stats,
        featurizer,
        targetizer_vector, 
        targetizer_structure,
        n_batch
    )   
    
def compute_loss_grad_and_jac_batch(
        input             : jnp.ndarray,
        targets           : jnp.ndarray, 
        base_x_embedding  : jnp.ndarray,
        model_fn          : Callable[[Dict, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        batch_stats       : FrozenDict,
        featurizer        : FrozenDict,
        targetizer_vector : jnp.ndarray, 
        targetizer_structure,
        n_batch           : int=1
        ) -> List[jnp.ndarray]:
    """
    INPUT:
    input : (d,)
    target: (n_base,)
    base_x_embedding: (n_base,d_base)
    
    RETURN:
    grad_on_vars      : (d_theta,)
    hessian_vars_and_x: (d_theta,d)
    """
    
    return value_and_jacfwd(compute_loss_grad_on_vars_batch, 0)(
        input, 
        targets,
        base_x_embedding,
        model_fn,
        batch_stats,
        featurizer,
        targetizer_vector, 
        targetizer_structure,
        n_batch
    )

