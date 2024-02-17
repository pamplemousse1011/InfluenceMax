from jax import vmap, jit, grad, tree_map, random 
import jax.numpy as jnp 
from jax.tree_util import Partial
from jax.flatten_util import ravel_pytree

# !pip install --upgrade -q git+https://github.com/google/flax.git
from flax.core import freeze 
from flax.core.frozen_dict import FrozenDict
from flax import linen as nn

from codes.influence_max.utils import value_and_jacfwd
from codes.utils import generate_seed_according_to_time, zero_mean_unit_var_denormalization
from codes.influence_max.model_module import StoJBlock, StoJSIN, RntJBlock

from typing import (
    Any,
    List, 
    Callable,
    Sequence,
    Tuple, 
    Dict
)
PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
 
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

def compute_enspred(
        model_fn:Callable[[FrozenDict,jnp.ndarray],jnp.ndarray], 
        model_vars:FrozenDict,  
        *args,
        **kwargs 
    ) -> jnp.ndarray:
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

