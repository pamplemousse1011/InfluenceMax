from jax import jit, grad
import jax.numpy as jnp  
from jax.tree_util import tree_map
from jax import random  

# !pip install --upgrade -q git+https://github.com/google/flax.git
from flax.core.frozen_dict import FrozenDict
from flax import linen as nn
 
from typing import Callable, Sequence, Tuple
from codes.utils import zero_mean_unit_var_denormalization, generate_seed_according_to_time
from codes.influence_max.model_module import RntJBlock, StoJBlock, RntJSIN, StoJSIN
from typing import (
    Any,
    Callable,
    Sequence,
    Tuple,
)
PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
    
class StoJMLP(nn.Module):
    n_hidden             : Sequence[int]
    latent_embedding_fn : Callable[[jnp.ndarray], jnp.ndarray]
    ymean                : float = 0.
    ystd                 : float = 1.
    dtype                : Dtype = jnp.float32 
    no_batch_norm        : bool  = False
    n_noise              : int   = 100
    noise_std            : float = 1.0
    noise_n_resample     : int   = 100 
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
        x: (..., d) 
        """
        x_re = x.reshape(-1, x.shape[-1]) 
        b = x_re.shape[0]
        if self.n_noise > 0:
            ## Sample new response data          
            emb_rep = jnp.tile(
                x_re, 
                (self.noise_n_resample, *([1]*x_re.ndim))
            ).reshape(                                                          # (n_resample, b, d)
                self.noise_n_resample*x_re.shape[0], 
                *x_re.shape[1:]
            )                                                                   # (n_resample*b, d)
            ## Pass into model 
            output = self.featurizer(emb_rep)                                   # (n_resample*b, ...)
            output  = self.targetizer(output).reshape(                          # (n_resample*b, out_dim=1) 
                self.noise_n_resample, b, -1                                    # (n_resample, b, out_dim=1)          
            ).mean(0)                                                           # (b, out_dim=1)                       
        else: 
            emb_rep = self.featurizer(x_re)                                     # (b, ...)
            output  = self.targetizer(emb_rep)                                  # (b, out_dim=1)  
        
        ## Average the resamples
        output = output.squeeze(-1)                                             # (b,)

        if b == 1:
            return output.reshape()
        return output
    
    def __call__(self, x:jnp.ndarray):
        """
        INPUT 
        x : (..., n_dim)

        RETURN
        (...,)

        - First sample new response data, x_rep, by creating identical copies, 
          of shape (n_resample, n, n_dim). 
        - Then pass the response data into model and obtain samples
        """ 

        ## Get x embeddings, (d)
        emb = self.latent_embedding_fn(x)    
        
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
    dtype                : Dtype = jnp.float32 
    no_batch_norm        : bool  = False
    n_noise              : int   = 100
    noise_std            : float = 1.0
    noise_n_resample     : int   = 100
    key                  : int   = generate_seed_according_to_time(1)[0]

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
        self.MLP = modulelist
    
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
            self.MLP,
            (self.n_noise>0), 
            self.noise_n_resample 
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
    dtype                : Dtype = jnp.float32 
    no_batch_norm        : bool  = False
    n_noise              : int   = 100
    noise_std            : float = 1.0
    noise_n_resample     : int   = 100
    key                  : int   = generate_seed_according_to_time(1)[0]

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
        self.MLP = modulelist

    def compute_output(self, emb: jnp.ndarray) -> jnp.ndarray:
        """
        emb: (d,)
        """
        # Concatenate the embeddings
        output = compute_ens_from_embedding(emb, self.MLP, (self.n_noise>0), self.n_resample)
        
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

class RntJMLP(nn.Module):
    n_hidden             : Sequence[int]
    latent_embedding_fn  : Callable[[jnp.ndarray], jnp.ndarray]
    ymean                : float = 0.
    ystd                 : float = 1.
    dtype                : Dtype = jnp.float32 
    no_batch_norm        : bool  = False
    n_noise              : int   = 100
    noise_std            : float = 1.0
    noise_n_resample     : int   = 100  
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
        x : (..., n_dim)

        RETURN
        (...,)

        - First sample new response data, x_rep, by creating identical copies, 
          of shape (resample_size, n, n_dim). 
        - Then pass the response data into model and obtain samples
        """ 

        ## Get x embeddings, (d,)
        latent_embedding = self.latent_embedding_fn(x)   
        ## Get model output ()
        output = self.compute_output(latent_embedding) 
        ## Denormalize
        output = zero_mean_unit_var_denormalization(output, self.ymean, self.ystd)
        return output

class RntJENS(nn.Module):
    n_model              : int 
    n_hidden             : Sequence[int] 
    latent_embedding_fn  : Callable[[jnp.ndarray], jnp.ndarray] 
    ymean                : float = 0.
    ystd                 : float = 1.
    dtype                : Dtype = jnp.float32
    no_batch_norm        : bool  = False
    n_noise              : int   = 0
    noise_std            : float = 1.0
    noise_n_resample     : int   = 100
    key                  : int   = generate_seed_according_to_time(1)[0]
    
    def setup(self): 
        modulelist = {} 
        for ii in range(self.n_model):
            modulelist[str(ii)] = RntJSIN(
                n_hidden        = self.n_hidden,  
                dtype           = self.dtype
            ) 
        self.MLP = modulelist
    
    def __call__(self, x:jnp.ndarray):
        """
        INPUT  
        x : (..., n_dim)

        RETURN
        (...,)
        """
        
        ## Get x embeddings, (...,d)
        emb = self.latent_embedding_fn(x) 

        ## Get model output, (...,)
        output = compute_ens_from_embedding(emb, self.MLP) 

        ## Denormalize
        output = zero_mean_unit_var_denormalization(output, self.ymean, self.ystd)

        return output

def jac_func(
        model_fn    : Callable[[FrozenDict, jnp.ndarray], jnp.ndarray], 
        x           : jnp.ndarray, 
        batch_stats : FrozenDict,
        featurizer  : FrozenDict, 
        targetizer  : FrozenDict
    ) -> jnp.ndarray:
    """Compute ∂𝜇(𝜃,𝑥)/∂𝑥
    Input x of size (d,)
    """ 
    res = jit(grad(model_fn, argnums=1))(
        {
            'params': {'featurizer': featurizer, 'targetizer': targetizer},
            'batch_stats': batch_stats
        }, 
        x
    )
    return res

def compute_loss(
        inputs            : jnp.ndarray,
        targets           : jnp.ndarray, 
        model_fn          : Callable[[jnp.ndarray], jnp.ndarray],
        batch_stats       : FrozenDict,
        featurizer        : FrozenDict,
        targetizer        : FrozenDict,
    ) -> jnp.ndarray:   
    """
    reduction='mean'
    """
    pred = jit(model_fn)(
        {
            'params': {'featurizer': featurizer, 'targetizer': targetizer},
            'batch_stats': batch_stats
        }, 
        inputs
    )
    loss = jit(jnp.mean)((pred - targets) ** 2)
    return loss

def compute_loss_vectorize_single(
        inputs            : jnp.ndarray,
        targets           : jnp.ndarray, 
        model_fn          : Callable[[jnp.ndarray], jnp.ndarray],
        batch_stats       : FrozenDict,
        featurizer        : FrozenDict,
        targetizer_vector : jnp.ndarray, 
        targetizer_structure    
    ) -> jnp.ndarray:   
    """
    reduction='mean': return ()  
    """
    targetizer = targetizer_structure(targetizer_vector)
    pred = jit(model_fn)(
        {
            'params': {'featurizer': featurizer, 'targetizer': targetizer},
            'batch_stats' : batch_stats 
        }, 
        inputs
    )
    loss = jit(jnp.mean)((pred - targets) ** 2)
    return loss

def compute_loss_vectorize(
        inputs                 : jnp.ndarray,
        targets                : jnp.ndarray, 
        model_fn               : Callable[[jnp.ndarray], jnp.ndarray],
        model_params           : FrozenDict,
        targetizer_vector_list : jnp.ndarray, 
        targetizer_structure,
    ):
    """
    Compute the sum, avoiding scaling issues. 
    """
    return jnp.stack(tree_map(lambda j: compute_loss_vectorize_single(
            inputs               = inputs, 
            targets              = targets,
            model_fn             = model_fn,
            batch_stats          = model_params['batch_stats']['MLP_'+str(j)],
            featurizer           = model_params['params']['MLP_'+str(j)]['featurizer'],
            targetizer_vector    = targetizer_vector_list[j],
            targetizer_structure = targetizer_structure
        ),
        list(range(len(model_params['params'])))
    ), axis=0).sum(0)

def compute_ens_from_embedding(x:jnp.ndarray, MLPs:nn.Module, stochastic:bool=False, n_resample:int=1) -> jnp.ndarray:
    """
    x: embedding (..., d)
    
    if stochastic:
    - First sample new response data, x_rep, by creating identical copies, 
        of shape (n_resample, n_dim). 
    - Then pass the response data into model and obtain samples, where
        samples[n_base*(i-1):n_sample*i,:] contains one sample for each data point, 
        for i = 1, ..., n_resample
    """
    n_model = len(MLPs)
    x_re = x.reshape(-1, x.shape[-1])
    b = x_re.shape[0]
    if stochastic > 0:
        ## Sample new response data 
        emb_rep = jnp.tile(
            x_re, 
            (n_resample, *([1]*x_re.ndim))
        ).reshape(                                          # (n_resample, b, d)
            n_resample*b, *x_re.shape[1:]  
        )                                                   # (n_resample*b, d)
        ## Pass into model and average over n_model
        output = jnp.stack(
            [MLPs[str(ii)](emb_rep) 
            for ii in range(n_model)],
            axis = 0
        ).mean(0).reshape(                                 # (n_resample*b, out_dim=1)
            n_resample, b, -1
        ).mean(0)                                          # (b, out_dim=1)
         
    else: 
        output = jnp.stack(
            [MLPs[str(ii)](x_re) 
            for ii in range(n_model)],
            axis = 0                                       # (n_model, b, out_dim=1) 
        ).mean(0)                                          # (b, out_dim=1) 
    output = output.squeeze(-1)                            # (b,)

    if b == 1:
        return output.reshape()
    return output                       

