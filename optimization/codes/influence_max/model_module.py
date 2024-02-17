import torch  
from jax import vmap, random, grad 
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
    Sequence,
    Tuple, 
    Dict
)
PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
 
from codes.utils import generate_seed_according_to_time

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
