#####################################################################  
from importlib import reload

from jax import random 
import jax.numpy as jnp

from codes.influence_max.opt_model_module import StoJMLPBatch

model_check = StoJMLPBatch(
    n_hidden            = [50,50,50],
    latent_embedding_fn = lambda x: x,
    n_noise             = 10,
    resample_size       = 10,
)

d1=22
d2=4
d=d1+d2

## MAIN CHECK FUNCTIONS
variables = model_check.init(random.PRNGKey(0), jnp.ones((d1,)), jnp.ones((d2,))) 
n_base = 10 
for bxe in [jnp.ones((1,d1)), jnp.ones((d1,)), jnp.ones((n_base, d1))]:
    output = model_check.apply(
        variables, 
        base_x_embedding=bxe,
        x=jnp.ones((d2,)),
        n_batch=3)
    print(output.shape)


#####################################################################
"""Check ∂2𝜇(b,𝑥)/∂b∂𝑥 and ∂2𝜇(b,𝑥)/∂𝑥∂b yield the same results"""
import numpy as np
from jax import grad, jit, vjp, jvp, random 
import jax.numpy as jnp
from jax.tree_util import Partial
from jax.flatten_util import ravel_pytree

import flax.linen as nn

class SOMEMODEL(nn.Module):
    n_feature :int = 12
    b         :jnp.ndarray=jnp.ones(8,)
    def setup(self):
        self.featurizer = nn.Dense(self.n_feature)
        self.targetizer = nn.Dense(1)
    def __call__(self,x:jnp.ndarray,a:float=2.):
        out = jnp.concatenate([self.b.reshape(-1), x.reshape(-1)])
        out = self.featurizer(out+a)
        out = self.targetizer(out)
        return out

import codes.influence_max.opt_model_module
reload(codes.influence_max.opt_model_module)
from codes.influence_max.opt_model_module import intermediate_grad_fn, mean_output

 
                
def intermediate_grad_fn_check(
        model_fn    , 
        batch_stats , 
        featurizer  , 
        targetizer  ,  
        x,
        a
    ):
    return grad(mean_output, argnums=2)(
        jit(model_fn),
        {
            'params': {'featurizer': featurizer, 'targetizer': targetizer},
            'batch_stats': batch_stats
        },  
        x,
        a
    )


## MAIN FUNCTION TO CHECK DIFFERENCE
def check_difference(model_fn, variables, x:jnp.ndarray, v:jnp.ndarray, a:float):
    """
    model_fn
    x: (d,)
    v: (d,)
    """
    # ∂/∂b (∂𝜇(b,𝑥)/∂𝑥)
    _, f_vjp = vjp(
        Partial(
            intermediate_grad_fn_check,
            jit(model_fn), 
            {},  
            variables['params']['featurizer'], 
            x=x,
            a=a
        ), 
        variables['params']['targetizer']
    ) 
    tangents_check = ravel_pytree(f_vjp(v)[0])[0]
    
    # ∂/∂𝑥 (∂𝜇(b,𝑥)/∂b)
    _, tangents = jvp(
        Partial(
            intermediate_grad_fn,
            jit(model_fn), 
            {},  
            variables['params']['featurizer'], 
            variables['params']['targetizer'],
            a=a
        ), 
        (x,), 
        (v,)
    )

    return jnp.max(tangents-tangents_check)

    
d1=4
d2=8
n_feature=d1+d2

b = jnp.array(np.random.normal(0,1,(d1,)))
somemodel = SOMEMODEL(n_feature, b)
variables = somemodel.init(random.PRNGKey(0), jnp.ones((d2,)))    

x = jnp.array(np.random.normal(0,1,(d2,)))
v = jnp.array(np.random.normal(0,1,(d2,)))
check_difference(somemodel.apply, variables, x, v, a=2.)

#####################################################################  
"""Check RntJMLPSimple with rnt_parameter_reconstruct and RntModelSimple yield the same results"""
import codes.influence_max.hyperparam_optimization.opt_model_module_pytorch
reload(codes.influence_max.hyperparam_optimization.opt_model_module_pytorch)
from codes.influence_max.hyperparam_optimization.opt_model_module_pytorch import  RntModelSimple
import codes.influence_max.opt_model_module
reload(codes.influence_max.opt_model_module)
from codes.influence_max.opt_model_module import RntJMLPSingle, rnt_parameter_reconstruct, preprocess
import torch

n_hidden = [100,50]
n_model  = 5

resmodel = RntModelSimple(*n_hidden,
    base_x_embedding_fn=None,
    base_x_embedding_dim=512,
    n_model=n_model, 
    search_domain=torch.from_numpy(np.array([[0.,1.],[-1.,2.],[3.,4.],[-2.2,-1.]])),
    trans_method="rbf",
    trans_rbf_nrad=5,
    use_double=False, 
    learning_rate=0.001, 
    weight_decay=0.01,
    gamma=0.1,  
    dropout_rate=0,
    diable_base_x_embedding_training=True,
    n_noise=2
)

## Creating the preprocess function...
latent_embedding_fn = preprocess(
    mu     = jnp.array(resmodel.latent_embedding_fn.mu.cpu().numpy()), 
    gamma  = jnp.array(resmodel.latent_embedding_fn.gamma.cpu().numpy()),
    method = "rbf")

resmodel_jax = RntJMLPSingle(n_hidden=n_hidden,latent_embedding_fn=latent_embedding_fn)

## Reconstruct RntJMLPSimple with rnt_parameter_reconstruct
variables_reconstruct = rnt_parameter_reconstruct(resmodel.nets)

## Check the results
b = jnp.array(np.random.normal(0,1,(512,)))
x = jnp.array(np.random.normal(0,1,(4,)))

resmodel.eval()
out = resmodel(torch.from_numpy(np.array(b)),torch.from_numpy(np.array(x)))
for j in range(n_model):
    out_reconstructed = resmodel_jax.apply(
        {'params': variables_reconstruct['params'][f'MLP_{j}'],
        'batch_stats': variables_reconstruct['batch_stats'][f'MLP_{j}']}, b, x)
    print(float(out_reconstructed) - out[j].item())
     