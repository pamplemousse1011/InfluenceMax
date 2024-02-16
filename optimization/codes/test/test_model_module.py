from importlib import reload

from jax import random 
import jax.numpy as jnp

from codes.influence_max.hyperparam_optimization.hpo_model_module import StoJMLPBatch

#####################################################################  
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

import codes.influence_max.model_module
reload(codes.influence_max.model_module)
from codes.influence_max.model_module import intermediate_grad_fn, mean_output

 
                
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
