from importlib import reload

import numpy as np
from jax import jit, numpy as jnp
from jax.tree_util import Partial
import torch

#####################################################################  
"""Check RntJMLP with rnt_parameter_reconstruct and RntModel yield the same results"""
import codes.influence_max.model_module
reload(codes.influence_max.model_module)
from codes.influence_max.model_module import rnt_parameter_reconstruct, preprocess
import codes.influence_max.noisy_funct_optimization.nfo_model_module_pytorch
reload(codes.influence_max.noisy_funct_optimization.nfo_model_module_pytorch)
from codes.influence_max.noisy_funct_optimization.nfo_model_module_pytorch import  RntModel
from codes.influence_max.noisy_funct_optimization.nfo_model_module import RntJMLP, RntJENS
import torch

n_hidden = [100,50]
n_model  = 5

resmodel = RntModel(
    *n_hidden,
    n_model=n_model, 
    search_domain=torch.from_numpy(np.array([[0.,1.],[-1.,2.],[3.,4.],[-2.2,-1.]])),
    trans_method="rbf",
    trans_rbf_nrad=5,
    use_double=False, 
    learning_rate=0.001, 
    weight_decay=0.01,
    gamma=0.1,  
    dropout_rate=0,
)

## Creating the preprocess function...
latent_embedding_fn = preprocess(
    mu     = jnp.array(resmodel.latent_embedding_fn.mu.cpu().numpy()), 
    gamma  = jnp.array(resmodel.latent_embedding_fn.gamma.cpu().numpy()),
    method = "rbf")

resmodel_jax = RntJMLP(n_hidden=n_hidden,latent_embedding_fn=latent_embedding_fn)

## Reconstruct RntJMLP with rnt_parameter_reconstruct
variables_reconstruct = rnt_parameter_reconstruct(resmodel.nets)

## Check the results
x = jnp.array(np.random.normal(0,1,(4,)))

resmodel.eval()
out = resmodel(torch.from_numpy(np.array(x).reshape(-1,4)))
for j in range(n_model):
    out_reconstructed = resmodel_jax.apply(
        {'params': variables_reconstruct['params'][f'MLP_{j}'],
        'batch_stats': variables_reconstruct['batch_stats'][f'MLP_{j}']}, x)
    print("{}-th model difference: {:.4f}".format(j, float(out_reconstructed) - out[j].item()))
     
#####################################################################  
"""Check RntJENS with rnt_parameter_reconstruct and RntModel yield the same results"""
n_hidden = [100,50]
n_model  = 5

resmodel = RntModel(
    *n_hidden,
    n_model=n_model, 
    search_domain=torch.from_numpy(np.array([[0.,1.],[-1.,2.],[3.,4.],[-2.2,-1.]])),
    trans_method="rbf",
    trans_rbf_nrad=5,
    use_double=False, 
    learning_rate=0.001, 
    weight_decay=0.01,
    gamma=0.1,  
    dropout_rate=0,
) 

## Creating the preprocess function...
latent_embedding_fn = preprocess(
    mu     = jnp.array(resmodel.latent_embedding_fn.mu.cpu().numpy()), 
    gamma  = jnp.array(resmodel.latent_embedding_fn.gamma.cpu().numpy()),
    method = "rbf")


## Reconstruct RntJENS with rnt_parameter_reconstruct
ensmodel_fn = RntJENS(n_hidden=n_hidden, n_model=n_model, latent_embedding_fn=latent_embedding_fn).apply
variables_ens = rnt_parameter_reconstruct(resmodel.nets)
ensmodel_fn   = Partial(jit(ensmodel_fn), variables_ens)

## Check the results
x = jnp.array(np.random.normal(0,1,(4,)))
ensmodel_fn(x)