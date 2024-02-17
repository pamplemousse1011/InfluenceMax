from importlib import reload

import numpy as np
import jax.numpy as jnp
import torch

#####################################################################  
"""Check RntJMLPSingle with rnt_parameter_reconstruct and RntModel yield the same results"""
import codes.influence_max.model_module
reload(codes.influence_max.model_module)
from codes.influence_max.model_module import rnt_parameter_reconstruct, preprocess
import codes.influence_max.hyperparam_optimization.hpo_model_module_pytorch
reload(codes.influence_max.hyperparam_optimization.hpo_model_module_pytorch)
from codes.influence_max.hyperparam_optimization.hpo_model_module_pytorch import  RntModel
from codes.influence_max.hyperparam_optimization.hpo_model_module import RntJMLPSingle
import torch

def main():
    n_hidden = [100,50]
    n_model  = 5

    resmodel = RntModel(*n_hidden,
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
        print("{}-th model difference: {:.4f}".format(j, float(out_reconstructed) - out[j].item()))
        

if __name__ == "__main__":
    torch.set_num_threads(1)
    main()

