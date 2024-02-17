import os

import numpy as np
import torch 
from torch import Tensor
import lightning.pytorch as pl
from lightning.pytorch.callbacks import RichProgressBar
from typing import Callable, Tuple 
import time 

from jax import jit, clear_caches, numpy as jnp
from jax.tree_util import Partial, tree_map
from flax.core.frozen_dict import freeze

from codes.utils import gc_cuda, print_x
from codes.influence_max.global_optimizer import global_optimization
from codes.influence_max.noisy_funct_optimization.nfo_data_module import OptDataModule 

def train_pl_model( 
    search_domain:np.ndarray,
    train_features:np.ndarray,
    train_labels:np.ndarray,   
    do_normalize_y:bool=False, 
    output_ensemble_xmin:bool=False,
    noiseed:int=101,
    device:str="cpu",
    **kwargs
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray], jnp.ndarray]:  
    
    if kwargs['use_double']:
        dtype = jnp.float64
    else:
        dtype = jnp.float32
     
    ## Create the data module 
    infmax_dm = OptDataModule(
        train_x=torch.from_numpy(train_features),
        train_y=torch.from_numpy(train_labels),   
        do_normalize_y=do_normalize_y,
        n_candidate_model=kwargs['n_candidate_model'],
        n_ensemble_model=kwargs['n_ensemble_model'],
        leave_one_out=kwargs['leave_one_out'],
        batch_size=kwargs['batch_size'], 
        **{'num_workers':kwargs['num_workers'], 'pin_memory':kwargs['pin_memory']}    
    )

    if kwargs.get('use_stochastic', False): 
        from codes.influence_max.noisy_funct_optimization.nfo_model_module_pytorch import StoModel as MModule 
        from codes.influence_max.model_module import preprocess, sto_parameter_reconstruct as parameter_reconstruct
        from codes.influence_max.noisy_funct_optimization.nfo_model_module import StoJMLP as JMLP, StoJENS as JENS
    else:
        from codes.influence_max.noisy_funct_optimization.nfo_model_module_pytorch import RntModel as MModule 
        from codes.influence_max.model_module import preprocess, rnt_parameter_reconstruct as parameter_reconstruct
        from codes.influence_max.noisy_funct_optimization.nfo_model_module import RntJMLP as JMLP, RntJENS as JENS 
        
    pl.seed_everything(kwargs['trial'], workers=True)       
    infmax_model = MModule(
        *kwargs['n_hidden'], 
        n_model=kwargs['n_candidate_model']+kwargs['n_ensemble_model'],  
        no_batch_norm=kwargs.get('no_batch_norm', False), 
        n_noise=kwargs.get('sto_n_noise', 0),
        noise_std=kwargs.get('sto_noise_std', 1.),
        noise_n_resample=kwargs.get('sto_n_resample', 100),
        search_domain=torch.from_numpy(search_domain),
        trans_method=kwargs.get('trans_method', 'rbf'), 
        trans_rbf_nrad=kwargs.get('trans_rbf_nrad', 5), 
        use_double=kwargs.get('use_double', False),   
        learning_rate=kwargs.get('learning_rate', 0.001), 
        weight_decay=kwargs.get('weight_decay', 0.001),
        gamma=kwargs.get('gamma',0.1),  
        dropout_rate=kwargs.get('dropout_rate', 0.0),
        save_logpath=[os.path.join(kwargs['path_logs'], 'i{:04d}_log_infmax-train.pt'.format(int(train_features.shape[0]))),
                      os.path.join(kwargs['path_logs'], 'i{:04d}_log_infmax-valid.pt'.format(int(train_features.shape[0])))]
    )
    
    ## Create the trainer
    kwargs_trainer = {
        'max_epochs': kwargs.get('max_epochs',1000), 
        'min_epochs': kwargs.get('min_epochs',1),
        'accelerator': device,
        'enable_model_summary': False,
        'check_val_every_n_epoch': kwargs['check_val_every_n_epoch'],
        'default_root_dir': kwargs['path_logs'],
        'enable_checkpointing': False,
        'enable_progress_bar': kwargs['progress_bar'],
        'logger': False,
        'deterministic': False,
        'devices': kwargs.get('n_devices',1),
    } 

    callbacks = [RichProgressBar()] if kwargs['progress_bar'] else []

    t0 = time.time()
    trainer = pl.Trainer(**kwargs_trainer, callbacks=callbacks)
    ## Fitting the model...
    trainer.fit(model=infmax_model, datamodule=infmax_dm) 
    ## Testing the model... 
    train_metrics = trainer.test(model=infmax_model, datamodule=infmax_dm)[0]
    t1 = time.time()-t0
    print("Trained {:d} infmax models. It takes {:.3f}s.".format(
        kwargs.get('n_candidate_model',5)+kwargs.get('n_ensemble_model',5), 
        t1
    ))
    
    ## Creating the preprocess function...
    latent_embedding_fn = preprocess(
        mu     = jnp.array(infmax_model.latent_embedding_fn.mu.cpu().numpy() if isinstance(infmax_model.latent_embedding_fn.mu, Tensor) 
                           else infmax_model.latent_embedding_fn.mu), 
        gamma  = jnp.array(infmax_model.latent_embedding_fn.gamma.cpu().numpy() if isinstance(infmax_model.latent_embedding_fn.gamma, Tensor) 
                           else infmax_model.latent_embedding_fn.gamma),
        method = kwargs.get('trans_method', 'rbf'))
    
    
    model = JMLP(
        n_hidden            = kwargs['n_hidden'], 
        latent_embedding_fn = latent_embedding_fn,  
        ymean               = jnp.array(infmax_dm.ymean),
        ystd                = jnp.array(infmax_dm.ystd),
        dtype               = dtype,
        no_batch_norm       = kwargs.get('no_batch_norm',False),
        n_noise             = kwargs.get('sto_n_noise', 0),
        noise_std           = kwargs.get('sto_noise_std', 1.),
        noise_n_resample    = kwargs.get('sto_n_resample', 100),
        key                 = noiseed
    ).apply 
    
    ensmodel_fn = JENS(
        n_model             = kwargs['n_ensemble_model'],
        n_hidden            = kwargs['n_hidden'],
        latent_embedding_fn = latent_embedding_fn,  
        ymean               = jnp.array(infmax_dm.ymean),
        ystd                = jnp.array(infmax_dm.ystd),
        dtype               = dtype,
        no_batch_norm       = kwargs.get('no_batch_norm',False),
        n_noise             = kwargs.get('sto_n_noise', 0),
        noise_std           = kwargs.get('sto_noise_std', 1.),
        noise_n_resample    = kwargs.get('sto_n_resample', 100),
        key                 = noiseed
    ).apply

    # variables = mlp.init(random.PRNGKey(0), jnp.ones((search_domain.shape[0],)))
    # variables = mlp.init(jnp.array([0, 1], dtype=jnp.uint32), jnp.ones((search_domain.shape[0],)))
    # del variables 

    variables     = parameter_reconstruct(infmax_model.nets[:kwargs['n_candidate_model']])
    variables_ens = parameter_reconstruct(infmax_model.nets[kwargs['n_candidate_model']:])
    ensmodel_fn   = Partial(jit(ensmodel_fn), variables_ens)
    del variables_ens
    del infmax_model
    gc_cuda()

    
    t0 = time.time()
    ## Obtain xmin  
    xmins = jnp.vstack(tree_map(
        lambda j: global_optimization(
            Partial(
                model,
                freeze({'params'     : variables['params']['MLP_'+str(j)],
                        'batch_stats': variables['batch_stats']['MLP_'+str(j)]}), 
            ),
            method          = kwargs.get('search_xmin_method', 'grid-search'), 
            search_domain   = jnp.array(search_domain), 
            nstart          = kwargs.get('search_xmin_nstart', 100), 
            optimize_method = kwargs.get('search_xmin_opt_method', 'trust-constr'),  
            use_double      = kwargs.get('use_double', False),
            tol             = kwargs.get('search_xmin_opt_tol', 1e-4), 
            disp            = kwargs.get('disp', False))[0], 
        list(range(kwargs['n_candidate_model']))))
    t1 = time.time() - t0 
    print("Obtained xmin from {:d} models. It takes {:.3f}s.".format(kwargs.get('n_candidate_model',5), t1))
    for ii in range(kwargs.get('n_candidate_model', 5)):
        print("xmin(M{:d})=({:s})".format(ii, print_x(xmins[ii]))) 


    xmin_star = np.empty_like(xmins[0])
    if output_ensemble_xmin:
        t0 = time.time()
        """
        Random choose one to obtain next 
        """
        # choosen_idx = np.random.choice(kwargs['n_candidate_model'])
        # xmin_star = xmins[choosen_idx]
        """
        Obtaining xmin for the ensemble model (excluding jackknife ones)
        """
        xmin_star, _ = global_optimization(
            ensmodel_fn,
            method          = kwargs.get('search_xmin_method', 'grid-search'),
            search_domain   = jnp.array(search_domain), 
            nstart          = kwargs.get('search_xmin_nstart', 100), 
            optimize_method = kwargs.get('search_xmin_opt_method', 'trust-constr'),  
            use_double      = kwargs.get('use_double', False),
            tol             = kwargs.get('search_xmin_opt_tol', 1e-4), 
            disp            = kwargs.get('disp', False))
         
        t1 = time.time()-t0
        print("Obtained xmin_star=({:s}). It takes {:.3f}s.".format(print_x(xmin_star), t1))    
    
    clear_caches()
    

    return (model, variables, xmins, xmin_star, ensmodel_fn, jnp.array(list(train_metrics.values()))) 
            
 
