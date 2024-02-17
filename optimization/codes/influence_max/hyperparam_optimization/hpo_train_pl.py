# https://github.com/xwshen51/engression/blob/main/engression-python/engression/models.py
from typing import Callable, Tuple 
import time
import os 
 

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader 
import lightning.pytorch as pl
from lightning.pytorch.callbacks import RichProgressBar 

import numpy as np
from jax import jit, clear_caches, numpy as jnp
from jax.tree_util import Partial, tree_map
from flax.core.frozen_dict import freeze
 
from codes.utils import print_x, gc_cuda 
from codes.influence_max.hyperparam_optimization.hpo_data_module import OptDataModule
from codes.influence_max.hyperparam_optimization.hpo_model_module import compute_enspred
from codes.influence_max.global_optimizer import global_optimization

def train_pl_model( 
    x:Tensor,
    search_domain:Tensor,
    base_dm:pl.LightningDataModule, 
    base_x_embedding_fn:nn.Module=None, 
    base_x_embedding_dim:int=512,
    train_y_fn:Callable[[np.ndarray], np.ndarray]=None,
    train_y_savedir:str=None,
    do_normalize_y:bool=False, 
    output_ensemble_xmin:bool=False,
    noiseed:int=101,
    device:str="cpu",
    **kwargs
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray], jnp.ndarray]:  
    """
    x: (n,d)
    base_x: (n_base, ...)
    base_y: (n_base, ...) 
    n_select_base: int
        total number of base data samples. For each set of x, 
        it selects the same subset of (base_x, base_y) of size 
        int(n_select_base/n) to obtain the corresponding y.
    y_fn: 
        callable function to give y that takes (x, base_x) as inputs.
        If y_savedir is None, y_fn has to be given.
    y_savedir:
        directory where the precomputed y are saved for all pairs of (x, base_x).
        It checks whether y_savedir is None first. If is None, y_fn has to be given.
    y_mchoise:
        model choice to compute y. Can be either 'last' or 'best'. 
    do_normalize_y: bool
    """
    
    if kwargs['use_double']:
        dtype = jnp.float64
    else:
        dtype = jnp.float32
     
    
    ## Creating the data module...
    infmax_dm = OptDataModule(
        base_dm=base_dm, 
        x=x, 
        n_select_base=kwargs.get('n_select_base', None),
        y_savedir=train_y_savedir, 
        y_fn=train_y_fn,
        do_normalize_y=do_normalize_y,
        n_model=kwargs.get('n_candidate_model',5)+kwargs.get('n_ensemble_model',5),
        batch_size=kwargs.get('batch_size', 1024),
        use_pretrained_featemb=kwargs.get('use_pretrained_featemb', False),
        differ_sample_for_each_model=kwargs.get('differ_sample_for_each_model', False)
    )
    
    if kwargs.get('use_stochastic', False):
        from codes.influence_max.hyperparam_optimization.hpo_model_module_pytorch import StoModel as MModule 
        from codes.influence_max.model_module import preprocess, sto_parameter_reconstruct as parameter_reconstruct
        from codes.influence_max.hyperparam_optimization.hpo_model_module import StoJMLPBatch as JMLPBatch, StoJMLPSingle as JMLPSingle

    else:
        from codes.influence_max.hyperparam_optimization.hpo_model_module_pytorch import RntModel as MModule 
        from codes.influence_max.model_module import preprocess, rnt_parameter_reconstruct as parameter_reconstruct
        from codes.influence_max.hyperparam_optimization.hpo_model_module import RntJMLPBatch as JMLPBatch, RntJMLPSingle as JMLPSingle
        
    infmax_model = MModule(
        *kwargs['n_hidden'], 
        base_x_embedding_fn=base_x_embedding_fn,
        base_x_embedding_dim=base_x_embedding_dim,
        n_model=kwargs.get('n_candidate_model',5)+kwargs.get('n_ensemble_model',5), 
        no_batch_norm=kwargs.get('no_batch_norm', False), 
        n_noise=kwargs.get('sto_n_noise', 500),
        noise_std=kwargs.get('sto_noise_std', 1.),
        search_domain=search_domain,
        trans_method=kwargs.get('trans_method', 'rbf'), 
        trans_rbf_nrad=kwargs.get('trans_rbf_nrad', 5), 
        use_double=kwargs.get('use_double', False),   
        learning_rate=kwargs.get('learning_rate', 0.001), 
        weight_decay=kwargs.get('weight_decay', 0.01),
        gamma=kwargs.get('gamma',0.1),  
        dropout_rate=kwargs.get('dropout_rate', 0.0),
        disable_base_x_embedding_training=kwargs.get('disable_base_x_embedding_training', True),
        use_pretrained_featemb=kwargs.get('use_pretrained_featemb', False),
        save_logpath=[os.path.join(kwargs['path_logs'], 'i{:04d}_log_infmax.pt'.format(int(x.shape[0]))),
                      os.path.join(kwargs['path_logs'], 'i{:04d}_log_infmax-strain.pt'.format(int(x.shape[0])))]
    )
    
    ## Creating the trainer...
    kwargs_trainer = {
        'max_epochs': kwargs.get('max_epochs', 10), 
        'min_epochs': kwargs.get('min_epochs', 1),
        'accelerator': device,
        'enable_model_summary': False,
        'check_val_every_n_epoch': kwargs.get('check_val_every_n_epoch',50),
        'default_root_dir': kwargs['path_logs'],
        'enable_checkpointing': False,
        'enable_progress_bar': kwargs.get('progress_bar', False),
        'logger': False,
        'deterministic': False,
        'devices': kwargs.get('n_devices',1),
        'reload_dataloaders_every_n_epochs': 1
    } 
    device = torch.device(device)

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
    
    if kwargs.get('use_pretrained_featemb', False):
        small_base_x_embedding_train = infmax_dm.get_precomputed_base_x_embedding(
            base_dm.small_base_train_i, 'train')
        small_base_x_embedding_targt = infmax_dm.get_precomputed_base_x_embedding(
            base_dm.small_base_targt_i, 'targt')
        
        if kwargs.get('disable_base_x_embedding_training', True):
            small_base_x_embedding_train = jnp.array(small_base_x_embedding_train)
            small_base_x_embedding_targt = jnp.array(small_base_x_embedding_targt)
        else:
            with torch.no_grad():
                infmax_model.base_x_embedding_fn.to(device)
                infmax_model.base_x_embedding_fn.eval()
            
                small_base_x_embedding_train = infmax_model.base_x_embedding_fn(
                        small_base_x_embedding_train.to(device))
                small_base_x_embedding_targt = infmax_model.base_x_embedding_fn(
                        small_base_x_embedding_targt.to(device))

            small_base_x_embedding_train = jnp.array(
                    small_base_x_embedding_train.cpu().numpy())
            small_base_x_embedding_targt = jnp.array(
                    small_base_x_embedding_targt.cpu().numpy())
    else:
        ## Construct base_x_embedding in JAX... 
        small_base_train_dl = DataLoader(
            base_dm.small_base_train_x,
            batch_size=base_dm.batch_size, 
            shuffle=False,
            drop_last=False,
            num_workers=base_dm.num_workers,
            pin_memory=base_dm.pin_memory)    
        small_base_targt_dl = DataLoader(
            base_dm.small_base_targt_x,
            batch_size=base_dm.batch_size, 
            shuffle=False,
            drop_last=False,
            num_workers=base_dm.num_workers,
            pin_memory=base_dm.pin_memory)    

        with torch.no_grad():
            infmax_model.to(device)
            infmax_model.eval()
            small_base_x_embedding_train = []
            small_base_x_embedding_targt = []
            for _, x in enumerate(small_base_train_dl):
                small_base_x_embedding_train.append(infmax_model.get_base_x_embedding(x.to(device)).cpu().numpy())

            for _, x in enumerate(small_base_targt_dl):
                small_base_x_embedding_targt.append(infmax_model.get_base_x_embedding(x.to(device)).cpu().numpy())
        
        small_base_x_embedding_train = jnp.array(np.concatenate(small_base_x_embedding_train, axis=0))
        small_base_x_embedding_targt = jnp.array(np.concatenate(small_base_x_embedding_targt, axis=0))
        
    ## Reconstruct the model in JAX ...
    model_fn_BATCH = JMLPBatch( 
        n_hidden            = kwargs['n_hidden'], 
        latent_embedding_fn = latent_embedding_fn,  
        ymean               = jnp.array(infmax_dm.ymean),
        ystd                = jnp.array(infmax_dm.ystd),
        no_batch_norm       = kwargs.get('no_batch_norm', False),
        n_noise             = kwargs.get('sto_n_noise', 500),
        noise_std           = kwargs.get('sto_noise_std', 1.),
        resample_size       = kwargs.get('sto_n_resample', 200),
        dtype               = dtype,
        key                 = noiseed
    ).apply

    model_fn_SINGLE = JMLPSingle( 
        n_hidden            = kwargs['n_hidden'], 
        latent_embedding_fn = latent_embedding_fn,  
        ymean               = jnp.array(infmax_dm.ymean),
        ystd                = jnp.array(infmax_dm.ystd),
        no_batch_norm       = kwargs.get('no_batch_norm', False),
        n_noise             = kwargs.get('sto_n_noise', 500),
        noise_std           = kwargs.get('sto_noise_std', 1.),
        resample_size       = kwargs.get('sto_n_resample', 200),
        dtype               = dtype,
        key                 = noiseed
    ).apply
    
    small_y_train  = jnp.array(infmax_dm.get_y(selected_base_indices=base_dm.small_base_train_i))
    model_vars = parameter_reconstruct(infmax_model.nets[:kwargs.get('n_candidate_model', 5)])
    model_vars_truehat = parameter_reconstruct(infmax_model.nets[kwargs.get('n_candidate_model', 5):])
    
    del infmax_model, base_dm, infmax_dm
    gc_cuda()     
    
     
    ## Obtaining xmin of test loss for each individual candidate model...
    t0 = time.time()
    xmins = jnp.vstack(tree_map(
        lambda j: global_optimization(
                Partial(
                    jit(model_fn_BATCH), 
                    freeze(
                        {'params'     : model_vars['params']['MLP_'+str(j)],
                        'batch_stats' : model_vars['batch_stats']['MLP_'+str(j)]}), 
                    small_base_x_embedding_targt
                ),
                # ###################### TOO SLOW ######################
                # ## If we put process_in_batches outside the StoJMLP  
                # ## then it takes around 3 mins 
                # ## Instead, if we put process_in_batches inside the StoJMLP  
                # ## then it only takes around 30 seconds
                # lambda x: process_in_batches(
                #             jit(Partial(model_fn, 
                #             freeze(
                #                 {'params'      : model_vars_all['params']['MLP_'+str(j)],
                #                 'batch_stats' : model_vars_all['batch_stats']['MLP_'+str(j)]}), 
                #             x=x)), 
                # small_base_x_embedding_targt,
                # 1,
                # "mean"),
            method          = kwargs.get('search_xmin_method', 'grid-search'),
            search_domain   = jnp.array(search_domain), 
            nstart          = kwargs.get('search_xmin_nstart', 100), 
            optimize_method = kwargs.get('search_xmin_opt_method', 'trust-constr'),  
            use_double      = kwargs.get('use_double', False),
            tol             = kwargs.get('search_xmin_opt_tol', 1e-4), 
            disp            = kwargs.get('disp', False))[0], 
        list(range(kwargs.get('n_candidate_model', 5)))
    ))
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
            Partial(
                compute_enspred,
                jit(model_fn_BATCH),
                model_vars,  
                small_base_x_embedding_targt 
            ),
            method          = kwargs.get('search_xmin_method', 'grid-search'),
            search_domain   = jnp.array(search_domain), 
            nstart          = kwargs.get('search_xmin_nstart', 100), 
            optimize_method = kwargs.get('search_xmin_opt_method', 'trust-constr'),  
            use_double      = kwargs.get('use_double', False),
            tol             = kwargs.get('search_xmin_opt_tol', 1e-4), 
            disp            = kwargs.get('disp', False))
         
        t1 = time.time()-t0
        print("Obtained xmin_star=({:s}). It takes {:.3f}s.".format(print_x(xmin_star), t1))    
    
    del model_fn_BATCH
    clear_caches()

    return (   
        model_fn_SINGLE,   
        model_vars,
        model_vars_truehat, 
        small_base_x_embedding_train,
        small_base_x_embedding_targt,
        small_y_train,                          # (n, n_base)
        xmins, 
        xmin_star,   
        jnp.array(list(train_metrics.values()))
    ) 
            
 
