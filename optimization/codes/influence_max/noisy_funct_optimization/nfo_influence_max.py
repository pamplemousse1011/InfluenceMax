import numpy as np
import jax
from jax import jvp, grad, jit, jacfwd, jacrev, clear_caches, vmap
import jax.numpy as jnp
from jax.tree_util import Partial, tree_map
from jax.flatten_util import ravel_pytree
  
from flax.core import freeze 
from flax.core.frozen_dict import FrozenDict

import dataclasses 

from typing import Sequence, Tuple, Callable, List, Union
import time

from codes.influence_max.global_optimizer import global_optimization 
from codes.influence_max.model_module import intermediate_grad_fn
from codes.influence_max.noisy_funct_optimization.nfo_model_module import compute_loss_vectorize_single, jac_func
from codes.influence_max.noisy_funct_optimization.nfo_ihvp import inverse_hvp_fn
from codes.influence_max.utils import value_and_jacfwd, sum_helper

@dataclasses.dataclass
class AcquisitionBatch:
    samples: Sequence[float] 
    scores: Sequence[float]

 
class InfluenceMax(object):
    def __init__(self,
                 train_features      : jnp.ndarray,
                 train_labels        : jnp.ndarray, 
                 xmins               : jnp.ndarray,
                 search_domain       : np.ndarray,  
                 model_fn            : Callable[[FrozenDict, jnp.ndarray], jnp.ndarray],
                 model_params        : FrozenDict,
                 ensmodel_fn         : Callable[[FrozenDict, jnp.ndarray], jnp.ndarray],
                 acquire_k           : int=1,
                 m_kmeansplusplus    : float=1, 
                 train_loss          : jnp.ndarray=None,
                 **kwargs):
        
        self.kwargs            = kwargs 
        self.model_fn          = model_fn  
        self.model_params      = model_params
        self.ensmodel_fn       = ensmodel_fn 
        self.dim               = search_domain.shape[0]
        self.acquire_k         = acquire_k
        self.m_kmeansplusplus  = m_kmeansplusplus 
        self.train_features    = train_features
        self.train_labels      = train_labels
        self.train_loss        = train_loss
        self.optimization_mode = True

        if kwargs['use_double']:
            self.dtype = jnp.float64
        else:
            self.dtype = jnp.float32
        self.get_ihvp = inverse_hvp_fn(kwargs['ihvp_method'])
        
        # Dataset
        self.search_domain     = search_domain
        self.xmins             = xmins 

    def compute_GETaskGoal(
            self, 
            model_params: FrozenDict, 
            xmin:jnp.ndarray, 
            e:float
        ) -> FrozenDict:  
        """
        Input xmin: (d,)
        """
        xmin = xmin.reshape(-1)
        """
        Compute temp3 = ∇𝑥 𝜇(𝜃,𝑥) here we use ∇𝑥 𝜇(𝜃hat,𝑥)
        """ 
        temp3 = grad(jit(self.ensmodel_fn))(xmin) #(d,)
         
        """
        Compute temp2 = -[∂2/∂𝑥2 𝜇(b,𝑥)]^{−1}  \dot [∇𝑥 𝜇(𝜃,𝑥)] 
                      = -[∂2/∂𝑥2 𝜇(b,𝑥)]^{−1}  \dot temp3
        """
        partial2_mu_x = jit(jacfwd(jacrev(self.model_fn, argnums=1), argnums=1))(
            model_params,
            xmin
        ) # (d,d)

        temp2 = - jit(jnp.linalg.solve)(
            partial2_mu_x + e * jnp.eye(self.dim).astype(partial2_mu_x.dtype), 
            temp3
        ) #(d,) 

        """
        Compute temp1 = [∂2𝜇(b,𝑥)/∂b∂𝑥] \dot temp2
        """ 
        temp1 = jvp(Partial(
            intermediate_grad_fn,
            jit(self.model_fn),
            model_params['batch_stats'], 
            model_params['params']['featurizer'], 
            model_params['params']['targetizer']), 
            (xmin,), 
            (temp2,)
        )[1]

        # print("jvp", temp1)
        # _, f_vjp = vjp(Partial(
        #     jac_func,
        #     jit(self.model_fn),
        #     xmin,    
        #     model_params['batch_stats'],  
        #     model_params['params']['featurizer']), 
        #     model_params['params']['targetizer'])

        # temp1 = jit(f_vjp)(temp2)[0]
        # print("vjp", temp1)
        del temp2
        clear_caches()

        return temp1
    
    def compute_GEPoolGoal(
            self,  
            inputs:jnp.ndarray, 
            targets:jnp.ndarray, 
            model_fn:Callable, 
            model_params:FrozenDict
        ) -> List[Union[jnp.ndarray, None]]:  
        """Compute gradients of expected training loss of x
        x is of dimension (d,)
        output: 
        """ 
        # x = x.reshape(-1, self.dim) # now x is of (nx,d)
        GEPoolGoal = grad(compute_loss_vectorize_single, argnums=5)(
            inputs, 
            targets,
            model_fn, 
            model_params['batch_stats'],
            model_params['params']['featurizer'], 
            *ravel_pytree(model_params['params']['targetizer']))
        return GEPoolGoal 
    
    def compute_GEPoolGoal_grad_and_jac(
            self,  
            inputs:jnp.ndarray, 
            targets:jnp.ndarray, 
            model_fn:Callable, 
            model_params:FrozenDict
        ) -> List[Union[jnp.ndarray, None]]:  
        """Compute gradients of expected training loss of x
        x is of dimension (d,)
        output: 
        """  
        GEPoolGoal, GEPoolGoal_x = value_and_jacfwd(grad(compute_loss_vectorize_single, argnums=5), 0)(
            inputs, 
            targets,
            model_fn, 
            model_params['batch_stats'],
            model_params['params']['featurizer'], 
            *ravel_pytree(model_params['params']['targetizer']))
        return GEPoolGoal, GEPoolGoal_x
    
    def compute_influence(
            self, 
            x:jnp.ndarray, 
            model_fn:Callable, 
            model_params:FrozenDict, 
            HinvGETasks:jnp.ndarray, 
            weights:float=1., 
        ) -> jnp.ndarray:  

        y0hat = jit(self.ensmodel_fn)(x) 
        GEPoolGoals = tree_map(
            lambda j: self.compute_GEPoolGoal(
                inputs=x, 
                targets=y0hat,
                model_fn=model_fn,
                model_params={
                    'params'     : model_params['params']['MLP_'+str(j)],
                    'batch_stats': model_params['batch_stats']['MLP_'+str(j)]
                }, 
            ), 
            list(range(self.kwargs['n_candidate_model']))
        )
        GEPoolGoals = jnp.stack(GEPoolGoals, axis=0) # (n_candidate_model, d_theta) 

        fval = - vmap(lambda x, y: jit(jnp.vdot)(x, y))(HinvGETasks, GEPoolGoals) @ weights
         
        del GEPoolGoals 
        clear_caches()
        return fval 
    
    def compute_influence_value_and_grad(
            self, 
            x:jnp.ndarray,
            model_fn:Callable, 
            model_params:FrozenDict, 
            HinvGETasks:jnp.ndarray, 
            weights:jnp.ndarray, 
        ) -> jnp.ndarray: 

        y0hat = jit(self.ensmodel_fn)(x) 

        GEPoolGoal_output = tree_map(
            lambda j: self.compute_GEPoolGoal_grad_and_jac(
                inputs=x, 
                targets=y0hat,
                model_fn=model_fn,
                model_params={
                    'params': model_params['params']['MLP_'+str(j)],
                    'batch_stats': model_params['batch_stats']['MLP_'+str(j)]
                }, 
            ), 
            list(range(self.kwargs['n_candidate_model']))
        )
        GEPoolGoals   = jnp.stack([v for v, _ in GEPoolGoal_output], axis=0) # (n_candidate_model, d_theta) 
        GEPoolGoal_xs = jnp.stack([v for _, v in GEPoolGoal_output], axis=0) # (n_candidate_model, d_theta, d) 
         
        fval = - vmap(lambda x, y: jit(jnp.vdot)(x, y))(HinvGETasks, GEPoolGoals) @ weights
        gval = - vmap(lambda x, y: jit(jnp.matmul)(x, y))(HinvGETasks, GEPoolGoal_xs).T @ weights
        
        del GEPoolGoals, GEPoolGoal_xs

        clear_caches()
        return fval, gval
        
    def compute_weight(self, train_x: jnp.ndarray, train_y: jnp.ndarray) -> jnp.ndarray:
        # compute ||train_y - train_y_bar||^2 / n_train 
        mss = jit(jnp.mean)((train_y - train_y.mean(axis=0))**2)
        # compute ||train_y - train_y_hat||^2 / n_train  
        if self.train_loss is None:
            train_yhat = jnp.row_stack(tree_map(
                lambda j : jit(self.model_fn)(
                    {'params': self.model_params['params']['_'.join(['MLP', str(j)])],
                     'batch_stats': self.model_params['batch_stats']['_'.join(['MLP',str(j)])]}, 
                    train_x), 
                list(range(self.kwargs['n_candidate_model']))))    # (n_candidate_model, nx) 
 
            self.train_loss = jit(jnp.mean)((train_y - train_yhat)**2, axis=-1) # (n_candidate_model, ) 
        # compute weights for each candidate model
        return jit(jax.nn.softmax)(-self.train_loss/mss) # (n_candidate_nets,) 

    def compute_optima(self):
        ######### ===== COMPUTE WEIGHTS ====== ############
        t0 = time.time()
        weights = self.compute_weight(self.train_features, self.train_labels) 
        t1 = time.time() - t0
        print("Step 0 takes {:.3f}s: Compute the weights.".format(t1))

        ######### ===== COMPUTE GRADIENT OF EXPECTED GOAL====== ############
        t0 = time.time()
        GETaskGoals = tree_map(lambda j: self.compute_GETaskGoal(
            model_params = {
                'params'      : self.model_params['params']['_'.join(['MLP',str(j)])],
                'batch_stats' : self.model_params['batch_stats']['_'.join(['MLP',str(j)])]
            }, 
            xmin         = self.xmins[j],  
            e            = self.kwargs['scaling_task']
        ), list(range(self.kwargs['n_candidate_model'])))
        GETaskGoals = jnp.stack(GETaskGoals, axis=0)
        t1 = time.time() - t0
        print("Step 1 takes {:.3f}s: Compute the gradient of expected goal for the TEST data.".format(t1)) 
        ## print info
        for kk in range(GETaskGoals.shape[0]):
            print("GETaskGoals[{}], (min, max)=({:.4f}, {:.4f}), mean ({:.4f})+/-({:.4f}) 1 std.".format(
                kk, jnp.min(GETaskGoals[kk]), jnp.max(GETaskGoals[kk]), jnp.mean(GETaskGoals[kk]), jnp.std(GETaskGoals[kk])))
        
        ######### ===== COMPUTE HESSIAN INVERSE GRADIENT OF EXPECTED GOAL ====== ############
        t0 = time.time()
        HinvGETasks = tree_map(lambda j: self.get_ihvp(
            v            = GETaskGoals[j], 
            inputs       = self.train_features,
            targets      = self.train_labels, 
            model_fn     = jit(self.model_fn),
            model_params = freeze({
                'params'      : self.model_params['params']['_'.join(['MLP',str(j)])],
                'batch_stats' : self.model_params['batch_stats']['_'.join(['MLP',str(j)])]
            }),  
            **self.kwargs
        ), list(range(self.kwargs['n_candidate_model'])))
        HinvGETasks = jnp.stack(HinvGETasks, axis=0)
        
        del GETaskGoals 
        clear_caches() 
        
        ## print info
        for kk in range(HinvGETasks.shape[0]):
            print("HinvGETasks[{}], (min, max)=({:.4f}, {:.4f}), mean ({:.4f})+/-({:.4f}) 1 std.".format(
                kk, jnp.min(HinvGETasks[kk]), jnp.max(HinvGETasks[kk]), jnp.mean(HinvGETasks[kk]), jnp.std(HinvGETasks[kk])))
        
        t1 = time.time() - t0
        print("Step 2 takes {:.3f}s: Compute the Hessian inverse vector product.".format(t1))
        
        ######### ===== COMPUTE WEIGHTED INFLUENCE SCORE ====== ############
        t0 = time.time()
        fun_infmax  = Partial(self.compute_influence, 
                              model_fn=jit(self.model_fn), 
                              model_params=self.model_params,
                              HinvGETasks=HinvGETasks, 
                              weights=weights)
        vng_infmax  = Partial(self.compute_influence_value_and_grad, 
                              model_fn=jit(self.model_fn), 
                              model_params=self.model_params,
                              HinvGETasks=HinvGETasks, 
                              weights=weights)
        
        del HinvGETasks, weights  
        clear_caches()
        
        x_Imin, Imin = global_optimization(
            fun_infmax, 
            top_k=self.kwargs['available_sample_k'],
            nstart=self.kwargs.get('search_xmin_nstart', 100), 
            method=self.kwargs.get('search_xmin_method', 'grid-search'),
            optimize_method=self.kwargs.get('search_xmin_opt_method', 'trust-constr'),  
            search_domain=self.search_domain,
            value_and_grad=vng_infmax, 
            tol=self.kwargs.get('search_xmin_opt_tol', 1e-4), 
            disp=self.kwargs.get('disp',False)
        )
        
        t1 = time.time() - t0
        print("Step 3 takes {:.3f}s: Compute optima.".format(t1))
 
        return AcquisitionBatch(
            x_Imin, 
            Imin
        ) 


    


