import numpy as np
import jax
from jax import vjp, grad, jit, jacfwd, jacrev, vmap
import jax.numpy as jnp
from jax.tree_util import Partial, tree_map
from jax.flatten_util import ravel_pytree
  
from flax.core import freeze 
from flax.core.frozen_dict import FrozenDict

import dataclasses 

from typing import Sequence, Tuple, Callable, List, Union
import time

from opt_jax.opt_jax_model_module import jac_func, compute_loss_vectorize, compute_loss_vectorize_single
from opt_jax.opt_jax_global_optimizer import global_optimization 
from opt_utils import gc_cuda 
from opt_jax.opt_jax_ihvp import inverse_hvp_fn
from opt_jax.opt_jax_utils import value_and_jacfwd, sum_helper
# from utils_influence_max import compute_loss, mu_func# , init_centers
  
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
        _, f_vjp = vjp(Partial(
            jac_func,
            jit(self.model_fn),
            xmin,    
            model_params['batch_stats'],  
            model_params['params']['featurizer']), 
            model_params['params']['targetizer'])

        temp1  = jit(f_vjp)(temp2)[0]

        del f_vjp, temp2
        gc_cuda()

        return temp1
    
    def compute_GEPoolGoal(
            self,  
            model_params:FrozenDict,
            x:jnp.ndarray
        ) -> List[Union[jnp.ndarray, None]]:  
        """Compute gradients of expected training loss of x
        x is of dimension (d,)
        output: 
        """ 
        # x = x.reshape(-1, self.dim) # now x is of (nx,d)

        y0hat = jit(self.ensmodel_fn)(x)   
        
        GEPoolGoal, GEPoolGoal_x = value_and_jacfwd(grad(compute_loss_vectorize_single, argnums=5), 0)(
            x, 
            y0hat,
            self.model_fn, 
            model_params['batch_stats'],
            model_params['params']['featurizer'], 
            *ravel_pytree(model_params['params']['targetizer']))
        return GEPoolGoal, GEPoolGoal_x # .transpose([1,0,2])  # (param_dim,), (nx, param_dim, x_dim)
    
        # GEPoolGoal = grad(compute_loss_vectorize_single, argnums=5)(
        #     x, 
        #     y0hat,
        #     self.model_fn, 
        #     model_params['batch_stats'],
        #     model_params['params']['featurizer'], 
        #     *ravel_pytree(model_params['params']['targetizer']))
        # return GEPoolGoal

    def compute_influence(
            self, 
            x:jnp.ndarray, 
            model_params:FrozenDict, 
            HinvGETask:jnp.ndarray, 
            weight:float=1., 
        ) -> jnp.ndarray: 
        GEPoolGoal, GEPoolGoal_xs = self.compute_GEPoolGoal(model_params, x) 
        
        fval = -jit(jnp.dot)(HinvGETask, GEPoolGoal) * weight 
        gval = -jit(jnp.matmul)(HinvGETask, GEPoolGoal_xs) * weight 
        
        del GEPoolGoal, GEPoolGoal_xs
        gc_cuda()
        return fval, gval
         
    def compute_score(
            self,
            x           : jnp.ndarray,
            HinvGETasks : jnp.ndarray,
            weights     : jnp.ndarray,
        ) -> jnp.ndarray:
        ## sum the individual influence values
        return tree_map(jit(sum_helper), *tree_map(lambda kk: self.compute_influence(
            x,
            {'params'      : self.model_params['params']['_'.join(['MLP', str(kk)])],
             'batch_stats' : self.model_params['batch_stats']['_'.join(['MLP',str(kk)])]}, 
            HinvGETasks[kk],
            weights[kk]
        ), list(range(self.kwargs['n_candidate_model']))))
    
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

            # print(train_yhat.shape)
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
            model_params = freeze({'params'      : self.model_params['params']['_'.join(['MLP',str(j)])],
                                   'batch_stats' : self.model_params['batch_stats']['_'.join(['MLP',str(j)])]
                                }), 
            xmin         = self.xmins[j],  
            e            = self.kwargs['scaling_task']
        ), list(range(self.kwargs['n_candidate_model'])))
        t1 = time.time() - t0
        print("Step 1 takes {:.3f}s: Compute the gradient of expected goal for the TEST data.".format(t1)) 
         
        ######### ===== COMPUTE HESSIAN INVERSE GRADIENT OF EXPECTED GOAL ====== ############
        t0 = time.time()
        HinvGETasks = tree_map(lambda j: self.get_ihvp(
            v            = GETaskGoals[j], 
            inputs       = self.train_features,
            targets      = self.train_labels, 
            model_fn     = jit(self.model_fn),
            model_params = freeze({'params'      : self.model_params['params']['_'.join(['MLP',str(j)])],
                                   'batch_stats' : self.model_params['batch_stats']['_'.join(['MLP',str(j)])]
                                }),  
            **self.kwargs
        ), list(range(self.kwargs['n_candidate_model'])))
        HinvGETasks = jnp.stack(HinvGETasks, axis=0)
        
        del GETaskGoals 
        gc_cuda() 
        
        ## print info
        for kk in range(HinvGETasks.shape[0]):
            print("HinvGETasks[{}], (min, max)=({:.4f}, {:.4f}), mean ({:.4f})+/-({:.4f}) 1 std.".format(
                kk, jnp.min(HinvGETasks[kk]), jnp.max(HinvGETasks[kk]), jnp.mean(HinvGETasks[kk]), jnp.std(HinvGETasks[kk])))
        
        t1 = time.time() - t0
        print("Step 2 takes {:.3f}s: Compute the Hessian inverse vector product.".format(t1))
        
        ######### ===== COMPUTE WEIGHTED INFLUENCE SCORE ====== ############
        t0 = time.time()
        fun_opt_infmax = Partial(self.compute_score,
            HinvGETasks = HinvGETasks,
            weights     = weights,
        )
        del HinvGETasks, weights  
        gc_cuda()  
             
        x_Imin, Imin = global_optimization(
            top_k           = int(self.acquire_k * self.m_kmeansplusplus),
            fun_to_opt      = fun_opt_infmax,
            automate_jac    = False,
            method          = self.kwargs['search_xmin_method'],  
            search_domain   = self.search_domain,  
            nstart          = self.kwargs['search_xmin_nstart'],  
            optimize_method = self.kwargs['search_xmin_opt_method'],   
            use_double      = self.kwargs['use_double'], 
        )
        t1 = time.time() - t0
        print("Step 3 takes {:.3f}s: Compute optima.".format(t1))

        # if self.m_kmeansplusplus > 1:
        #     idx_k = init_centers(Imin, x_Imin, self.acquire_k)
        #     x_Imin = x_Imin[idx_k]
        #     Imin = Imin[idx_k]

        
        return AcquisitionBatch(
            x_Imin, 
            Imin
        ) 


    


