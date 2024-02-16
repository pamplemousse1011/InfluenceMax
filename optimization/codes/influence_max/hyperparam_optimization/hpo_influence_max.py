import numpy as np
import jax
from jax import jvp, jit, jacfwd, jacrev, vmap, clear_caches
import jax.numpy as jnp
from jax.tree_util import Partial, tree_map
from jax.flatten_util import ravel_pytree
   
from flax.core.frozen_dict import FrozenDict

import dataclasses 

from typing import Sequence, Callable, List, Union
import time  
from codes.influence_max.hyperparam_optimization.hpo_ihvp import inverse_hvp_fn
from codes.influence_max.model_module import compute_enspred, intermediate_grad_fn
from codes.influence_max.hyperparam_optimization.hpo_model_module import process_in_batches
from codes.influence_max.hyperparam_optimization.hpo_model_module import compute_enspred_grad_on_x_SINGLE 
from codes.influence_max.hyperparam_optimization.hpo_model_module import compute_loss_grad_and_jac, compute_loss_grad_on_vars
from codes.influence_max.global_optimizer import global_optimization 
  
@dataclasses.dataclass
class AcquisitionBatch:
    samples : Sequence[float] 
    scores  : Sequence[float]

 
class InfluenceMax(object):
    def __init__(self,
                 available_x            : jnp.ndarray,
                 available_y            : jnp.ndarray,
                 xmins                  : jnp.ndarray,
                 search_domain          : np.ndarray,   
                 model_fn               : Callable[[FrozenDict, jnp.ndarray, int], jnp.ndarray],
                 base_x_embedding_train : np.ndarray,
                 base_x_embedding_targt : np.ndarray,   
                 model_vars             : FrozenDict, 
                 model_vars_true        : FrozenDict, 
                 acquire_k              : int=1,
                 m_kmeansplusplus       : float=1,  
                 train_loss             : jnp.ndarray=None,
                 **kwargs):
        
        self.kwargs                 = kwargs 
        self.model_fn               = model_fn 
        self.base_x_embedding_train = base_x_embedding_train
        self.base_x_embedding_targt = base_x_embedding_targt
        self.model_vars             = model_vars  
        self.model_vars_true        = model_vars_true 
        self.dim                    = search_domain.shape[0]
        self.acquire_k              = acquire_k
        self.m_kmeansplusplus       = m_kmeansplusplus 
        self.train_loss             = train_loss
        self.optimization_mode      = True

        if kwargs['use_double']:
            self.dtype = jnp.float64
        else:
            self.dtype = jnp.float32
        self.get_ihvp = inverse_hvp_fn(kwargs['ihvp_method'])
        
        ## Dataset
        self.search_domain   = search_domain
        self.xmins           = xmins
        self.available_x     = available_x    # (n,)
        self.available_y     = available_y    # (n, n_base_train)
        self.n               = self.available_x.shape[0]

    def compute_GETaskGoal(self, model_fn:Callable, model_vars: FrozenDict, xmin:jnp.ndarray, e:float) -> FrozenDict:  
        """Compute expected gradient of model parameters on TEST loss"""
        ## Input xmin: (d,) on TEST loss
        xmin = xmin.reshape(-1)
        
        ## Compute temp3 = ∇𝑥 𝜇(𝜃,𝑥) here we use ∇𝑥 𝜇(𝜃hat,𝑥), 
        ## where 𝜇(𝜃hat,𝑥) is the jackknife estimator for the true 𝜇(𝜃,𝑥) 
        temp3 = process_in_batches(
            Partial(compute_enspred_grad_on_x_SINGLE, 
                    model_fn, 
                    self.model_vars_true, 
                    x=xmin),
            self.base_x_embedding_targt, 
            n_batch=1, 
            reduction="mean"
        ) # (d,)
        # temp3 = compute_enspred_grad_on_x_BATCH(
        #     model_fn, 
        #     self.model_vars_true, 
        #     self.base_x_embedding_targt, 
        #     xmin
        # )

        ## Compute temp2 = -[∂2/∂𝑥2 𝜇(b,𝑥)]^{−1}  \dot [∇𝑥 𝜇(𝜃,𝑥)] 
        ##               = -[∂2/∂𝑥2 𝜇(b,𝑥)]^{−1}  \dot temp3
        partial2_mu_x = process_in_batches(
            lambda be: jacfwd(jacrev(model_fn, argnums=2), argnums=2)(
                model_vars, 
                be, 
                xmin), # (d,d)
            self.base_x_embedding_targt,
            n_batch=1,
            reduction="mean"
        ) # (d,d)

        # partial2_mu_x = jacfwd(jacrev(model_fn, argnums=2), argnums=2)(
        #     model_vars, 
        #     self.base_x_embedding_targt, 
        #     xmin
        # )  # (d,d)
        
        temp2 = - jit(jnp.linalg.solve)(
            partial2_mu_x + e * jnp.eye(self.dim).astype(self.dtype), 
            temp3
        ) #(d,)  
        
        del partial2_mu_x, temp3
        clear_caches()

        ## Compute temp1 = [∂2𝜇(b,𝑥)/∂𝑥∂b] \dot temp2
        temp1 = process_in_batches(
            lambda be: jvp(Partial(
                intermediate_grad_fn,
                model_fn, 
                model_vars['batch_stats'], 
                model_vars['params']['featurizer'], 
                model_vars['params']['targetizer'],
                be), 
                (xmin,), 
                (temp2,)
            )[1],
            self.base_x_embedding_targt,
            n_batch=1,
            reduction="mean"
        )
        # temp1 = jvp(Partial(intermediate_grad_fn, 
        #                     model_fn, 
        #                     model_vars['batch_stats'], 
        #                     model_vars['params']['featurizer'],
        #                     model_vars['params']['targetizer'],
        #                     self.base_x_embedding_targt), 
        #             (xmin,), 
        #             (temp2,)
        #         )[1] 

        del temp2 

        clear_caches()
        
        return temp1
    
    def compute_GEPoolGoal(self, input:jnp.ndarray, targets:jnp.ndarray, model_fn:Callable, model_vars: FrozenDict) -> List[Callable]:  
        """Compute gradients of expected training loss of the model variables

        input      : (d,)
        targets    : (n_base_train,1)
        GEPoolGoal : (d_theta,)
        """ 
        targetizer_vector, targetizer_structure = ravel_pytree(model_vars['params']['targetizer']) 
        
        fn = Partial(
            compute_loss_grad_on_vars, 
            input, 
            model_fn=model_fn,
            batch_stats=model_vars['batch_stats'],
            featurizer=model_vars['params']['featurizer'],
            targetizer_vector=targetizer_vector,
            targetizer_structure=targetizer_structure
        )
        
        GEPoolGoals = vmap(fn)(targets, self.base_x_embedding_train) 
        return jnp.mean(GEPoolGoals, 0)
        # batch_size = 2000

        # n_input = self.base_x_embedding_train.shape[0]
        # n_batch = math.ceil(n_input / batch_size)

        # GEPoolGoals   = 0. 
        # for i in range(n_batch):
        #     lower = i*batch_size
        #     upper = min((i+1)*batch_size, n_input)
        #     grad_val = vmap(fn)(
        #         targets[lower:upper], 
        #         self.base_x_embedding_train[lower:upper])
        #     GEPoolGoals += grad_val.sum(0) 
        # return GEPoolGoals/n_input 
             
    def compute_GEPoolGoal_grad_and_jac(self, input:jnp.ndarray, targets:jnp.ndarray, model_fn:Callable, model_vars: FrozenDict) -> List[Callable]:  
        """Compute gradient of expected training loss of the model variables, and jacobian of the input on the gradients

        input        : (d,)
        targets      : (n_base_train,1)
        GEPoolGoal   : (d_theta,)
        GEPoolGoal_x : (d_theta,d)
        """
        targetizer_vector, targetizer_structure = ravel_pytree(model_vars['params']['targetizer']) 
        
        fn = Partial(
            compute_loss_grad_and_jac, 
            input, 
            model_fn=model_fn,
            batch_stats=model_vars['batch_stats'],
            featurizer=model_vars['params']['featurizer'],
            targetizer_vector=targetizer_vector,
            targetizer_structure=targetizer_structure
        )
        
        GEPoolGoals, GEPoolGoal_xs = vmap(fn)(targets, self.base_x_embedding_train)
        return jnp.mean(GEPoolGoals, 0), jnp.mean(GEPoolGoal_xs,0)
        
        # batch_size = 3000

        # n_input = self.base_x_embedding_train.shape[0]
        # n_batch = math.ceil(n_input / batch_size)

        # GEPoolGoals   = 0.
        # GEPoolGoal_xs = 0.
        # for i in range(n_batch):
        #     lower = i*batch_size
        #     upper = min((i+1)*batch_size, n_input)
        #     grad_val, jac_val = vmap(fn)(
        #         targets[lower:upper], 
        #         self.base_x_embedding_train[lower:upper])
        #     GEPoolGoals   += grad_val.sum(0)
        #     GEPoolGoal_xs += jac_val.sum(0)
        # return GEPoolGoals/n_input, GEPoolGoal_xs/n_input
         
    def compute_influence_value_and_grad(self, x: jnp.ndarray, model_fn:Callable, model_vars:FrozenDict, HinvGETasks: jnp.ndarray, weights: jnp.ndarray=1.) -> Union[jnp.ndarray, List]: 
        """Compute weighted influence value of x
        
        x           : (d,)
        HinvGETasks : (n_candidate_model, d_theta)
        weights     : (n_candidate_model,)
        """   
        
        y0hat = process_in_batches(
            Partial(
                compute_enspred, 
                model_fn, 
                self.model_vars_true, 
                x=x
            ),
            self.base_x_embedding_train
        ).reshape(-1) # (n_base_train,)

        # y0hat = compute_enspred(
        #     self.model_fn, 
        #     self.model_vars_true, 
        #     self.base_x_embedding_train, 
        #     x
        # ) # (n_base_train,)

        GEPoolGoal_output = tree_map(
            lambda j: self.compute_GEPoolGoal_grad_and_jac(
                input=x, 
                targets=y0hat,
                model_fn=model_fn,
                model_vars={
                    'params': model_vars['params']['MLP_'+str(j)],
                    'batch_stats': model_vars['batch_stats']['MLP_'+str(j)]
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
    
    def compute_influence(self, x: jnp.ndarray, model_fn:Callable, model_vars:FrozenDict, HinvGETasks: jnp.ndarray, weights: jnp.ndarray=1.) -> Union[jnp.ndarray, List]: 
        """Compute weighted influence value of x
        
        x           : (d,)
        HinvGETasks : (n_candidate_model, d_theta)
        weights     : (n_candidate_model,)
        """   
        y0hat = process_in_batches(
            Partial(
                compute_enspred, 
                model_fn, 
                self.model_vars_true, 
                x=x
            ),
            self.base_x_embedding_train
        ) # (n_base_train,)

        GEPoolGoals = tree_map(
            lambda j: self.compute_GEPoolGoal(
                input=x, 
                targets=y0hat,
                model_fn=model_fn,
                model_vars={
                    'params': model_vars['params']['MLP_'+str(j)],
                    'batch_stats': model_vars['batch_stats']['MLP_'+str(j)]
                }, 
            ), 
            list(range(self.kwargs['n_candidate_model']))
        )
        GEPoolGoals = jnp.stack(GEPoolGoals, axis=0) # (n_candidate_model, d_theta) 

        fval = - vmap(lambda x, y: jit(jnp.vdot)(x, y))(HinvGETasks, GEPoolGoals) @ weights
        
        del GEPoolGoals
        clear_caches()

        return fval
    
    def compute_weight(self, train_x: jnp.ndarray, train_y: jnp.ndarray) -> jnp.ndarray:
        # compute ||train_y - train_y_bar||^2 / n_train 
        mss = jit(jnp.mean)((train_y - train_y.mean(axis=0))**2)
        # compute ||train_y - train_y_hat||^2 / n_train  
        # if self.train_loss is None:
        #     train_yhat = jnp.vstack(tree_map(
        #         lambda j : jit(self.model_fn_train)(
        #             freeze({'params': self.allmodel_vars['params']['MLP_'+str(j)],
        #                     'batch_stats': self.allmodel_vars['batch_stats']['MLP_'+str(j)]
        #                    }),
        #             train_x), 
        #         list(range(self.kwargs['n_candidate_model']))))    # (n_candidate_model, n, n_base) 

        #     # print(train_yhat.shape)
        #     self.train_loss = jit(jnp.mean)((train_y - train_yhat)**2, axis=[-2,-1]) # (n_candidate_model, ) 
        # compute weights for each candidate model
        return jit(jax.nn.softmax)(-self.train_loss/mss/2) # (n_candidate_nets,) 

    def compute_optima(self):
        t0 = time.time()
        weights = self.compute_weight(self.available_x, self.available_y) 
        t1 = time.time() - t0
        print("Step 0 takes {:.3f}s: Compute the weights.".format(t1))
        
        t0 = time.time()
        GETaskGoals = tree_map(lambda j: self.compute_GETaskGoal( 
            model_fn=jit(self.model_fn),
            model_vars={
                'params': self.model_vars['params']['MLP_'+str(j)],
                'batch_stats': self.model_vars['batch_stats']['MLP_'+str(j)]
            }, 
            xmin       = self.xmins[j],  
            e          = self.kwargs['scaling_task']
        ), list(range(self.kwargs['n_candidate_model'])))
        GETaskGoals = jnp.stack(GETaskGoals, axis=0)          # (n_candidate_model, d_theta)
        t1 = time.time() - t0
        print("Step 1 takes {:.3f}s: Compute the gradient of expected goal for the TEST data.".format(t1))
        for kk in range(GETaskGoals.shape[0]):
            print("GETaskGoals[{}], (min, max)=({:.4f}, {:.4f}), mean ({:.4f})+/-({:.4f}) 1 std.".format(
                kk, jnp.min(GETaskGoals[kk]), jnp.max(GETaskGoals[kk]), jnp.mean(GETaskGoals[kk]), jnp.std(GETaskGoals[kk])))

        t0 = time.time() 
        HinvGETasks = tree_map(lambda j: self.get_ihvp(
            v          = GETaskGoals[j], 
            inputs     = self.available_x,             # (n, )
            targets    = self.available_y,             # (n, n_base_train)
            model_fn   = jit(self.model_fn),
            model_vars = {
                'params'     : self.model_vars['params']['MLP_'+str(j)],
                'batch_stats': self.model_vars['batch_stats']['MLP_'+str(j)]
            },  
            base_x_embedding = self.base_x_embedding_train,
            **self.kwargs
        ), list(range(self.kwargs['n_candidate_model']))) 
        
        del GETaskGoals 
        clear_caches() 

        HinvGETasks = jnp.stack(HinvGETasks, axis=0)         # (n_candidate_model, d_theta)
        for kk in range(HinvGETasks.shape[0]):
            print("HinvGETasks[{}], (min, max)=({:.4f}, {:.4f}), mean ({:.4f})+/-({:.4f}) 1 std.".format(
                kk, jnp.min(HinvGETasks[kk]), jnp.max(HinvGETasks[kk]), jnp.mean(HinvGETasks[kk]), jnp.std(HinvGETasks[kk])))

        t1 = time.time() - t0
        print("Step 2 takes {:.3f}s: Compute the Hessian inverse vector product.".format(t1))
 
      
        t0 = time.time()
        fun_infmax  = Partial(self.compute_influence, 
                              model_fn=jit(self.model_fn), 
                              model_vars=self.model_vars,
                              HinvGETasks=HinvGETasks, 
                              weights=weights)
        vng_infmax  = Partial(self.compute_influence_value_and_grad, 
                              model_fn=jit(self.model_fn), 
                              model_vars=self.model_vars,
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
        # n_rad = 11
        # all_pt = jnp.stack(
        #     [jnp.linspace(start=xmin+(xmax-xmin)/(10*n_rad), 
        #                 stop=xmax-(xmax-xmin)/(10*n_rad), 
        #                 num=n_rad,
        #                 endpoint=True
        #     ) for (xmin, xmax) in self.search_domain], 
        #     axis=0
        # )

        # initial_samples = jnp.stack(jnp.meshgrid(*all_pt), -1).reshape(-1,4)
        # x_Imin, Imin = global_optimization(
        #     fun_infmax, 
        #     top_k=self.kwargs['available_sample_k'],
        #     # nstart=self.kwargs.get('search_xmin_nstart', 100), 
        #     # method=self.kwargs.get('search_xmin_method', 'grid-search'),
        #     initial_samples=initial_samples,
        #     optimize_method="sampling", # self.kwargs.get('search_xmin_opt_method', 'trust-constr'),  
        #     search_domain=self.search_domain,
        #     # value_and_grad=vng_infmax, 
        #     tol=self.kwargs.get('search_xmin_opt_tol', 1e-4), 
        #     disp=self.kwargs.get('disp',False)
        # )
         
        t1 = time.time() - t0
        print("Step 3 takes {:.3f}s: Compute optima.".format(t1))
    
        del fun_infmax # , vng_infmax
        clear_caches

        return AcquisitionBatch(x_Imin, Imin) 


    


