import jax.numpy as jnp 
from jax import grad, jvp, jit
from jax.tree_util import Partial 
from jax.flatten_util import ravel_pytree
from jax.scipy.sparse.linalg import cg
from flax.core.frozen_dict import FrozenDict

from typing import Callable 
from tqdm.auto import tqdm
from copy import deepcopy

from codes.influence_max.noisy_funct_optimization.nfo_model_module import compute_loss
from codes.influence_max.conjugate import conjugate_gradient   
from codes.influence_max.utils import data_loader
from codes.utils import generate_seed_according_to_time

def hvp(f, inputs, vectors):
    """Forward-over-reverse hessian-vector product"""
    return jvp(jit(grad(f)), inputs, vectors)[1]

def inverse_hvp_fn(method: str) -> Callable:
    if method == 'cg':
        return get_inverse_hvp_cg 
    elif method == 'lissa':
        return get_inverse_hvp_lissa 
    elif method == 'cg-linalg':
        return get_inverse_hvp_cg_linalg
    else:
        raise ValueError(f"Not supported for method={method}; should either be 'cg' or 'lissa'.")

def get_inverse_hvp_cg_linalg(
        v               : jnp.ndarray,
        inputs          : jnp.ndarray,
        targets         : jnp.ndarray, 
        model_fn        : Callable[[FrozenDict, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        model_params    : FrozenDict,
        **kwargs
    ) -> jnp.ndarray:
    """
    This function solves the optimization problem to compute H^{-1}v. 
    Then apply regularization and return H^{-1}v + lamda*v.

    Here we solve:
        H^{-1}v = argmin_y y^T (HH + lambda*I)y - (Hv)^T y
    Compared to the classic formulation 
        H^{-1}v = argmin_y y^T Hy - V^T y
    - In practice the Hessian may have negative eigenvalues, since we run a SGD 
      and the final Hessian matrix H may not at a local minimum exactly.    
    - HH is guaranteed to be positive definite as long as H is invertible, 
      even when H has negative eigenvalues
    - The rate of convergence of CG depends on the condition number of HH, 
      which can be very large if HH is ill-conditioned
      lambda here serves as a damping term to stabilize the solution and 
      to encourage faster convergence when HH is ill-conditioned
    """
    
    _, targetizer_structure = ravel_pytree(model_params['params']['targetizer'])
    
    ## Compute Hv
    Ax_fn=Partial(compute_H2x, 
        inputs,
        targets,  
        model_fn, 
        model_params['batch_stats'],
        model_params['params']['featurizer'],
        model_params['params']['targetizer'],
        kwargs['cg_lambda']
    ) 

    b=compute_b(
        inputs,
        targets,
        model_fn,
        model_params['batch_stats'],
        model_params['params']['featurizer'],
        model_params['params']['targetizer'],
        targetizer_structure(v)
    )

    ## Compute H^{-1}v
    out = cg(Ax_fn, b)

    # if kwargs['disp']:
    #     print(out.info)
    # x = out.x

    ## Return (H^{-1}+scaling)*v = H^{-1}v + scaling*v
    return out[0] + kwargs.get('cg_scaling', 0.) * v

def get_inverse_hvp_lissa(
        v               : jnp.ndarray,
        inputs          : jnp.ndarray,
        targets         : jnp.ndarray, 
        model_fn        : Callable[[FrozenDict, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        model_params    : FrozenDict,
        **kwargs
    ) -> jnp.ndarray:     
    """
    inputs           : (n,d)
    targets          : (n,) 

    kwargs
    - ihvp_batch_size: int 
    - lissa_damping: float  
    - lissa_scaling: float
    - lissa_T: int (num_samples)
    - lissa_J: int (recursion_depth)
    - pin_memory: bool=False
    - num_workers: int=0
    - progress_bar: bool=True
    """
    _, targetizer_structure = ravel_pytree(model_params['params']['targetizer'])

    out = jnp.array(0.)
    for _ in range(kwargs['lissa_T']):
        lissa_loader = data_loader(inputs, targets, 
                                   batch_size=kwargs['ihvp_batch_size'], 
                                   shuffle=True, 
                                   seed=generate_seed_according_to_time(1)[0])
        estimate_curr = deepcopy(v)
        if inputs.shape[0] > kwargs['ihvp_batch_size']: 
            lissa_iter = iter(lissa_loader)
            for _ in tqdm(range(kwargs['lissa_J']), disable=not kwargs['progress_bar']):
                try:
                    curr_x, curr_y = next(lissa_iter)
                except StopIteration:
                    lissa_iter = iter(lissa_loader)
                    curr_x, curr_y = next(lissa_iter)
                
                """
                curr_x: (b,d)
                curr_y: (b,n_base)
                base_x_embedding: (n_base,d_base)
                """ 
                hvpres = compute_b(
                    inputs      = curr_x,
                    targets     = curr_y,
                    model_fn    = model_fn,
                    batch_stats = model_params['batch_stats'], 
                    featurizer  = model_params['params']['featurizer'],
                    targetizer  = model_params['params']['targetizer'], 
                    v           = targetizer_structure(estimate_curr)
                )
                 
                estimate_curr = (
                    v 
                    + (1-kwargs['lissa_damping']) * estimate_curr
                    - kwargs['lissa_scaling'] * hvpres
                )
                 
        else:
            curr_x, curr_y = next(iter(lissa_loader))
            for _ in tqdm(range(kwargs['lissa_J']), disable=not kwargs['progress_bar']):
                """
                curr_x: (b,d)
                curr_y: (b,n_base)
                base_x_embedding: (n_base,d_base)
                """ 
                hvpres = compute_b(
                    inputs      = curr_x,
                    targets     = curr_y,
                    model_fn    = model_fn,
                    batch_stats = model_params['batch_stats'], 
                    featurizer  = model_params['params']['featurizer'],
                    targetizer  = model_params['params']['targetizer'], 
                    v           = targetizer_structure(estimate_curr)
                )
                
                estimate_curr = (
                    v 
                    + (1-kwargs['lissa_damping']) * estimate_curr
                    - kwargs['lissa_scaling'] * hvpres
                )

        out += kwargs['lissa_scaling'] * estimate_curr
        
    return out / kwargs['lissa_T']

def get_inverse_hvp_cg(
    v               : jnp.ndarray,
    inputs          : jnp.ndarray,
    targets         : jnp.ndarray,
    model_fn        : Callable[[FrozenDict, jnp.ndarray], jnp.ndarray],
    model_params    : FrozenDict,
    **kwargs
) -> jnp.ndarray: 
    """
    This function solves the optimization problem to compute H^{-1}v. 
    Here we solve:
        H^{-1}v = argmin_y y^T (HH + lambda*I)y - (Hv)^T y
    Compared to the classic formulation 
        H^{-1}v = argmin_y y^T Hy - V^T y
    - In practice the Hessian may have negative eigenvalues, since we run a SGD 
      and the final Hessian matrix H may not at a local minimum exactly.    
    - HH is guaranteed to be positive definite as long as H is invertible, 
      even when H has negative eigenvalues
    - The rate of convergence of CG depends on the condition number of HH, 
      which can be very large if HH is ill-conditioned
      lambda here serves as a damping term to stabilize the solution and 
      to encourage faster convergence when HH is ill-conditioned
    """
    _, targetizer_structure = ravel_pytree(model_params['params']['targetizer'])

    ## Compute Hv
    Ax_fn=Partial(compute_H2x, 
        inputs,
        targets,  
        model_fn, 
        model_params['batch_stats'],
        model_params['params']['featurizer'],
        model_params['params']['targetizer'],
        kwargs['cg_lambda']
    ) 

    b=compute_b(
        inputs,
        targets,
        model_fn,
        model_params['batch_stats'],
        model_params['params']['featurizer'],
        model_params['params']['targetizer'],
        targetizer_structure(v)
    )

    # not jit in conjugate_gradient
    result = conjugate_gradient(
        Ax_fn, 
        b, 
        method=kwargs['cg_method'], 
        disp=kwargs['disp']
    )
    return jnp.array(result)
 
def _regularizer(xh2p, x, lamda):
    """Compute (H2 + lamda*I)x"""
    return xh2p + lamda*x

def compute_H2x(
    inputs     : jnp.ndarray,
    targets    : jnp.ndarray,
    model_fn   : Callable[[FrozenDict, jnp.ndarray], jnp.ndarray],
    batch_stats: FrozenDict,
    featurizer : FrozenDict,
    targetizer : FrozenDict,
    lamda      : float,
    x          : jnp.ndarray
) -> jnp.ndarray:
    """Compute (HH + lamda*I)x"""
    
    _, target_structure = ravel_pytree(targetizer)
    x_pytree = target_structure(x)
    ## Compute Hx
    xhp = hvp(Partial(
        compute_loss, 
        inputs,
        targets,  
        model_fn, 
        batch_stats,
        featurizer), 
        (targetizer,),
        (x_pytree,)    
    )
    ## Compute H(Hx)
    xh2p = hvp(Partial(
        compute_loss, 
        inputs,
        targets,  
        model_fn,
        batch_stats, 
        featurizer), 
        (targetizer,), 
        (xhp,)
    )
    ## Return (H2 + lamda*I)x 
    # output = tree_map(lambda w1, w2: helper_Hx(w1, w2, lamda), xh2p, x)
    output = _regularizer(ravel_pytree(xh2p)[0], x, lamda)
    return output  

def compute_b(
    inputs     : jnp.ndarray,
    targets    : jnp.ndarray,
    model_fn   : Callable[[FrozenDict, jnp.ndarray], jnp.ndarray],
    batch_stats: FrozenDict,
    featurizer : FrozenDict,
    targetizer : FrozenDict,
    v          : FrozenDict,
) -> jnp.ndarray:
    ## Return b=Hv 
    b = hvp(Partial(compute_loss, 
                    inputs, 
                    targets, 
                    model_fn, 
                    batch_stats,
                    featurizer), 
        (targetizer,), 
        (v,)
    )
    return ravel_pytree(b)[0]