from typing import Callable

import jax.numpy as jnp 
from jax import grad, jvp, hessian, jit
from jax.tree_util import Partial 
from jax.flatten_util import ravel_pytree
from jax.scipy.sparse.linalg import cg
 
from codes.influence_max.hyperparam_optimization.hpo_model_module import compute_loss, compute_loss_vectorize, process_in_batches
 
from flax.core.frozen_dict import FrozenDict

from tqdm.auto import tqdm
from copy import deepcopy

from codes.influence_max.utils import data_loader
from codes.influence_max.conjugate import conjugate_gradient  
from codes.utils import generate_seed_according_to_time

 
def hvp(f, inputs, vectors):
    """Forward-over-reverse hessian-vector product"""
    return jvp(grad(f), inputs, vectors)[1]

def inverse_hvp_fn(method: str) -> Callable:
    if method == 'cg':
        return get_inverse_hvp_cg 
    elif method == 'lissa':
        return get_inverse_hvp_lissa 
    elif method == 'cg-linalg':
        return get_inverse_hvp_cg_linalg
    elif method == 'direct':
        return get_inverse_hvp_direct
    else:
        raise ValueError(f"Not supported for method={method}; should either be 'cg' or 'lissa'.")

def get_inverse_hvp_direct(
        v               : jnp.ndarray,
        inputs          : jnp.ndarray,
        targets         : jnp.ndarray,
        base_x_embedding: jnp.ndarray,
        model_fn        : Callable[[FrozenDict, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        model_vars      : FrozenDict,
        **kwargs
    ) -> jnp.ndarray:
    """Compute x = (H + lamda*I)^{-1}b, by first compute (H + lamda*I) and taking inverse 
    inputs           : (n,d)
    targets          : (n,n_base)
    base_x_embedding : (n_base,d_base)
    x                : (d_theta,)

    kwargs   
    - pin_memory: bool=False
    - num_workers: int=0
    - progress_bar: bool=True
    """
    targetizer_vector, targetizer_structure = ravel_pytree(model_vars['params']['targetizer'])
    
    H_matrix = 0.
    n_input = inputs.shape[0] 
    for i in range(n_input):
        H_matrix += process_in_batches(
            lambda be: compute_H_single(
                inputs[i],  # (d,)
                targets[i], # (n_base,)
                be,         # (d_base,)
                model_fn,  
                model_vars['batch_stats'],
                model_vars['params']['featurizer'],
                targetizer_vector,
                targetizer_structure,  
            ),
            base_x_embedding,
            n_batch=10,
            reduction="mean"
        ) 

    output = jnp.linalg.solve(H_matrix + kwargs['cg_lambda'] * jnp.eye(H_matrix.shape[0]), v)
    return output  
  
def compute_H_single(
        input           : jnp.ndarray,
        target          : jnp.ndarray,
        base_x_embedding: jnp.ndarray,
        model_fn        : Callable[[FrozenDict, jnp.ndarray], jnp.ndarray],
        batch_stats     : FrozenDict,
        featurizer      : FrozenDict,
        targetizer_vector, 
        targetizer_structure,      
    ) -> jnp.ndarray:
    """Compute (H + lamda*I)x
    
    input : (d,)
    target: (n_base,)
    base_x_embedding: (d_base,)
    x     : (d_theta,)
    """
    H_matrix = hessian(compute_loss_vectorize, argnums=6)(
        input,
        target, 
        base_x_embedding, 
        model_fn,
        batch_stats, 
        featurizer,
        targetizer_vector, 
        targetizer_structure,
    ) 
    return H_matrix

def get_inverse_hvp_cg_linalg(
        v               : jnp.ndarray,
        inputs          : jnp.ndarray,
        targets         : jnp.ndarray,
        base_x_embedding: jnp.ndarray,
        model_fn        : Callable[[FrozenDict, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        model_vars      : FrozenDict,
        **kwargs
    ) -> jnp.ndarray:
    """Compute x = H^{-1}v, i.e., Hx = v, then apply regularization and return H^{-1}v + lamda*v
    inputs           : (n,d)
    targets          : (n,n_base)
    base_x_embedding : (n_base,d_base)
    x                : (d_theta,)
    """

    # targetizer_vector, targetizer_structure = ravel_pytree(model_vars['params']['targetizer'])
    # ## Compute Hv
    # Ax_fn=Partial(compute_Hx, 
    #     inputs,
    #     targets, 
    #     base_x_embedding,
    #     model_fn, 
    #     model_vars['batch_stats'],
    #     model_vars['params']['featurizer'],
    #     targetizer_vector, 
    #     targetizer_structure,
    # )
    
    # ## Compute H^{-1}v
    # out = cg(Ax_fn, v)
    # # if kwargs['disp']:
    # #     print(out.info)
    # # x = out.x

    # ## Return (H^{-1}+lambda)*v = H^{-1}v + lambda*v
    # return out[0]  + kwargs.get('cg_scaling', 0.001) * v
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
    
    targetizer_vector, targetizer_structure = ravel_pytree(model_vars['params']['targetizer'])
    ## Compute (HH + lambda*I)v
    Ax_fn=Partial(compute_H2x, 
        inputs,
        targets, 
        base_x_embedding,
        model_fn, 
        model_vars['batch_stats'],
        model_vars['params']['featurizer'],
        targetizer_vector, 
        targetizer_structure,
        kwargs['cg_lambda']
    )
    ## Compute b=Hv  
    b=compute_b(
        inputs,
        targets,
        base_x_embedding,
        model_fn,
        model_vars['batch_stats'],
        model_vars['params']['featurizer'],
        targetizer_vector, 
        targetizer_structure,
        v
    )
    ## Compute H^{-1}v
    out = cg(Ax_fn, b)
    # if kwargs['disp']:
    #     print(out.info)
    # x = out.x

    ## Return (H^{-1}+lambda)*v = H^{-1}v + lambda*v
    return out[0] + kwargs.get('cg_scaling', 0.) * v

def compute_Hx_single(
    input           : jnp.ndarray,
    target          : jnp.ndarray,
    base_x_embedding: jnp.ndarray,
    model_fn        : Callable[[FrozenDict, jnp.ndarray], jnp.ndarray],
    batch_stats     : FrozenDict,
    featurizer      : FrozenDict,
    targetizer_vector, 
    targetizer_structure,
    x               : jnp.ndarray
    ) -> jnp.ndarray:
    """Compute (H + lamda*I)x
    
    input : (d,)
    target: (n_base,)
    base_x_embedding: (d_base,)
    x     : (d_theta,)
    """
    ## Compute Hx
    targetizer = targetizer_structure(targetizer_vector)
    x_pytree   = targetizer_structure(x)
    xhp = hvp(Partial(
        compute_loss, 
        input,
        target, 
        base_x_embedding, 
        model_fn,
        batch_stats, 
        featurizer), 
        (targetizer,),
        (x_pytree,)    
    ) 
    ## Return HHx  
    return ravel_pytree(xhp)[0] 

def compute_Hx(
    inputs           : jnp.ndarray,
    targets          : jnp.ndarray,
    base_x_embedding : jnp.ndarray,
    model_fn         : Callable[[FrozenDict, jnp.ndarray], jnp.ndarray],
    batch_stats      : FrozenDict,
    featurizer       : FrozenDict,
    targetizer_vector, 
    targetizer_structure, 
    x                : jnp.ndarray
    ) -> jnp.ndarray:
    """Compute Hb
    
    inputs           : (n,d)
    targets          : (n,n_base)
    base_x_embedding : (n_base,d_base)
    x                : (d_theta,)
    """

    xhp = 0.
    n_input = inputs.shape[0]

    for i in range(n_input):
        xhp += process_in_batches(
            lambda be: compute_Hx_single(
                inputs[i],  # (d,)
                targets[i], # (n_base,)
                be,         # (d_base,)
                model_fn,  
                batch_stats,
                featurizer, 
                targetizer_vector,
                targetizer_structure, 
                x,
            ),
            base_x_embedding,
            n_batch=1,
            reduction="mean"
        ) 
    
    ## Return Hx 
    # output = tree_map(lambda w1, w2: _regularizer(w1, w2, lamda), xh2p, x)
    # output = _regularizer(xhp/n_input, x, lamda)
    return xhp/n_input

def get_inverse_hvp_lissa(
        v               : jnp.ndarray,
        inputs          : jnp.ndarray,
        targets         : jnp.ndarray,
        base_x_embedding: jnp.ndarray,
        model_fn        : Callable[[FrozenDict, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        model_vars      : FrozenDict,
        **kwargs
    ) -> jnp.ndarray:     
    """
    inputs           : (n,d)
    targets          : (n,n_base)
    base_x_embedding : (n_base,d_base)
    x                : (d_theta,)

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
    _, targetizer_structure = ravel_pytree(model_vars['params']['targetizer'])

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
                hvpres = jnp.array(0.)
                for i in range(curr_x.shape[0]):
                    hvpres += process_in_batches(
                        lambda be: compute_b_single(
                            curr_x[i], 
                            curr_y[i],
                            be,  
                            model_fn, 
                            model_vars['batch_stats'], 
                            model_vars['params']['featurizer'],
                            model_vars['params']['targetizer'], 
                            targetizer_structure(estimate_curr)
                        ),
                        base_x_embedding,
                        n_batch=1,
                        reduction="mean"
                    )
                hvpres /= curr_x.shape[0]
                
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
                hvpres = jnp.array(0.)
                for i in range(curr_x.shape[0]):
                    hvpres += process_in_batches(
                        lambda be: compute_b_single(
                            curr_x[i], 
                            curr_y[i],
                            be,  
                            model_fn, 
                            model_vars['batch_stats'], 
                            model_vars['params']['featurizer'],
                            model_vars['params']['targetizer'], 
                            targetizer_structure(estimate_curr)
                        ),
                        base_x_embedding,
                        n_batch=1,
                        reduction="mean"
                    )
                hvpres /= curr_x.shape[0]
                
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
    base_x_embedding: jnp.ndarray,
    model_fn        : Callable[[FrozenDict, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    model_vars      : FrozenDict,
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
    targetizer_vector, targetizer_structure = ravel_pytree(model_vars['params']['targetizer'])
     
    ## Compute (HH + lambda*I)v
    Ax_fn=Partial(compute_H2x, 
        inputs,
        targets, 
        base_x_embedding,
        model_fn, 
        model_vars['batch_stats'],
        model_vars['params']['featurizer'],
        targetizer_vector, 
        targetizer_structure,
        kwargs['cg_lambda']
    ) 
    ## Compute b=Hv  
    b=compute_b(
        inputs,
        targets,
        base_x_embedding,
        model_fn,
        model_vars['batch_stats'],
        model_vars['params']['featurizer'],
        targetizer_vector, 
        targetizer_structure,
        v
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

def compute_H2x_single(
    input           : jnp.ndarray,
    target          : jnp.ndarray,
    base_x_embedding: jnp.ndarray,
    model_fn        : Callable[[FrozenDict, jnp.ndarray], jnp.ndarray],
    batch_stats     : FrozenDict,
    featurizer      : FrozenDict,
    targetizer_vector, 
    targetizer_structure,
    x               : jnp.ndarray
) -> jnp.ndarray:
    """Compute (HH + lamda*I)x
    
    input : (d,)
    target: (n_base,)
    base_x_embedding: (d_base,)
    x     : (d_theta,)
    """
    ## Compute Hx
    targetizer = targetizer_structure(targetizer_vector)
    x_pytree   = targetizer_structure(x)
    xhp = hvp(Partial(
        compute_loss, 
        input,
        target, 
        base_x_embedding, 
        model_fn,
        batch_stats, 
        featurizer), 
        (targetizer,),
        (x_pytree,)    
    )
    ## Compute H(Hx)
    xh2p = hvp(Partial(
        compute_loss, 
        input,
        target, 
        base_x_embedding, 
        model_fn, 
        batch_stats,
        featurizer), 
        (targetizer,), 
        (xhp,)
    )
    ## Return HHx  
    return ravel_pytree(xh2p)[0] 

def compute_H2x(
    inputs           : jnp.ndarray,
    targets          : jnp.ndarray,
    base_x_embedding : jnp.ndarray,
    model_fn         : Callable[[FrozenDict, jnp.ndarray], jnp.ndarray],
    batch_stats      : FrozenDict,
    featurizer       : FrozenDict,
    targetizer_vector, 
    targetizer_structure,
    lamda            : float,
    x                : jnp.ndarray
) -> jnp.ndarray:
    """Compute (HH + lamda*I)x
    
    inputs           : (n,d)
    targets          : (n,n_base)
    base_x_embedding : (n_base,d_base)
    x                : (d_theta,)
    """

    xh2p = 0.
    n_input = inputs.shape[0]

    for i in range(n_input):
        xh2p += process_in_batches(
            lambda be: compute_H2x_single(
                inputs[i],  # (d,)
                targets[i], # (n_base,)
                be,         # (d_base,)
                model_fn,  
                batch_stats,
                featurizer, 
                targetizer_vector,
                targetizer_structure, 
                x,
            ),
            base_x_embedding,
            n_batch=1,
            reduction="mean"
        ) 

    ## Return (H2 + lamda*I)x 
    # output = tree_map(lambda w1, w2: _regularizer(w1, w2, lamda), xh2p, x)
    output = _regularizer(xh2p/n_input, x, lamda)
    return output  

def compute_b_single(
    input            : jnp.ndarray,
    target           : jnp.ndarray,
    base_x_embedding : jnp.ndarray,
    model_fn         : Callable[[FrozenDict, jnp.ndarray], jnp.ndarray],
    batch_stats      : FrozenDict,
    featurizer       : FrozenDict,
    targetizer       : FrozenDict,
    v_pytree         : FrozenDict
) -> jnp.ndarray:
    """Compute b=Hv 
    
    input : (d,)
    target: (n_base,)
    base_x_embedding: (d_base,)
    v     : (d_theta,)
    """ 
    return ravel_pytree(
        hvp(Partial(compute_loss, 
                    input, 
                    target,
                    base_x_embedding,  
                    model_fn, 
                    batch_stats,
                    featurizer), 
        (targetizer,), 
        (v_pytree,)
    ))[0]

def compute_b(
    inputs           : jnp.ndarray,
    targets          : jnp.ndarray,
    base_x_embedding : jnp.ndarray,
    model_fn         : Callable[[FrozenDict, jnp.ndarray], jnp.ndarray],
    batch_stats      : FrozenDict,
    featurizer       : FrozenDict,
    targetizer_vector, 
    targetizer_structure,
    v                : jnp.ndarray,
) -> jnp.ndarray: 
    """Compute b=Hv 
    
    inputs           : (n,d)
    targets          : (n,n_base)
    base_x_embedding : (n_base,d_base)
    v                : (d_theta,)
    """
    n_input = inputs.shape[0]

    targetizer = targetizer_structure(targetizer_vector)
    v_pytree = targetizer_structure(v)
    
    b = 0.
    for i in range(n_input):
        b += process_in_batches(
            lambda be: compute_b_single(
                inputs[i], 
                targets[i],
                be,  
                model_fn, 
                batch_stats,
                featurizer, 
                targetizer, 
                v_pytree
            ),
            base_x_embedding,
            n_batch=1,
            reduction="mean"
        )
    return b/n_input
     
    
