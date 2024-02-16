# https://github.com/google/jax/blob/a131828194a1a69195adb970f35903a2208eb2f1/jax/api.py#L651
from typing import (Callable, Union, Sequence)
from tqdm.auto import tqdm
from functools import partial

import torch
from torch import Tensor
import numpy as np

from jax import random, linear_util as lu 
from jax._src.api_util import argnums_partial  
from jax import vmap  
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_map

from jax._src.api import ( 
    _check_input_dtype_jacfwd,
    _check_output_dtype_jacfwd,
    _std_basis,
    _jacfwd_unravel,
    _jvp
)

def zero_one_normalization(X:Union[np.ndarray, jnp.ndarray, Tensor], lower=None, upper=None):
    if isinstance(X, np.ndarray):
        if lower is None:
            lower = np.min(X, axis=0)
        if upper is None:
            upper = np.max(X, axis=0)
        X_normalized = np.true_divide((X - lower), (upper - lower))

    elif isinstance(X, jnp.ndarray):
        if lower is None:
            lower = jnp.min(X, axis=0)
        if upper is None:
            upper = jnp.max(X, axis=0)
        X_normalized = jnp.true_divide((X - lower), (upper - lower))

    elif isinstance(X, Tensor):
        if lower is None:
            lower = jnp.min(X, axis=0)
        if upper is None:
            upper = jnp.max(X, axis=0)
        X_normalized = torch.true_divide((X - lower), (upper - lower))

    else:
        raise ValueError("Unrecognized array. Can only be 'Tensor', 'numpy.ndarray' or 'jax.numpy.ndarray'.")
        
    return X_normalized, lower, upper

def zero_mean_unit_var_normalization(X:Union[np.ndarray, jnp.ndarray, Tensor], mean=None, std=None):
    if isinstance(X, np.ndarray):
        if mean is None:
            mean = np.mean(X, axis=0)
        if std is None:
            std = np.std(X, axis=0)

        X_normalized = np.true_divide(X - mean, std)

    elif isinstance(X, jnp.ndarray):
        if mean is None:
            mean = jnp.mean(X, axis=0)
        if std is None:
            std = jnp.std(X, axis=0)

        X_normalized = jnp.true_divide(X - mean, std)

    elif isinstance(X, Tensor):
        if mean is None:
            mean = torch.mean(X, dim=0)
        if std is None:
            std = torch.std(X, dim=0)

        X_normalized = torch.true_divide(X - mean, std)

    else:
        raise ValueError("Unrecognized array. Can only be 'Tensor', 'numpy.ndarray' or 'jax.numpy.ndarray'.")
    
    return X_normalized, mean, std

def batch_min(fn, x:jnp.ndarray, batch_size:int=8192, disp:bool=False, description="Influence"):
    fval = batch_compute(fn, x, batch_size, disp)
    idx = lax.argmin(fval, axis=0, index_dtype=int).reshape(-1)
    print("{}, (min, max)=({:.4f}, {:.4f}), mean ({:.4f})+/-({:.4f}) 1 std.".format(
        description, jnp.min(fval), jnp.max(fval), jnp.mean(fval), jnp.std(fval)))
    return x[idx], fval[idx]

def batch_compute(fn: Callable[[jnp.ndarray], jnp.ndarray], x:jnp.ndarray, batch_size:int=8192, disp:bool=False):
    n = x.reshape(-1, x.shape[-1]).shape[0]
    n_batches = n // batch_size
     
    batches = jnp.array_split(x.reshape(n, -1), n_batches, axis=0)

    output = []
    for batch in tqdm(batches, disable=not disp):
        output.append(fn(batch))

    return jnp.concatenate(output, axis=0).reshape(*x.shape[:-1])

def reject_outliers(x:np.ndarray, iq_range:float=0.7):
    pcnt = (1 - iq_range) / 2
    qlow, median, qhigh = np.quantile(x, [pcnt, 0.50, 1-pcnt])
    iqr = qhigh - qlow
    return x[np.abs(x - median) <= iqr]

def sum_helper(*x):
    return jnp.vstack(x).sum(0)

def row_stack_helper(*x):
    return jnp.vstack(x)

def value_and_jacfwd(fun: Callable, argnums: Union[int, Sequence[int]] = 0,
                     has_aux: bool = False, holomorphic: bool = False):
    """Creates a function which evaluates both `fun` and the Jacobian of `fun`.

    The Jacobian of `fun` is evaluated column-by-column using forward-mode AD.

    Args:
        fun: Function whose Jacobian is to be computed.
        argnums: Optional, integer or tuple of integers. Specifies which positional
        argument(s) to differentiate with respect to (default `0`).
        holomorphic: Optional, bool. Indicates whether `fun` is promised to be
        holomorphic. Default False.

    Returns:
        A function with the same arguments as `fun`, that evaluates both `fun` and
        the Jacobian of `fun` using forward-mode automatic differentiation, and
        returns them as a two-element tuple `(val, jac)` where `val` is the
        value of `fun` and `jac` is the Jacobian of `fun`.

    >>> def f(x):
    ...   return jax.numpy.asarray(
    ...     [x[0], 5*x[2], 4*x[1]**2 - 2*x[2], x[2] * jax.numpy.sin(x[0])])
    ...
    ... val, jac = jax.value_and_jacfwd(f)(np.array([1., 2., 3.])))
    ...
    >>> print(val)
    [ 1.         15.         10.          2.52441295]
    >>> print(jac)
    [[ 1.        ,  0.        ,  0.        ],
    [ 0.        ,  0.        ,  5.        ],
    [ 0.        , 16.        , -2.        ],
    [ 1.6209068 ,  0.        ,  0.84147096]]
    """
    
    def jacfun(*args, **kwargs):
        f = lu.wrap_init(fun, kwargs)
        f_partial, dyn_args = argnums_partial(f, argnums, args,
                                              require_static_args_hashable=False)
        tree_map(partial(_check_input_dtype_jacfwd, holomorphic), dyn_args)
        if not has_aux:
            pushfwd: Callable = partial(_jvp, f_partial, dyn_args)
            y, jac = vmap(pushfwd, out_axes=(None, -1))(_std_basis(dyn_args))
            aux = None 
        else:
            pushfwd: Callable = partial(_jvp, f_partial, dyn_args, has_aux=True)
            y, jac, aux = vmap(pushfwd, out_axes=(None, -1, None))(_std_basis(dyn_args))
        tree_map(partial(_check_output_dtype_jacfwd, holomorphic), y)

        example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
        jac_tree = tree_map(partial(_jacfwd_unravel, example_args), y, jac)
         
        if has_aux:
            return (y, aux), jac_tree
        else:
            return y, jac_tree

    return jacfun

def data_loader(x:jnp.ndarray, y:jnp.ndarray, batch_size:int, shuffle:bool=False, seed:int=1):
    """
    A generator function to yield batches of data.

    Parameters:
    x (array): Input data of shape (n, ...)
    y (array): Labels or targets of shape (n, ...)
    batch_size (int): Size of each batch
    shuffle (bool): Whether to shuffle the data before creating batches

    Yields:
    Batches of (batch_x, batch_y)
    """
    n = x.shape[0]
    indices = jnp.arange(n)
    if shuffle:
        key = random.PRNGKey(seed)
        key, subkey = random.split(key)
        indices = random.permutation(subkey, indices)

    for start_idx in range(0, n, batch_size):
        end_idx = min(start_idx + batch_size, n)
        batch_indices = indices[start_idx:end_idx]
        batch_x = x[batch_indices]
        batch_y = y[batch_indices]
        yield batch_x, batch_y
