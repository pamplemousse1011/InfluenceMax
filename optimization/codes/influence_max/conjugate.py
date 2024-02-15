
from typing import Callable

import jax.numpy as jnp

from codes.influence_max.scipy_optimize_jax_wrapper import ScipyMinimize

def conjugate_gradient(Ax_fn: Callable[[jnp.ndarray], jnp.ndarray], 
                       b: jnp.ndarray,
                       method="trust-ncg", 
                       disp=False) -> jnp.ndarray:
    """
    Computes the solution to Ax - b = 0 by minimizing the conjugate objective
    f(x) = x^T A x / 2 - b^T x. This does not require evaluating the matrix A
    explicitly, only the matrix vector product Ax.
    - Ax_fn: A function that return Ax given x.
    - b: The vector b.

    :param method: 'Newton-CG', 'trust-ncg', 'trust-krylov'
    :param xtol: Average relative error in solution xopt acceptable for convergence.
    :param gtol: Gradient norm must be less than `gtol` before successful termination.
    :param debug_callback: An optional debugging function that reports
                           the current optimization function. Takes two parameters:
                           the current solution and a helper function that
                           evaluates the quadratic and linear parts of the conjugate
                           objective separately.
    :return: The conjugate optimization solution.
    """

    # cg_callback = None
    # if debug_callback:
    #     cg_callback = lambda x: debug_callback(x, 0.5 * np.dot(x, Ax_fn(x)), -np.dot(b, x))
    # result = fmin_ncg(f=lambda x: 0.5 * np.dot(x, Ax_fn(x)) - np.dot(b, x),
    #                   x0=np.zeros_like(b),
    #                   fprime=lambda x: Ax_fn(x) - b,
    #                   fhess_p=lambda x, p: Ax_fn(p),
    #                   callback=cg_callback,
    #                   avextol=avextol,
    #                   maxiter=maxiter,
    #                   disp=0)
    
    # result = minimize(fun=lambda x: 0.5 * np.dot(x, Ax_fn(x)) - np.dot(b, x),
    #                   x0=np.zeros_like(b),
    #                   method=method,
    #                   jac=lambda x: Ax_fn(x) - b,
    #                   hessp=lambda x, p: Ax_fn(p),
    #                   options={'disp': disp})['x']
    # result = result.astype(b.dtype)
    
    ff = ScipyMinimize(
        method=method, 
        fun=lambda x: 0.5 * jnp.dot(x, Ax_fn(x)) - jnp.dot(b, x),
        jac=lambda x: Ax_fn(x) - b,
        hessp=lambda x, p: Ax_fn(p),
        options={'disp': disp})
    
    x0 = jnp.zeros_like(b)
    output = ff.run(x0)
    result = output.params.astype(b.dtype)
       
    return result
