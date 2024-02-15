# See Cornell-MOE/pes/PES/global_optimization.py
# See https://github.com/weichiyao/RoBO/blob/c8ac6b19ece752c68c4f568d3da392a2d567925a/robo/maximizers/scipy_optimizer.py#L11
from typing import Tuple, Callable, Union
from scipy import stats
from sklearn.metrics import pairwise_distances

import numpy as np 

import jax.numpy as jnp 
from jax import lax
from jax.tree_util import tree_map

# from jaxopt import ScipyBoundedMinimize
from codes.influence_max.scipy_optimize_jax_wrapper import ScipyBoundedMinimize
from codes.influence_max.utils import row_stack_helper
from codes.data_modules.data_generator import generate_samples

import time
from tqdm.auto import tqdm

## kmeans ++ initialization
def init_centers(y:np.ndarray, x:np.ndarray, K:int) -> np.ndarray:
    ind = np.argmax(np.array(y))
    mu = [x[ind]]
    indsAll = [ind]
    # centInds = [0.] * len(X)
    # cent = 0
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(x, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(x, [mu[-1]]).ravel().astype(float)
            for i in range(len(x)):
                if D2[i] >  newD[i]:
                    # centInds[i] = cent
                    D2[i] = newD[i]
        # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        # if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll: ind = customDist.rvs(size=1)[0] # regenerate if in the pool
        mu.append(x[ind])
        indsAll.append(ind)
        # cent += 1
    return np.stack(mu, axis=0)

def get_opt_result(fun:ScipyBoundedMinimize, bounds:Tuple[jnp.ndarray, jnp.ndarray], x0:jnp.ndarray):
    x0 = jnp.clip(x0, bounds[0], bounds[1])
    output = fun.run(x0, bounds = bounds) 
    return jnp.array(output.params, dtype=x0.dtype), jnp.array([output.state.fun_val],dtype=x0.dtype)

def global_optimization(
    fun_to_opt      : Callable[[jnp.ndarray], jnp.ndarray],
    top_k           : int=1,
    method          : str='multi-start',  
    search_domain   : np.ndarray=None,  
    nstart          : int=500, 
    initial_samples : np.ndarray=None, 
    optimize_method : str='sampling-lhs',   
    use_double      : bool=False,
    value_and_grad  : Union[bool, Callable]=False,
    tol             : float=1e-4,
    disp            : bool=False
) -> Tuple[np.ndarray, np.ndarray]: 
    """Function to find the global minimum of the input function. 
    ARGUMENTS
    ==============
    method: 'grid-search' or 'multi-start'
        - 'grid-search': We first divide the domain into grids 
                         and then evaluate the input function at 
                         the grid points. 
                         Furthermore, we use the grid point with 
                         minimum nstart input function values 
                         as our starting points to put into the 
                         optimization function.
        - 'multi-start': Randomly sample initializations 
    optimize_method: 'sampling-grid', 'sampling-lhs', 'sampling-grid', 'L-BFGS-B', 'TNC', 'SLSQP', 'Powell', or 'trust-constr'
        method used to optimize influence function to acquire the next data points.
    funct_to_opt: 
        the input function to optimize
    search_domain: numpy ndarray of size (d,2)
        lower bounds and upper bounds for each dimension
    nstart: int
        the number of times for optimization, each with a different x0
    initial_samples (and initial_yvalues):
        - 'multi-start': if not None: generate half of the nstart starting points randomly
                         generate the rest around the initial_samples;
                         else: generate nstart random samples (default with LHS)
        - 'grid-search': if None, generate grid points and obtain the initial_yvalues
                         select nstart centers for optimization.
    top_k: int
        return top k solutions
      
    RETURNS 
    ==============
    x: (top_k,d)
        minimizer
    y: (1,)
        corresponding minimum value
    """
    d = search_domain.shape[0]
    if use_double:
        dtype = jnp.float64
    else:
        dtype = jnp.float32
    
    assert top_k == 1, "Current implementation only allows for top_k=1 (no batch)"

    if optimize_method.split('-')[0] == 'sampling':
        if initial_samples is None:
            try:
                sampling_method = optimize_method.split('-')[1]
            except:
                sampling_method = 'lhs'
            initial_samples = generate_samples(
                n=8192*8, 
                search_domain=search_domain, 
                method=sampling_method
            )
        fvals = []
        for xx in tqdm(range(initial_samples), desc="Optimization by sampling", disable=not disp):
            fval_curr = fun_to_opt(xx)
            fvals.append(fval_curr)
        fvals = jnp.hstack(fvals)

        idx = lax.argmin(fvals, axis=0, index_dtype=int).reshape(-1)
        return initial_samples[idx], fvals[idx]
    
    assert method in ['multi-start','grid-search'], "`method` should be either 'multi-start' or 'grid-search'" 
    
    if initial_samples is not None:
        # generate starting points randomly
        starts = generate_samples(int(nstart*top_k*0.5), search_domain)
        # generate starting points around the initial_samples
        rand_incs = jnp.array(
            [np.random.normal(loc=initial_samples, scale=np.ones([d]) * 0.5) 
            for _ in range(int(nstart*top_k*0.5))],
            dtype=dtype
        )
        starts = jnp.append(starts, rand_incs, axis=0)
    else:
        starts = jnp.array(generate_samples(nstart*top_k, search_domain, 'lhs'), dtype=dtype)

    # if method == 'multi-start':
    #     if initial_samples is not None:
    #         # generate starting points randomly
    #         starts = generate_samples(int(nstart*top_k*0.5), search_domain)
    #         # generate starting points around the initial_samples
    #         rand_incs = jnp.array(
    #             [np.random.normal(loc=initial_samples, scale=np.ones([d]) * 0.5) 
    #             for _ in range(int(nstart*top_k*0.5))],
    #             dtype=dtype
    #         )
    #         starts = jnp.append(starts, rand_incs, axis=0)
    #     else:
    #         starts = jnp.array(generate_samples(nstart*top_k, search_domain, 'lhs'), dtype=dtype)
    # else:
    #     if initial_samples is None:
    #         # generate starting points randomly
    #         initial_samples = jnp.array(
    #             generate_samples(nstart, search_domain, 'grid'), 
    #             dtype=dtype) # (5**ndim, ndim)
    #         initial_yvalues = vmap(fun_to_opt)(initial_samples)
    #         if isinstance(initial_yvalues, tuple):
    #             initial_yvalues = initial_yvalues[0]                     # (5 ** ndim, )
    #     # negative: since we want minimum values   
    #     starts = jnp.array(init_centers(-initial_yvalues, initial_samples, nstart*top_k), dtype=dtype) 
     
    starts_top_k = starts.reshape(-1,d*top_k) # (nstart, d*top_k)
    search_domain_top_k = jnp.tile(search_domain,(top_k,1))  # (d*top_k, 2)

    opt_func = multistart_func if method == 'multi-start' else gridsearch_func
    xmin_topk, ymin_topk = opt_func(fun_to_opt, 
                                    starts_top_k, 
                                    search_domain_top_k, 
                                    optimize_method, 
                                    value_and_grad, 
                                    tol,
                                    disp)
     
    return xmin_topk.reshape(-1,d), ymin_topk
    
def multistart_func(
    fun_to_opt: Callable[[jnp.ndarray], jnp.ndarray],
    starts: jnp.ndarray, 
    bounds: jnp.ndarray,  
    method: str='trust-constr', 
    value_and_grad: Union[bool, Callable]=False, 
    tol: float=1e-6,
    disp: bool=False
) -> jnp.ndarray:
    """Function to find the global minimum of the input function via 
    multiple randomly sample initializations 

    :param method: 'L-BFGS-B', 'TNC', 'SLSQP', 'Powell', 'trust-constr'
    :param bounds: np.ndarray (d,2)
    """  
    ff = ScipyBoundedMinimize(
        method=method, 
        fun=fun_to_opt,
        value_and_grad=value_and_grad, 
        tol=tol,
        options={'disp':disp}
    )

    # vv = lambda start: get_opt_result(func=ff, x0=start, bounds=bounds)
    # mv = vmap(vv, 0, 0) 
    # output = mv(starts) 
    cand, cand_vals = tree_map(row_stack_helper, *tree_map(
        lambda x: get_opt_result(
            fun=ff, x0=x, bounds=tuple([bounds[:,0], bounds[:,1]])
        ), list(starts)
    ))  

    bestid = np.argmin(cand_vals)
    xmin = cand[bestid].astype(starts.dtype)
    ymin = cand_vals[bestid]
        
    return xmin, ymin

def gridsearch_func(
    fun_to_opt: Callable[[jnp.ndarray], jnp.ndarray],
    starts: jnp.ndarray,
    bounds: jnp.ndarray,
    method: str='trust-constr',
    value_and_grad: Union[bool, Callable]=False, 
    tol: float=1e-6,
    disp: bool=False
) -> jnp.ndarray:
    """Function to find the global minimum of the input function via 
    multiple randomly sample initializations 

    :param method: 'L-BFGS-B', 'TNC', 'SLSQP', 'Powell', 'trust-constr'
    :param bounds: np.ndarray (d,2)
    """
    t0=time.time()
    print("Grid-Search starting computation of initial points...")
    fvals=[]
    for xx in starts: 
        fval_curr = fun_to_opt(xx)
        fvals.append(fval_curr)
    fvals= jnp.hstack(fvals) 
    idx = lax.argmin(fvals, axis=0, index_dtype=int).reshape(-1)
    t1=time.time() - t0
    print("Grid-Search finished search of initial points ({:.2f}s). ".format(t1))
    
    t0=time.time()
    ff = ScipyBoundedMinimize(
        method=method, 
        fun=fun_to_opt, 
        value_and_grad=value_and_grad,
        tol=tol,
        options={'disp': disp})
     
    xmin, ymin = get_opt_result(
        fun=ff,
        x0=starts[idx].reshape(-1),
        bounds=tuple([bounds[:,0], bounds[:,1]])
    )
    t1=time.time() - t0
    print("Grid-Search finished optimization ({:.2f}s). ".format(t1))

    return xmin, ymin



