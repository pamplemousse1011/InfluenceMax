import time 

import numpy as np
import jax.numpy as jnp
from jaxopt import ScipyBoundedMinimize

def rosen(x):
    """The Rosenbrock function"""
    return jnp.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = jnp.zeros_like(x) 
    der = der.at[1:-1].set(200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm))
    der = der.at[0].set(-400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0]))
    der= der.at[-1].set(200*(x[-1]-x[-2]**2))
    return der
    
def rosen_value_and_grad(x):
    func_value = rosen(x)
    grad_value = rosen_der(x)
    return func_value, grad_value

d=4 
x0=jnp.array(np.random.normal(0,1,(d,)))
search_domain = jnp.array(np.stack([np.random.uniform(0,1,(d,)), 
                                    np.random.normal(2,4,(d,))], 
                                    axis = -1))

t0=time.time()
ff = ScipyBoundedMinimize(method='trust-constr', fun=rosen,
                          value_and_grad=rosen_value_and_grad)
x0 = x0
output = ff.run(x0, 
                bounds = tuple([search_domain[:,0], search_domain[:,1]]))
t1=time.time()-t0
print(t1)

from jaxopt._src import base
ff.fun, ff._grad_fun, ff._value_and_grad_fun = (
    base._make_funs_without_aux(ff.fun, ff.value_and_grad, ff.has_aux)
)
print("wether value_and_grad prints the same results as fun and grad_fun in ScipyBoundedMinimize:")
print(f"For x0={x0}, ff.fun(x0)={ff.fun(x0)}, ff._grad_fun(x0)={ff._grad_fun(x0)}")
print(f"ff._value_and_grad_fun(x0)={ff._value_and_grad_fun(x0)}")
print(f"The input function vlaue_and_grad returns {rosen_value_and_grad(x0)}")