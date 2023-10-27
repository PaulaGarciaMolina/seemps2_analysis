from typing import Callable
from seemps.cross import Mesh
import numpy as np
import vegas


# TODO: Complete the implementation
def integrate_function(func: Callable, mesh: Mesh, **kwargs):
    domain = [list(pair) for pair in zip(alist, blist)]
    integrator = vegas.Integrator(domain)
    # Precondition the integrator
    integrator(func, **kwargs, rtol=1e-6)
    return integrator(func, **kwargs)
