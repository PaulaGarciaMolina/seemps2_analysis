import numpy as np
from typing import Callable, List


def step(x_cusp: float = 0) -> Callable:
    """Returns a one-dimensional step function centered at x_cusp."""
    return lambda x: np.heaviside(x - x_cusp, 1)


def step_smooth(x_cusp: float = 0, c: float = 0.1) -> Callable:
    """Returns a smooth approximation to the one-dimensional step function
    centered at x_cusp."""
    return lambda x: 0.5 + np.arctan((x - x_cusp) / c) / np.pi


def abs(x_cusp: float = 0) -> Callable:
    """Returns a one-dimensional absolute value function centered at x_cusp."""
    return lambda x: np.abs(x - x_cusp)


def rectifier(x_cusp: float = 0, slope: float = 1) -> Callable:
    """Returns a one-dimensional linear-rectifier function centered at x_cusp
    and with a given slope."""
    return lambda x: slope * x * (x - x_cusp > 0)


def piecewise(funcs: List[Callable], x_cusps: List[float]) -> Callable:
    """Returns a one-dimensional piecewise function by sewing together a given
    collection of functions at a given collection of cusp points."""
    x_cusps.sort()

    def piecewise_function(x):
        func_idx = next((i for i, xc in enumerate(x_cusps) if x < xc), len(funcs) - 1)
        return funcs[func_idx](x)

    return piecewise_function
