import numpy as np
from typing import Callable, List


def step(x_cusp: float = 0, a: float = 1, b: float = 0) -> Callable:
    """Returns a one-dimensional step function centered at x_cusp."""
    return lambda x: a * np.heaviside(x - x_cusp, a / 2) + b


def step_smooth(x_cusp: float = 0, c: float = 0.1) -> Callable:
    """Returns a smooth approximation to the one-dimensional step function
    centered at x_cusp."""
    return lambda x: 0.5 + np.arctan((x - x_cusp) / c) / np.pi


def absolute(x_cusp: float = 0, a: float = 1, b: float = 0) -> Callable:
    """Returns a one-dimensional absolute value function centered at x_cusp."""
    return lambda x: a * np.abs(x - x_cusp) + b


def rectifier(
    x_cusp: float = 0,
    a: float = 1,
    b: float = 0,
    cutoff: str = "top",
    adjust: bool = False,
) -> Callable:
    """Returns a one-dimensional linear-rectifier function centered at x_cusp
    and with a given slope."""
    if cutoff == "top":
        condition = lambda x: x > x_cusp
    elif cutoff == "bottom":
        condition = lambda x: x < x_cusp

    if adjust:
        return lambda x: a * (x - x_cusp) * condition(x) + b
    else:
        return lambda x: a * x * condition(x) + b


def piecewise(funcs: List[Callable], x_cusps: List[float]) -> Callable:
    """Returns a one-dimensional piecewise function by sewing together a given
    collection of functions at a given collection of cusp points."""
    x_cusps.sort()

    def piecewise_function(x):
        func_idx = next((i for i, xc in enumerate(x_cusps) if x < xc), len(funcs) - 1)
        return funcs[func_idx](x)

    return piecewise_function
