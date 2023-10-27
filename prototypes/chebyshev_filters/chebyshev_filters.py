# Add parent directory to path:
import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + "/src/")

import numpy as np
from numpy import pi, sin, cos, exp, sqrt

import matplotlib.pyplot as plt


def linear_map(vec, a1, b1, a2, b2):
    """Maps linearly an interval [a1, b1] into another [a2, b2]."""
    return (b2 - a2) * (vec - a1) / (b1 - a1) + a2


def interval(a, b, n, mesh):
    if mesh == "e":
        interval = np.array([a + i * (b - a) / 2**n for i in range(2**n)])
    elif mesh == "c":
        b += (b - a) / (2**n - 1)
        interval = np.array([a + i * (b - a) / 2**n for i in range(2**n)])
    elif mesh == "z":
        d = 2**n
        cnodes = np.array(
            [np.cos(np.pi * (2 * k - 1) / (2 * d)) for k in range(1, d + 1)]
        )
        interval = np.flip(linear_map(cnodes, -1, 1, a, b))
    return interval


def chebyshev_zeros(d):
    return np.array([cos(pi * (2 * k - 1) / (2 * d)) for k in range(1, d + 1)])


def chebyshev_eval(d):
    return lambda i: np.array(
        [cos(i * (2 * k - 1) * pi / (2 * d)) for k in range(1, d + 1)]
    )


def chebyshev_coefficients(f, d):
    ccoefs = np.zeros(d)
    extrema = chebyshev_zeros(d)

    cheb_eval = chebyshev_eval(d)
    f_eval = f(extrema)

    ccoefs[0] = np.sum(f_eval) / d
    for i in range(1, d):
        ccoefs[i] = np.sum(f_eval * cheb_eval(i)) * (2 / d)

    return ccoefs


def chebyshev_vector(g, a, b, n, d, mesh="e"):
    x = interval(a, b, n, mesh=mesh)
    x = g(x)

    Tj = x
    Tk = np.ones(2**n)

    if d == 0:
        return Tk, None
    elif d == 1:
        return Tj, Tk

    xop = np.diag(x)
    recurrence = lambda Tj, Tk: 2 * (xop.dot(Tj)) - Tk

    Tj, Tk = chebyshev_vector(g, a, b, n, d - 1)
    Ti = recurrence(Tj, Tk)

    return Ti, Tj


def chebyshev_expansion(f, g, a, b, n, d, mesh="e"):
    ccoefs = chebyshev_coefficients(f, d)
    fvector = np.zeros(2**n)

    for d in range(len(ccoefs)):
        Td, _ = chebyshev_vector(g, a, b, n, d, mesh=mesh)
        fvector += ccoefs[d] * Td

    return fvector


def rectifier(x_cusp: float = 0, slope: float = 1):
    """Returns a one-dimensional linear-rectifier function centered at x_cusp
    and with a given slope."""
    return lambda x: slope * x * (x - x_cusp > 0)


def step(x_cusp: float = 0):
    """Returns a one-dimensional step function centered at x_cusp."""
    return lambda x: np.heaviside(x - x_cusp, 1)


def abs(x_cusp: float = 0):
    """Returns a one-dimensional absolute value function centered at x_cusp."""
    return lambda x: np.abs(x - x_cusp)


a = -0.5
b = 1.00
n = 5
d = 50
mesh = "e"

# f = lambda x: x * (x > 0)
filter = rectifier(0)
func = lambda x: np.exp(-(x**2))
fname = f"Composición de ReLU con sin(x) para d = {d}"

x = interval(a, b, n, mesh=mesh)
fvector = chebyshev_expansion(
    filter, func, a, b, n, d, mesh=mesh
)  # Efectivamente, es tan facil como hacer x -> f(x)

plt.plot(x, func(x), label="func(x)")
plt.plot(x, filter(x), label="filter(x)")
plt.plot(x, filter(func(x)), label="filter(func(x))")
plt.plot(x, fvector, "o", label="Chebyshev filter(func(x))")
plt.legend()
plt.show()

# plt.plot(x, func(x), label="func(x)")
# plt.plot(x, filter(x), label="filter(x)")
# plt.plot(x, func(x) * filter(x), label="func(x) * filter(x)")
# # plt.plot(x, func(filter(x)), label="func(filter(x))")

# # plt.plot(x, func(filter(x)), color="k", zorder=100, label="Exact")
# plt.legend()
# plt.title(fname)
# plt.show()
