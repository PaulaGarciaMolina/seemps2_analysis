import numpy as np
from scipy.optimize import curve_fit


def fit_exponential(x, y):
    """Fits an exponential of the form $a * \exp{b * x} + c$ to a vector y(x)
    with respect to its domain x and returns the parameters (a, b, c)."""

    def exponential_func(x, a, b, c):
        return a * np.exp(b * x) + c

    params, _ = curve_fit(exponential_func, x, y)
    return params
