import numpy as np
import matplotlib.pyplot as plt
import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).parents[2]))

from seemps.cross import Mesh, RegularHalfOpenInterval
from analysis.methods.chebyshev_vector import _coef_tensor


def clenshaw_1d(coef_vector: np.ndarray, x: np.ndarray) -> np.ndarray:
    c = np.flip(coef_vector)
    y = [0] * (len(c) + 2)
    for i in range(2, len(y)):
        y[i] = c[i - 2] - y[i - 2] + 2 * x * y[i - 1]
    return y[-1] - x * y[-2]


sites = 5
order = 10
func = lambda x: np.exp(x)
interval = RegularHalfOpenInterval(-1, 1, 2**sites)
x = interval.to_vector()
mesh = Mesh([interval])
coef_vector = _coef_tensor(func, mesh, [order])
y = clenshaw_1d(coef_vector, x)

plt.plot(x, func(x), label="Exact")
plt.plot(x, y, "o", label="Clenshaw")
plt.legend()
plt.show()
