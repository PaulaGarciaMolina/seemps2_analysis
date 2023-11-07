import numpy as np
import matplotlib.pyplot as plt
import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).parents[2]))

from analysis.methods import chebyshev_compose, mpo_position, join_list, mps_identity
from analysis.plots import plot_surface


if __name__ == "__main__":
    a = -1.0
    b = 1.0
    n = 5
    mpo_x = mpo_position(a, b, n)
    mpo_y = mpo_position(a, b, n)
    mpo_xy = join_list([mpo_x, mpo_y])

    func = lambda x: x
    mpo = chebyshev_compose(func, mpo_xy, 10, -1, 1)
    mps = mpo.apply(mps_identity(2 * n)).toMPS()
    tensor = mps.to_vector().reshape([2**n, 2**n])
    x = np.linspace(a, b, 2**n)
    X, Y = np.meshgrid(x, x)
    plot_surface(X, Y, tensor)  # x * y
