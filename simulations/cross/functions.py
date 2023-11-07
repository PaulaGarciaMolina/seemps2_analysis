import numpy as np
from typing import Callable, List
from dataclasses import replace
import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).parents[2]))

from seemps.state import Strategy, DEFAULT_TOLERANCE
from seemps.cross import Mesh, RegularClosedInterval, cross_interpolation

from analysis.functions import gaussian_product
from analysis.plots import plot_line, plot_surface, plot_contour, PlotParameters

ABS_PATH = str(pathlib.Path(__file__).parent.absolute())
DATA_PATH = ABS_PATH + "/data/"
FIGURES_PATH = ABS_PATH + "/figures/"


def plot_function(func: Callable, mesh: Mesh, name: str):
    dim = mesh.dimension
    mps = cross_interpolation(func, mesh)
    save_path = FIGURES_PATH + "functions_" + name + ".png"
    x = mesh.to_tensor()
    y_mps = mps.to_vector().reshape(x.shape[:-1])
    y_vec = np.apply_along_axis(func, axis=-1, arr=x)
    parameters = PlotParameters(
        title="Chebyshev expansion",
        save_path=save_path,
    )

    if dim == 1:
        params = replace(
            parameters, style=["-", "o"], legend_title="Type", legend_labels=["Vector", "MPS"]
        )
        plot_line(x, [y_vec, y_mps], parameters=params)
    elif dim == 2:
        X = x[:, :, 0]
        Y = x[:, :, 1]
        plot_surface(X, Y, y_mps, parameters=parameters)
        plot_contour(X, Y, y_mps, parameters=parameters)


if __name__ == "__main__":
    # Common parameters
    a = -5.0
    b = 5.0
    n = 5

    # # 1. 1D Exponential
    # m = 1
    # name = "exponential_1d"
    # func = lambda vec: np.exp(vec[0])
    # mesh = Mesh([RegularClosedInterval(a, b, 2**n) for _ in range(m)])
    # plot_function(func, mesh, name)

    # # 2. 2D Exponential
    # m = 2
    # name = "exponential_2d"
    # func = lambda vec: np.exp(vec[0] + vec[1])
    # mesh = Mesh([RegularClosedInterval(a, b, 2**n) for _ in range(m)])
    # plot_function(func, mesh, name)

    # # 3. 2D Step
    # m = 2
    # name = "step_2d"
    # func = lambda vec: np.heaviside((vec[0] - 1) + 0.5 * vec[1] ** 2, 0.5)
    # mesh = Mesh([RegularClosedInterval(a, b, 2**n) for _ in range(m)])
    # plot_function(func, mesh, name)

    # 4. Filtered 2D gaussian with step
    m = 2
    name = "filtered_gaussian_2d"
    func = gaussian_product(m, [1, 1], [0])
    filter = lambda vec: np.heaviside((vec[0] - 1) + (vec[1] - 1), 0.5)
    filtered_func = lambda vec: func(vec) * filter(vec)
    mesh = Mesh([RegularClosedInterval(a, b, 2**n) for _ in range(m)])
    plot_function(filtered_func, mesh, name)
