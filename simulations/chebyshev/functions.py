import numpy as np
from typing import Callable, List
from dataclasses import replace
import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).parents[2]))

from seemps.state import Strategy, DEFAULT_TOLERANCE
from seemps.cross import Mesh, RegularClosedInterval

from analysis.methods import chebyshev_expand
from analysis.plots import plot_line, plot_surface, plot_contour, PlotParameters

ABS_PATH = str(pathlib.Path(__file__).parent.absolute())
DATA_PATH = ABS_PATH + "/data/"
FIGURES_PATH = ABS_PATH + "/figures/"


def plot_function(
    func: Callable,
    mesh: Mesh,
    orders: List[int],
    name: str,
    tolerance: float = DEFAULT_TOLERANCE,
):
    dim = mesh.dimension
    mps = chebyshev_expand(func, mesh, orders, strategy=Strategy(tolerance=tolerance))
    save_path = FIGURES_PATH + "functions_" + name + ".png"
    x = mesh.to_tensor()
    y_mps = mps.to_vector().reshape(mesh.shape()[:-1])
    y_vec = np.apply_along_axis(lambda x: func(*x), axis=-1, arr=x)
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
    a = -3.0
    b = 3.0
    n = 5
    d = 50

    # # 1. 1D Exponential
    # m = 1
    # name = "exponential_1d"
    # func = lambda x: np.exp(x)
    # mesh = Mesh([RegularClosedInterval(a, b, 2**n) for _ in range(m)])
    # plot_function(func, mesh, [d], name)

    # # 2. 2D Exponential
    # m = 2
    # name = "exponential_2d"
    # func = lambda x, y: np.exp(x + y)
    # mesh = Mesh([RegularClosedInterval(a, b, 2**n) for _ in range(m)])
    # plot_function(func, mesh, [d, d], name)

    # 3. 2D Step
    m = 2
    name = "step_2d"
    func = lambda x, y: np.heaviside((x - 1) + 0.5 * y**2, 0.5)
    mesh = Mesh([RegularClosedInterval(a, b, 2**n) for _ in range(m)])
    plot_function(func, mesh, [d, d], name)
