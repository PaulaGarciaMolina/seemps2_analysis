import numpy as np
from typing import Callable, Dict, List
from dataclasses import replace
import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).parents[2]))

from seemps.cross import cross_interpolation, CrossStrategy, Mesh, RegularClosedInterval

from analysis.methods import integrate_mps, integrate_function
from analysis.functions import gaussian_product
from analysis.loops import param_loop
from analysis.utils import time_this
from analysis.plots import PlotParameters, plot_surface, plot_line, set_mosaic

ABS_PATH = str(pathlib.Path(__file__).parent.absolute())
DATA_PATH = ABS_PATH + "/data/"
FIGURES_PATH = ABS_PATH + "/figures/"


def helper(func: Callable, filter: Callable, m: int, a: float, b: float) -> Callable:
    def foo(n: int) -> Dict:
        # Compute MPS and times for the function and the filter
        mesh = Mesh([RegularClosedInterval(a, b, 2**n) for _ in range(m)])
        func_mps, func_time = time_this(cross_interpolation)(func, mesh)
        filter_mps, filter_time = time_this(cross_interpolation)(filter, mesh)

        # Multiply function and filter
        filtered_mps = func_mps * filter_mps

        # Compute the integral and integration time
        integral_mps, integral_mps_time = time_this(integrate_mps)(
            filtered_mps, mesh, integral_type="simpson"
        )
        time_total = func_time + filter_time + integral_mps_time

        # Return results
        data = {"integral": integral_mps, "time": time_total}
        return data

    return foo


def main(
    func: Callable,
    filter: Callable,
    m: int,
    a: float,
    b: float,
    nlist: List[int],
    name: str,
    show: bool = True,
):
    params = {"n": nlist}
    results = param_loop(helper(func, filter, m, a, b), params, name=name, path=DATA_PATH)
    integrals = np.array([result["integral"] for result in results.flatten()])
    times = np.array([result["time"] for result in results.flatten()])

    # PLOT
    fig, axs = set_mosaic(1, 2)

    parameters = PlotParameters(xlabel="n", marker="-o")
    parameters_1 = replace(parameters, ylabel="Integral")
    plot_line(nlist, integrals, axs=axs[0], parameters=parameters_1)
    parameters_2 = replace(parameters, ylabel="Time(s)")
    plot_line(nlist, times, axs=axs[1], parameters=parameters_2)

    fig.suptitle(f"Cross Interpolation convergence analysis for {name}")
    fig.tight_layout()
    fig.savefig(FIGURES_PATH + f"mosaic_{name}" + ".png", dpi=400)
    if show:
        fig.show()


if __name__ == "__main__":
    # Common parameters
    a = -3.0
    b = 3.0
    show = True

    # 1. 2D product gaussian filtered with Heaviside
    m = 2
    nlist = [10, 12, 14, 16, 18, 20]
    func = gaussian_product(m, [1, 1], [0])
    filter = lambda vec: np.heaviside(0.5 * vec[0] + 0.3 * vec[1], 0.5)
    name = f"product_heaviside_{m}d"
    main(func, filter, m, a, b, nlist, name, show=show)

    # # Compute the integral for the MPS and for MonteCarlo
    # integral_mc, integral_mc_time = integrate_function(func, filter, mesh)
    # # Check the results (optional)
    # x = None
    # shape = None
    # func_tensor = func_mps.to_vector().reshape(shape)
    # filter_tensor = filter_mps.to_vector().reshape(shape)
    # plot_surface(x, func_tensor)
    # plot_surface(x, filter_tensor)

    # # Multiply function and filter
    # filtered_mps = func_mps * filter_mps
    # filtered_tensor = filtered_mps.to_vector().reshape(shape)
    # plot_surface(x, filtered_tensor)
