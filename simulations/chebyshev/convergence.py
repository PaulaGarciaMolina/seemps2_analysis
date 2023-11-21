import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import Callable, List, Dict
from dataclasses import replace
import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).parents[2]))

from seemps.state import Strategy
from seemps.cross import Mesh, RegularClosedInterval

from analysis.utils import time_this
from analysis.methods import chebyshev_expand, integrate_mps
from analysis.functions import distance_norm_1, distance_norm_2, distance_norm_inf, step, absolute
from analysis.loops import param_loop
from analysis.plots import PlotParameters, plot_line, set_mosaic

ABS_PATH = str(pathlib.Path(__file__).parent.absolute())
DATA_PATH = ABS_PATH + "/data/"
FIGURES_PATH = ABS_PATH + "/figures/"


def helper(func: Callable, m: int, a: float, b: float) -> Callable:
    """Helper function for the Chebyshev convergence simulations."""

    def foo(n: int, d: int, t: int) -> Dict:
        # Run the algoritm
        mesh = Mesh([RegularClosedInterval(a, b, 2**n) for _ in range(m)])
        orders = [d for _ in range(m)]
        strategy = Strategy()  # tolerance=t)
        mps, time_expand = time_this(chebyshev_expand)(
            func, mesh, orders, method="clenshaw", strategy=strategy
        )

        # Compute integrals
        integrals = np.array([])
        times = np.array([time_expand])
        for integral_type in ["midpoint", "trapezoidal", "simpson", "fifth_order"]:
            integral, time = time_this(integrate_mps)(mps, mesh, integral_type=integral_type)
            integrals = np.append(integrals, integral)
            times = np.append(times, time)
        times[times < 1e-5] = None

        # Compute distances
        distances = np.array([])
        y_mps = mps.to_vector()
        y_vec = np.apply_along_axis(func, axis=-1, arr=mesh.to_tensor()).flatten()
        norm_1, time_norm_1 = time_this(distance_norm_1)(y_mps, y_vec)
        norm_2, time_norm_2 = time_this(distance_norm_2)(y_mps, y_vec)
        norm_inf, time_norm_inf = time_this(distance_norm_inf)(y_mps, y_vec)
        distances = np.append(distances, [norm_1, norm_2, norm_inf])
        times = np.append(times, [time_norm_1, time_norm_2, time_norm_inf])

        # Save results (should be an object)
        data = {
            "integrals": np.array(integrals, dtype=np.float64),
            "distances": np.array(distances, dtype=np.float64),
            "times": np.array(times),
        }
        return data

    return foo


def main(
    func: Callable,
    dim: int,
    a: float,
    b: float,
    nlist: List[int],
    dlist: List[int],
    tlist: List[float],
    exact_integral: float,
    name: str,
    show: bool = True,
) -> None:
    """Main function of the Chebyshev convergence simulations."""
    params_n = {"n": nlist, "d": [dlist[-1]], "t": [tlist[-1]]}
    params_d = {"n": [nlist[-1]], "d": dlist, "t": [tlist[-1]]}
    results_n = param_loop(helper(func, dim, a, b), params_n, name=name, path=DATA_PATH)
    results_d = param_loop(helper(func, dim, a, b), params_d, name=name, path=DATA_PATH)

    # TODO: Process data, load it into a numpy array and produce plots
    integrals_n = np.array([result["integrals"] for result in results_n.flatten()])
    integral_errors_n = abs(integrals_n - exact_integral)
    distances_n = np.array([result["distances"] for result in results_n.flatten()])
    times_n = np.array([result["times"] for result in results_n.flatten()])

    integrals_d = np.array([result["integrals"] for result in results_d.flatten()])
    integral_errors_d = abs(integrals_d - exact_integral)
    distances_d = np.array([result["distances"] for result in results_d.flatten()])
    times_d = np.array([result["times"] for result in results_d.flatten()])

    # PLOT MOSAIC
    fig, axs = set_mosaic(2, 3, figsize=(15, 10))
    axs[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0, 2].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1, 2].xaxis.set_major_locator(MaxNLocator(integer=True))
    # First Row: Plots as a function of n (integral, norm and runtime).
    parameters_n = PlotParameters(xlabel="n", style="-o", yscale="log")
    parameters_n1 = replace(
        parameters_n,
        ylabel=r"$\frac{I - I_{{exact}}}{I_{{exact}}}$",
        legend_title="Integral type",
        legend_labels=["midpoint", "trapezoidal", "simpson", "fifth_order"],
    )

    plot_line(nlist, integral_errors_n, axs=axs[0, 0], parameters=parameters_n1)
    parameters_n2 = replace(
        parameters_n,
        ylabel=r"$\frac{|f|^p - |f|^p_{{exact}}}{2^n}$",
        title=f"(fixing d = {dlist[-1]}, t = {tlist[-1]})",
        legend_title="Norm type",
        legend_labels=["L1 normalized", "L2 normalized", "Linf"],
    )
    plot_line(nlist, distances_n, axs=axs[0, 1], parameters=parameters_n2)
    parameters_n3 = replace(
        parameters_n,
        ylabel=r"$time(s)$",
        xscale="log",
        legend_title="Computation period",
        legend_labels=[
            "chebyshev expansion",
            "midpoint",
            "trapezoidal",
            "simpson",
            "fifth_order",
            "norm-1",
            "norm-2",
            "norm-inf",
        ],
    )
    plot_line(nlist, times_n[:, 0:5], axs=axs[0, 2], parameters=parameters_n3)

    # Second Row: Plots as a function of d (integral, norm and runtime).
    parameters_d = PlotParameters(xlabel="d", style="-o", yscale="log")
    parameters_d1 = replace(parameters_d, ylabel=r"$\frac{I - I_{{exact}}}{I_{{exact}}}$")
    plot_line(dlist, integral_errors_d, axs=axs[1, 0], parameters=parameters_d1)
    parameters_d2 = replace(
        parameters_d,
        ylabel=r"$\frac{|f|^p - |f|^p_{{exact}}}{2^n}$",
        title=f"(fixing n = {nlist[-1]}, t = {tlist[-1]})",
    )
    plot_line(dlist, distances_d, axs=axs[1, 1], parameters=parameters_d2)
    parameters_d3 = replace(parameters_d, ylabel=r"$time(s)$", xscale="log")
    plot_line(dlist, times_d[:, 0:5], axs=axs[1, 2], parameters=parameters_d3)

    plt.suptitle(f"Chebyshev convergence analysis for {name}")
    plt.tight_layout()
    plt.savefig(FIGURES_PATH + f"mosaic_{name}" + ".png", dpi=400)
    if show:
        plt.show()


if __name__ == "__main__":
    # Common parameters
    m = 1
    a = -1.0
    b = 1.0
    nlist = list(range(2, 21))
    dlist = list(range(1, 21))
    tlist = [1e-16]
    show = True

    # 1. e^x
    func = lambda x: np.exp(x)
    exact_integral = np.exp(b) - np.exp(a)
    name = f"exponential_[{a},{b}]"
    main(func, m, a, b, nlist, dlist, tlist, exact_integral, name=name, show=show)

    # 2. e^(-x)
    func = lambda x: np.exp(-x)
    exact_integral = np.exp(-a) - np.exp(-b)
    name = f"exponential_inv_[{a},{b}]"
    main(func, m, a, b, nlist, dlist, tlist, exact_integral, name=name, show=show)

    # 3. x^2
    func = lambda x: x**2
    exact_integral = (b**3 - a**3) / 3
    name = f"xsquared_[{a},{b}]"
    main(func, m, a, b, nlist, [2, 3], tlist, exact_integral, name=name, show=show)

    # 4. cos(k*x)
    k = 3
    func = lambda x: np.cos(k * x)
    exact_integral = (np.sin(b * k) - np.sin(a * k)) / k
    name = f"cos({k}x)_[{a},{b}]"
    main(func, m, a, b, nlist, dlist, tlist, exact_integral, name=name, show=show)

    # 5. e^(-x**2)
    σ = 0.5
    func = lambda x: np.exp(-(x**2) / (2 * σ))
    exact_integral = np.sqrt(σ * np.pi / 2) * (
        np.math.erf(b / (np.sqrt(2) * np.sqrt(σ))) - np.math.erf(a / (np.sqrt(2) * np.sqrt(σ)))
    )
    nlist_ext = nlist  # + [22, 24, 26]
    name = f"gaussian_[{a},{b}]"
    main(func, m, a, b, nlist_ext, dlist, tlist, exact_integral, name=name, show=show)

    dlist = range(1, 101)

    # 6. step(x) centered
    func = step(0.0)
    exact_integral = b
    name = f"step_[{a},{b}]"
    main(func, m, a, b, nlist, dlist, tlist, exact_integral, name=name, show=show)

    # 6. step(x) displaced
    x_cut = 0.5
    func = step(x_cut)
    exact_integral = b - x_cut
    name = f"step_cut{x_cut}_[{a},{b}]"
    main(func, m, a, b, nlist, dlist, tlist, exact_integral, name=name, show=show)

    # 7. abs(x) centered
    func = absolute(0)
    exact_integral = (b**2 + a**2) / 2
    name = f"abs_[{a},{b}]"
    main(func, m, a, b, nlist, dlist, tlist, exact_integral, name=name, show=show)

    # # 8. abs(x) displaced
    # x_cut = 0.5
    # func = absolute(x_cut)
    # exact_integral = (b**2 + a**2) / 2
    # name = f"abs_cut{x_cut}_[{a},{b}]"
    # main(func, m, a, b, nlist, dlist, tlist, exact_integral, name=name, show=show)
