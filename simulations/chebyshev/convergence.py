import numpy as np
from typing import Callable, List, Dict
import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).parents[2]))

from seemps.state import Strategy
from seemps.cross import (
    Mesh,
    RegularClosedInterval,
    ChebyshevZerosInterval,
)

from analysis.methods import chebyshev_expansion, integrate_mps
from analysis.functions import distance_norm_1, distance_norm_2, distance_norm_inf
from analysis.loops import param_loop_pickle
from analysis.plots import plot_mosaic_3


DATA_PATH = str(pathlib.Path(__file__).parent.absolute()) + "/data/"


def helper_chebyshev_convergence(
    func: Callable, m: int, a: float, b: float
) -> Callable:
    """Returns a helper function for a given function on a square mesh given by the
    cartesian product of m intervals [a, b]."""

    def helper(n: int, d: int, t: int) -> Dict:
        """Computes the integrals and norms of a multivariate Chebyshev expansion with
        orders d, MPS truncations t, and on a square mesh with n qubits."""
        mesh = Mesh([RegularClosedInterval(a, b, 2**n) for _ in range(m)])
        dlist = [d for _ in m]
        strategy = Strategy(tolerance=t)
        mps = chebyshev_expansion(func, mesh, dlist, strategy=strategy)
        mesh_fejer = Mesh([ChebyshevZerosInterval(a, b, 2**n) for _ in range(m)])
        mps_fejer = chebyshev_expansion(func, mesh_fejer, dlist, strategy=strategy)

        integral_types = ["midpoint", "trapezoidal", "simpson", "fifth"]
        integrals = []
        for integral_type in integral_types[-1]:
            integrals.append(integrate_mps(mps, integral_type=integral_type))
        integrals.append(integrate_mps(mps_fejer, integral_type="fejer"))

        y_mps = mps.to_vector()
        y_vec = func(mesh.to_tensor).flatten()
        norms = []
        norms.append(distance_norm_1(y_mps, y_vec))
        norms.append(distance_norm_2(y_mps, y_vec))
        norms.append(distance_norm_inf(y_mps, y_vec))

        data = {
            "integrals": np.array(integrals),
            "norms": np.array(norms),
            "time": None,
        }

        return data

    return helper


def chebyshev_convergence(
    func: Callable,
    dim: int,
    a: float,
    b: float,
    nlist: List[int],
    dlist: List[int],
    tlist: List[float],
    exact_integral: float,
) -> None:
    """Main function of the Chebyshev convergence simulations."""
    params = {"n": nlist, "d": dlist[-1], "t": tlist[-1]}
    convergence_with_qubits = param_loop_pickle(
        helper_chebyshev_convergence(func, dim, a, b), params, path=DATA_PATH
    )

    # TODO: Process data, load it into a numpy array and produce plots
    for dct in convergence_with_qubits:
        integrals = dct["integrals"]
        norms = dct["norms"]
