import numpy as np
import h5py
import pathlib
from copy import deepcopy
from typing import Callable, List, Union, Optional

from seemps.typing import Sequence
from seemps.hdf5 import read_mps, write_mps, read_mpo, write_mpo
from seemps.cross import Mesh, Interval, ChebyshevZerosInterval
from seemps.state import MPS, Strategy, DEFAULT_TOLERANCE
from seemps.mpo import MPO, MPOSum

from .factories import mps_empty, mps_identity, mps_position
from .factories import mpo_empty, mpo_identity

DATA_PATH = str(pathlib.Path(__file__).parent.absolute()) + "/data/"


def _zeros_evaluated(order: int, i: int) -> np.ndarray:
    return np.array(
        [np.cos(np.pi * i * (2 * k - 1) / (2 * order)) for k in range(1, order + 1)]
    )


def _zeros(order: int) -> np.ndarray:
    return _zeros_evaluated(order, i=1)


def _zeros_matrix(order: int) -> np.ndarray:
    zeros_matrix = _zeros_evaluated(order, np.arange(order))
    zeros_matrix[:, 0] *= 0.5
    return zeros_matrix


def mps_chebyshev(
    mps_0: MPS,
    order: int,
    name: Optional[str] = None,
    strategy: Strategy = Strategy(tolerance=DEFAULT_TOLERANCE),
) -> MPS:
    path = DATA_PATH + name + ".hdf5" if name is not None else None
    sites = len(mps_0)
    try:
        with h5py.File(path, "r") as file:
            Ti = read_mps(file, f"order_{order}")
            Tj = (
                read_mps(file, f"order_{order-1}") if order > 0 else mps_identity(sites)
            )
    except:
        if order == 0:
            Ti = mps_identity(sites)
            Tj = mps_identity(sites)
        elif order == 1:
            Ti = mps_0
            Tj = mps_identity(sites)
        else:
            Tj = mps_chebyshev(mps_0, order - 1, name)
            Tk = mps_chebyshev(mps_0, order - 2, name)
            Ti = (2.0 * mps_0 * Tj - Tk).toMPS(strategy=strategy)
        if name is not None:
            with h5py.File(path, "a") as file:
                write_mps(file, f"order_{order}", Ti)
    return Ti


def mpo_chebyshev(mpo_0: Union[MPO, MPOSum], order: int, name: str) -> MPOSum:
    path = DATA_PATH + name + ".hdf5"
    try:
        with h5py.File(path, "r") as file:
            Ti = read_mpo(file, f"order_{order}")
            Tj = read_mpo(file, f"order_{order-1}")
    except:
        sites = len(mpo_0)
        if order == 0:
            Ti = mpo_identity(sites)
            Tj = mpo_identity(sites)
        elif order == 1:
            Ti = mpo_0
            Tj = mpo_identity(sites)
        else:
            Tj = mpo_chebyshev(mpo_0, order - 1, name)
            Tk = mpo_chebyshev(mpo_0, order - 2, name)
            Ti = 2.0 * mpo_0 * Tj - Tk
        with h5py.File(path, "a") as file:
            write_mpo(file, f"order_{order}", Ti)
    return Ti


def _func_tensor(func: Callable, mesh: Mesh, orders: List[int]) -> np.ndarray:
    intervals = [
        ChebyshevZerosInterval(interval.start, interval.stop, orders[idx])
        for idx, interval in enumerate(mesh.intervals)
    ]
    cheb_tensor = Mesh(intervals).to_tensor()
    return np.apply_along_axis(func, -1, cheb_tensor)


def _coef_tensor(func: Callable, mesh: Mesh, orders: List[int]) -> np.ndarray:
    prefactor = (2 ** len(orders)) / (np.prod(orders))
    coef_tensor = prefactor * np.flip(_func_tensor(func, mesh, orders))
    for idx, order in enumerate(orders):
        matrix = _zeros_matrix(order)
        coef_tensor = np.swapaxes(coef_tensor, 0, idx)
        coef_tensor = np.einsum("i..., ik... -> k...", coef_tensor, matrix)
        coef_tensor = np.swapaxes(coef_tensor, idx, 0)
    return coef_tensor


def _differentiate_coef_tensor(coef_tensor: np.ndarray, m: int) -> np.ndarray:
    pass


def _integrate_coef_tensor(
    coef_tensor: np.ndarray, m: int, c_int: Optional[float] = None
) -> np.ndarray:
    pass


def _join(mps_list: List[MPS]) -> MPS:
    """Returns a MPS that is given by the union of a list of MPS by their extremes."""
    nested_sites = [mps._data for mps in mps_list]
    flattened_sites = [site for sites in nested_sites for site in sites]
    return MPS(flattened_sites)


def chebyshev_expand(
    func: Callable,
    mesh: Mesh,
    orders: List[int],
    strategy: Strategy = Strategy(tolerance=DEFAULT_TOLERANCE),
) -> MPS:
    """Encodes a multivariate function in a MPS by means of a truncated Chebyshev expansion."""
    coef_tensor = _coef_tensor(func, mesh, orders)
    mps = mps_empty(int(np.log2(np.prod(mesh.shape()[:-1]))))
    for each_order in np.ndindex(coef_tensor.shape):
        each_mps = []
        for idx, order in enumerate(each_order):
            interval = deepcopy(mesh.intervals[idx])
            interval.start = -1.0
            interval.stop = 1.0
            name = f"mps_chebyshev-type_{interval.type}-sites_{int(np.log2(interval.size))}"
            each_mps.append(mps_chebyshev(mps_position(interval), order, name))
        mps += coef_tensor[each_order] * _join(each_mps)
        mps = mps.toMPS(strategy=strategy)
    return mps


def chebyshev_compose(
    func: Callable,
    tensor_network_0: Union[MPS, MPO, MPOSum],
    order: int,
    start: float,
    stop: float,
    strategy: Strategy = Strategy(tolerance=DEFAULT_TOLERANCE),
) -> Union[MPS, MPOSum]:
    mesh = Mesh([ChebyshevZerosInterval(start, stop, 0)])
    coef_vector = _coef_tensor(func, mesh, [order]).flatten()
    if isinstance(tensor_network_0, MPS):
        tensor_network = mps_empty(len(tensor_network_0))
        for idx, coef in enumerate(coef_vector):
            name = None
            tensor_network += coef * mps_chebyshev(tensor_network_0, idx, name)
            tensor_network = tensor_network.toMPS(strategy=strategy)
    elif isinstance(tensor_network_0, Union[MPO, MPOSum]):
        tensor_network = mpo_empty(len(tensor_network_0))
        for idx, coef in enumerate(coef_vector):
            name = "name"
            tensor_network += coef * mps_chebyshev(tensor_network_0, idx, name)
    else:
        raise ValueError("Invalid tensor network")
    return tensor_network


# def chebyshev_compose_mps(
#     mps_0: MPS,
#     func: Callable,
#     start: float,
#     stop: float,
#     order: int,
#     strategy: Strategy = Strategy(tolerance=DEFAULT_TOLERANCE),
# ) -> MPS:
#     """Filters a given MPS with a univariate function by means of a truncated Chebyshev expansion."""
#     mesh = Mesh[Interval(start, stop, 0)]
#     coef_vector = _coef_tensor(func, mesh, [order])
#     mps = mps_empty(len(mps_0))
#     for idx, coef in enumerate(coef_vector):
#         name = "implement_me"
#         mps += coef * mps_chebyshev(mps_0, idx, name)
#         mps = mps.toMPS(strategy=strategy)
#     return mps


# def chebyshev_compose_mpo(
#     mpo_0: MPS, func: Callable, start: float, stop: float, order: int
# ) -> MPS:
#     """Filters a given MPO with a univariate function by means of a truncated Chebyshev expansion."""
#     mesh = Mesh[Interval(start, stop, 0)]
#     coef_vector = _coef_tensor(func, mesh, [order])
#     mpo = mpo_empty(len(mpo_0))
#     for idx, coef in enumerate(coef_vector):
#         name = "implement_me"
#         mpo += coef * mpo_chebyshev(mpo_0, idx, name)
#     return mpo
