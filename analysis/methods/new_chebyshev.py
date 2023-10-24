import numpy as np
import h5py
import pathlib
from typing import Callable, List, Optional

from seemps.hdf5 import read_mps, write_mps
from seemps.state import MPS, Strategy, DEFAULT_TOLERANCE
from seemps.cross import Mesh, ChebyshevZerosInterval

from .factories import mps_empty, mps_identity, mps_position

DATA_PATH = str(pathlib.Path(__file__).parent.absolute()) + "/data/"


def zeros(order: int) -> np.ndarray:
    """ """
    return zeros_evaluated(order, i=1)


def zeros_evaluated(order: int, i: int) -> np.ndarray:
    """ """
    return np.array(
        [np.cos(np.pi * i * (2 * k - 1) / (2 * order)) for k in range(1, order + 1)]
    )


def zeros_matrix(order: int) -> np.ndarray:
    """ """
    zeros_matrix = zeros_evaluated(order, np.arange(order))
    zeros_matrix[:, 0] *= 0.5
    return zeros_matrix


def chebyshev_mps(
    order: int,
    sites: int,
    mesh_type: str = "o",
    mps0: Optional[MPS] = None,
) -> MPS:
    """ """
    if mesh_type not in ["o", "c", "z"]:
        raise ValueError("Invalid mesh_type")
    name_Ti = f"order_{order}-sites_{sites}-type_{mesh_type}"
    name_Tj = f"order_{order-1}-sites_{sites}-type_{mesh_type}"
    path = DATA_PATH + "chebyshev_mps.hdf5"
    try:
        with h5py.File(path, "r") as file:
            Ti = read_mps(file, name_Ti)
            Tj = read_mps(file, name_Tj)
    except:
        if mps0 is None:
            mps0 = mps_position(-1.0, 1.0, sites, mesh_type)
        if order == 0:
            Ti = mps_identity(sites)
            Tj = mps_identity(sites)
        elif order == 1:
            Ti = mps0
            Tj = mps_identity(sites)
        else:
            Tj = chebyshev_mps(order - 1, sites, mesh_type, mps0)
            Tk = chebyshev_mps(order - 2, sites, mesh_type, mps0)
            Ti = (2.0 * mps0 * Tj - Tk).toMPS(
                strategy=Strategy(tolerance=DEFAULT_TOLERANCE)
            )
        with h5py.File(path, "a") as file:
            write_mps(file, name_Ti, Ti)
    return Ti


def func_tensor(func: Callable, mesh: Mesh, orders: List[int]) -> np.ndarray:
    """ """
    intervals = [
        ChebyshevZerosInterval(interval.start, interval.stop, orders)
        for interval in mesh.intervals
    ]
    cheb_tensor = Mesh(intervals).to_tensor()
    return np.apply_along_axis(func, -1, cheb_tensor)


def coef_tensor(func: Callable, mesh: Mesh, orders: List[int]) -> np.ndarray:
    """ """
    prefactor = (2 ** len(orders)) / (np.prod(orders))
    coef_tensor = prefactor * np.flip(func_tensor(func, mesh, orders))
    for idx, order in enumerate(orders):
        matrix = zeros_matrix(order)
        coef_tensor = np.swapaxes(coef_tensor, 0, idx)
        coef_tensor = np.einsum("i..., ik... -> k...", coef_tensor, matrix)
        coef_tensor = np.swapaxes(coef_tensor, idx, 0)
    return coef_tensor


def differentiate_coef_tensor(coef_tensor: np.ndarray, m: int) -> np.ndarray:
    """ """

    def take_coef(i):
        return np.take(coef_tensor, i, axis=m)

    def put_at(inds, axis=-1):
        if isinstance(inds, (int, slice)):
            inds = (inds,)
        index = [slice(None)] * abs(axis) + list(inds) + [slice(None)] * (axis - 1)
        return tuple(index)

    shape = coef_tensor.shape
    d = shape[m]
    new_tensor = np.zeros(shape[:m] + (d + 1,) + shape[m + 1 :])
    for i in range(d - 2, -1, -1):
        new_tensor[put_at(i, axis=m)] = 2 * (i + 1) * take_coef(i + 1) + np.take(
            new_tensor, i + 2, axis=m
        )
    new_tensor = np.take(new_tensor, range(d - 1), axis=m)
    return new_tensor


def integrate_coef_tensor(
    coef_tensor: np.ndarray, m: int, c_int: Optional[float] = None
) -> np.ndarray:
    """ """

    def take_coef(i):
        return np.take(coef_tensor, i, axis=m)

    def put_at(inds, axis=-1):
        if isinstance(inds, (int, slice)):
            inds = (inds,)
        index = [slice(None)] * abs(axis) + list(inds) + [slice(None)] * (axis - 1)
        return tuple(index)

    shape = coef_tensor.shape
    d = shape[m]
    new_tensor = np.zeros(shape[:m] + (d - 2,) + shape[m + 1 :])
    new_tensor[put_at(1, axis=m)] = (2 * take_coef(0) - take_coef(2)) / 2
    for i in range(2, d - 2):
        new_tensor[put_at(i, axis=m)] = (take_coef(i - 1) - take_coef(i + 1)) / (2 * i)
    if c_int is None:
        c_int = take_coef(0)  # Equivalent to c=0
    new_tensor[put_at(0, axis=m)] = c_int
    return new_tensor


def chebyshev_mps_expansion(
    func: Callable,
    mesh: Mesh,
    orders: List[int],
    strategy: Strategy = Strategy(tolerance=DEFAULT_TOLERANCE),
) -> MPS:
    """ """
    coef_tensor = coef_tensor(func, mesh, orders)
    qubits = [int(np.log2(s)) for s in mesh.shape()[:-1]]
    mps = mps_empty(np.sum(qubits))
    for order_combination in np.ndindex(coef_tensor.shape):
        mps_combination = []
        for idx, order in enumerate(order_combination):
            mps_combination.append(chebyshev_mps(order, qubits[idx]))
        mps_term = cartesian_product(mps_combination)
        mps += coef_tensor[order_combination] * mps_term
        mps = mps.toMPS(strategy=strategy)
    return mps


def chebyshev_mpo_expansion():
    pass


def chebyshev_mps_filter_1d(
    func: Callable,
    mps: MPS,
    order: int,
    start: float,
    stop: float,
    strategy: Strategy = Strategy(tolerance=DEFAULT_TOLERANCE),
) -> MPS:
    """ """
    coef_tensor = coef_tensor(func, mesh, order)
    for order in np.ndindex(coef_tensor.shape):
        pass


def chebyshev_mpo_filter_1d():
    pass
