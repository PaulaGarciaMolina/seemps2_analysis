import numpy as np
import h5py
import pathlib
from copy import deepcopy
from typing import Callable, List, Optional
from scipy.linalg import eigvals

from seemps.hdf5 import read_mps, write_mps
from seemps.cross import Mesh, ChebyshevZerosInterval
from seemps.state import MPS, Strategy
from seemps.truncate import simplify

from .factories_mps import mps_empty, mps_identity, mps_position, join_list

DATA_PATH = str(pathlib.Path(__file__).parent.absolute()) + "/data/"


def _zeros(order: int) -> np.ndarray:
    """
    Returns the zeros of a Chebyshev polynomial of a given order.

    Parameters:
        order (int): The order of the Chebyshev polynomial.
    """
    return _zeros_evaluated(order, i=1)


def _zeros_evaluated(order: int, i: int) -> np.ndarray:
    """
    Returns the evaluation of a Chebyshev polynomial of order 'i' on the zeros
    of a Chebyshev polynomial of a given order.

    Parameters:
        order (int): The order of the the Chebyshev polynomial.
        i (int): The order of the Chebyshev polynomial to evaluate.
    """
    return np.array([np.cos(np.pi * i * (2 * k - 1) / (2 * order)) for k in range(1, order + 1)])


def _zeros_matrix(order: int) -> np.ndarray:
    """
    Compute a matrix of zeros for Chebyshev polynomials up to the given order.

    Parameters:
        order (int): The order of Chebyshev polynomials.
    """
    zeros_matrix = _zeros_evaluated(order, np.arange(order))
    zeros_matrix[:, 0] *= 0.5
    return zeros_matrix


def mps_chebyshev(
    mps_0: MPS,
    order: int,
    name: Optional[str] = None,
    strategy: Strategy = Strategy(),
) -> MPS:
    """
    Returns a Chebyshev polynomial of order d of a MPS.

    Parameters:
        mps_0 (MPS): The MPS.
        order (int): The order of the Chebyshev polynomial.
        name (str, optional): The name of the dataset to store results in an HDF5 file.
                              Default is None (no storage).
        strategy (Strategy): An optional strategy for the SeeMPS methods on the MPS.

    Returns:
        MPS: The MPS corresponding to the Chebyshev polynomial of the starting MPS mps_0.

    Example:
        mps_initial = ...  # Create an initial MPS
        order = 5  # Specify the order of the Chebyshev polynomial
        result_mps = mps_chebyshev(mps_initial, order)
        # 'result_mps' will contain the MPS after applying the Chebyshev polynomial of order 5.
    """
    path = DATA_PATH + name + ".hdf5" if name is not None else None
    sites = len(mps_0)
    try:
        with h5py.File(path, "r") as file:
            Ti = read_mps(file, f"order_{order}")
            Tj = read_mps(file, f"order_{order-1}") if order > 0 else mps_identity(sites)
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
            Ti = (2.0 * mps_0 * Tj - Tk).join(strategy=strategy)
        if name is not None:
            with h5py.File(path, "a") as file:
                write_mps(file, f"order_{order}", Ti)
    return Ti


def _func_tensor(func: Callable, mesh: Mesh, orders: List[int]) -> np.ndarray:
    """
    Returns the evaluation a given function over a Chebyshev mesh.

    Parameters:
        func (Callable): The function to be computed over the Chebyshev mesh.
        mesh (Mesh): The Chebyshev mesh over which the function is computed.
        orders (List[int]): A list of orders corresponding to each interval in the mesh.
    """
    intervals = [
        ChebyshevZerosInterval(interval.start, interval.stop, orders[idx])
        for idx, interval in enumerate(mesh.intervals)
    ]
    cheb_tensor = Mesh(intervals).to_tensor()
    return np.apply_along_axis(lambda x: func(*x), -1, cheb_tensor)


def _coef_tensor(func: Callable, mesh: Mesh, orders: List[int]) -> np.ndarray:
    """
    Compute a tensor of coefficients for the Chebyshev polynomial expansion of a function.

    Parameters:
        func (Callable): The function to be expanded.
        mesh (Mesh): The Chebyshev mesh over which the function is expanded.
        orders (List[int]): A list of orders corresponding to each interval in the mesh.

    Returns:
        np.ndarray: A tensor of computed coefficients for the Chebyshev polynomial expansion.

    Example:
        mesh = ...  # Create a Chebyshev mesh with intervals
        orders = [3, 4, 2]  # Specify Chebyshev orders for each interval
        result_tensor = _coef_tensor(my_function, mesh, orders)
        # 'result_tensor' will contain the coefficients for the Chebyshev polynomial expansion
        # of 'my_function' over the specified Chebyshev mesh.
    """
    prefactor = (2 ** len(orders)) / (np.prod(orders))
    coef_tensor = prefactor * np.flip(_func_tensor(func, mesh, orders))
    for idx, order in enumerate(orders):
        matrix = _zeros_matrix(order)
        coef_tensor = np.swapaxes(coef_tensor, 0, idx)
        coef_tensor = np.einsum("i..., ik... -> k...", coef_tensor, matrix)
        coef_tensor = np.swapaxes(coef_tensor, idx, 0)
    return coef_tensor


def chebyshev_expand(func, mesh, orders, method="clenshaw", strategy=Strategy()):
    """
    Encode a multivariate function in a Matrix Product State (MPS) using a truncated Chebyshev expansion.

    Parameters:
        func (callable): The function to be approximated and represented as an MPS.
        mesh (Mesh): The mesh on which the function is sampled.
        orders (List[int]): The list of Chebyshev polynomial orders to be used for the expansion.
        method (str, optional): The method for performing the Chebyshev expansion ("sum", "clenshaw", or "factor").
        strategy (Strategy, optional): The strategy used for MPS construction (default: Strategy()).

    Returns:
        MPS: The MPS representation of the function obtained through Chebyshev expansion.
    """
    if method == "sum":
        return chebyshev_partial_sum(func, mesh, orders, strategy)
    elif method == "clenshaw":
        return chebyshev_clenshaw(func, mesh, orders, strategy)
    elif method == "factor":
        return chebyshev_factor(func, mesh, orders, strategy)


def chebyshev_partial_sum(
    func: Callable, mesh: Mesh, orders: List[int], strategy: Strategy = Strategy()
) -> MPS:
    """
    Encode a multivariate function in a Matrix Product State (MPS) by evaluating its Chebyshev partial sum.

    Parameters:
        func (Callable): The multivariate function to be encoded.
        mesh (Mesh): The Chebyshev mesh over which the function is approximated.
        orders (List[int]): A list of Chebyshev orders for each interval in the mesh.
        strategy (Strategy): An optional strategy for the SeeMPS methods on the MPS.

    Returns:
        MPS: An MPS encoding the function through the Chebyshev partial sum.
    """
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
        mps += coef_tensor[each_order] * join_list(each_mps)
        mps = mps.join(strategy=strategy)
    return mps


def chebyshev_clenshaw(
    func: Callable, mesh: Mesh, order: int, strategy: Strategy = Strategy()
) -> MPS:
    """
    Encode a function in a Matrix Product State (MPS) using the Clenshaw algorithm.

    Parameters:
        func (Callable): The function to be encoded.
        mesh (Mesh): The Chebyshev mesh over which the function is approximated.
        order (int): The order of the Chebyshev expansion.
        strategy (Strategy): An optional strategy for joining tensors during the computation.

    Returns:
        MPS: An MPS encoding the function using the Clenshaw algorithm for Chebyshev expansion.
    """
    # TODO: Generalize to multivariate case
    coef_vector = _coef_tensor(func, mesh, order)
    interval = deepcopy(mesh.intervals[0])
    interval.start = -1.0
    interval.stop = 1.0
    mps0 = mps_position(interval)

    c = np.flip(coef_vector)
    zero_mps = mps_empty(len(mps0))
    id_mps = mps_identity(len(mps0))
    y = [zero_mps] * (len(c) + 2)
    for i in range(2, len(y)):
        y[i] = (c[i - 2] * id_mps - y[i - 2] + 2 * mps0 * y[i - 1]).join(strategy=strategy)
    mps = (y[-1] - mps0 * y[-2]).join(strategy=strategy)
    return mps


def chebyshev_factor(func, mesh, orders, strategy):
    """
    Encode a function in a Matrix Product State (MPS) using the factorization method.

    Parameters:
        func (Callable): The function to be encoded.
        mesh (Mesh): The Chebyshev mesh over which the function is approximated.
        order (int): The order of the Chebyshev expansion.
        strategy (Strategy): An optional strategy for joining tensors during the computation.

    Returns:
        MPS: An MPS encoding the function using the monomial representation of the Chebyshev polynomial.
    """
    if mesh.dimension != 1:
        raise ValueError("At the moment this method only works for one-dimensional functions")
    # 1. Compute Chebyshev coefficients
    coef_tensor = _coef_tensor(func, mesh, orders)

    # 2. Find the roots of the polynomial in standard form
    companion = np.polynomial.chebyshev.chebcompanion(coef_tensor)
    roots = eigvals(companion)

    # 3. Represent each monomial as an MPS
    mps_list = []
    mps_x = mps_position(mesh.interval)
    for root in roots:
        mps_root = root * mps_identity
        mps_list.append(mps_x - mps_root)

    # 4. Compute the wavefunction product of all the MPS
    mps = mps_list[0]
    for mps_term in mps_list[1:]:
        mps *= mps_term
        mps = simplify(mps, strategy)
    return mps


def _put_at(indices, axis):
    """
    Auxiliary function to index tensors more easily.
    From: https://stackoverflow.com/questions/42656930/numpy-assignment-like-numpy-take

    This function provides a convenient way to index tensors by specifying the indices to put
    at a specific axis, and allows to use a notation similar than with 1d tensors.

    Parameters:
        indices (int or slice or array-like): The indices to put at the specified axis.
        axis (int): The axis at which to place the indices.

    Returns:
        tuple: A tuple of slice objects and indices that can be used for indexing tensors.
    """
    slc = (slice(None),)
    return (axis < 0) * (Ellipsis,) + axis * slc + (indices,) + (-1 - axis) * slc


def _differentiate_coef_tensor(coef_tensor: np.ndarray, m: int) -> np.ndarray:
    """
    Compute the derivative along a specified dimension of a tensor of Chebyshev coefficients
    exploiting the properties of Chebyshev polynomials.

    Parameters:
        coef_tensor (np.ndarray): The tensor of Chebyshev coefficients.
        m (int): The dimension along which the derivative is computed.

    Returns:
        np.ndarray: The resulting tensor of coefficients after differentiation along the specified dimension.
    """
    shape = coef_tensor.shape
    d = shape[m]
    ctensor_d = np.zeros(shape[:m] + (d + 1,) + shape[m + 1 :])
    for i in range(d - 2, -1, -1):
        ctensor_d[_put_at(i, m)] = 2 * (i + 1) * np.take(coef_tensor, i + 1, axis=m) + np.take(
            ctensor_d, i + 2, axis=m
        )
    ctensor_d[_put_at(range(d), m)] /= 2
    ctensor_d[_put_at(0, m)] /= 2
    return np.take(ctensor_d, range(d - 1), axis=m)


def _integrate_coef_tensor(
    coef_tensor: np.ndarray, m: int, c_int: Optional[float] = None
) -> np.ndarray:
    """
    Compute the integral along a specified dimension of a tensor of Chebyshev coefficients
    exploiting the properties of Chebyshev polynomials.

    Parameters:
        coef_tensor (np.ndarray): The tensor of Chebyshev coefficients.
        m (int): The dimension along which the integral is computed.

    Returns:
        np.ndarray: The resulting tensor of coefficients after integration along the specified dimension.
    """
    shape = coef_tensor.shape
    d = shape[m]
    ctensor_i = np.zeros(shape[:m] + (d - 2,) + shape[m + 1 :])
    ctensor_i[_put_at(1, m)] = (
        2 * np.take(coef_tensor, 0, axis=m) - np.take(coef_tensor, 2, axis=m)
    ) / 2
    for i in range(2, d - 2):
        ctensor_i[_put_at(i, m)] = (
            np.take(coef_tensor, i - 1, axis=m) - np.take(coef_tensor, i + 1, axis=m)
        ) / (2 * i)
    ctensor_i[_put_at(range(d - 2), m)] *= 2
    if c_int == None:
        c_int = np.take(coef_tensor, 0, axis=m)  # Equivalent to c=0
    ctensor_i[_put_at(0, m)] = c_int  # Constant of integration
    return ctensor_i


# def mpo_chebyshev(
#     mpo_0: Union[MPO, MPOSum],
#     order: int,
#     name: str,
#     strategy: Strategy = Strategy(tolerance=DEFAULT_TOLERANCE),
# ) -> MPO:
#     """
#     Returns a Chebyshev polynomial of order d of a MPO.

#     Parameters:
#         mpo_0 (MPS): The MPS.
#         order (int): The order of the Chebyshev polynomial.
#         name (str, optional): The name of the dataset to store results in an HDF5 file.
#                               Default is None (no storage).
#         strategy (Strategy): An optional strategy for the SeeMPS methods on the MPS.

#     Returns:
#         MPO: The MPO corresponding to the Chebyshev polynomial of the starting MPO mpo_0.

#     Example:
#         mpo_initial = ...  # Create an initial MPO
#         order = 5  # Specify the order of the Chebyshev polynomial
#         result_mpo = mpo_chebyshev(mpo_initial, order)
#         # 'result_mpo' will contain the MPO after applying the Chebyshev polynomial of order 5.
#     """
#     path = DATA_PATH + name + ".hdf5" if name is not None else None
#     sites = len(mpo_0)
#     try:
#         with h5py.File(path, "r") as file:
#             Ti = read_mpo(file, f"order_{order}")
#             Tj = read_mpo(file, f"order_{order-1}") if order > 0 else mpo_identity(sites)
#     except:
#         if order == 0:
#             Ti = mpo_identity(sites)
#             Tj = mpo_identity(sites)
#         elif order == 1:
#             Ti = mpo_0
#             Tj = mpo_identity(sites)
#         else:
#             Tj = mpo_chebyshev(mpo_0, order - 1, name)
#             Tk = mpo_chebyshev(mpo_0, order - 2, name)
#             Ti = (2.0 * MPOList([mpo_0, Tj]).join(strategy) - Tk).join(strategy)
#         if name is not None:
#             with h5py.File(path, "a") as file:
#                 write_mpo(file, f"order_{order}", Ti)
#     return Ti


# def chebyshev_compose(
#     func: Callable,
#     tensor_network_0: Union[MPS, MPO, MPOSum],
#     order: int,
#     start: float,
#     stop: float,
#     strategy: Strategy = Strategy(tolerance=DEFAULT_TOLERANCE),
# ) -> Union[MPS, MPOSum]:
#     mesh = Mesh([ChebyshevZerosInterval(start, stop, 0)])
#     coef_vector = _coef_tensor(func, mesh, [order]).flatten()
#     if isinstance(tensor_network_0, MPS):
#         tensor_network = mps_empty(len(tensor_network_0))
#         for idx, coef in enumerate(coef_vector):
#             name = f"mps_chebyshev-type_open-sites_{len(tensor_network_0)}"
#             tensor_network += coef * mps_chebyshev(tensor_network_0, idx, name)
#             tensor_network = tensor_network.toMPS(strategy=strategy)
#     elif isinstance(tensor_network_0, Union[MPO, MPOSum]):
#         tensor_network = mpo_empty(len(tensor_network_0))
#         for idx, coef in enumerate(coef_vector):
#             name = f"mpo_chebyshev-type_open-sites_{len(tensor_network_0)}"
#             tensor_network += coef * mpo_chebyshev(tensor_network_0, idx, name)
#     else:
#         raise ValueError("Invalid tensor network")
#     return tensor_network

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
