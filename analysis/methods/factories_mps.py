import numpy as np
from typing import List, Union

from seemps.cross import Interval
from seemps.state import MPS
from seemps.operators import MPO


def mps_identity(sites: int) -> MPS:
    """
    Return a Matrix Product State (MPS) representing the identity vector, which
    is a superposition state given by (|000...> + |111...>) in an unnormalized form.

    Parameters:
        sites (int): The number of sites or qubits for the identity vector.
    """
    return MPS([np.ones((1, 2, 1))] * sites)


def mps_empty(sites: int) -> MPS:
    """
    Return a Matrix Product State (MPS) representing a vector of zeros, which
    is an unnormalized state of the form |000...>.

    Parameters:
        sites (int): The number of sites or qubits for the zero vector.

    """
    return MPS([np.zeros((1, 2, 1))] * sites)


def mps_interval(interval: Interval) -> MPS:
    """
    Return a Matrix Product State (MPS) corresponding to the specified interval.

    Parameters:
        interval (Interval): The interval for which the MPS is constructed.

    """
    sites = int(np.log2(interval.size))
    if interval.type == "open":
        pass
    elif interval.type == "closed":
        pass
    elif interval.type == "zeros":
        pass


def mps_exponential():
    # TODO: Implement
    pass


def mps_cosine(start: float, stop: float, sites: int) -> MPS:
    """
    Return a Matrix Product State (MPS) corresponding to the cosine function over the
    interval [start, stop] by means of the exponential MPS constructed exactly with
    bond dimension 1.

    Parameters:
        start (float): The starting point of the interval.
        stop (float): The ending point of the interval.
        sites (int): The number of sites or qubits for the MPS.

    """
    mps_1 = mps_exponential(start, stop, sites, c=1j)
    mps_2 = mps_exponential(start, stop, sites, c=-1j)

    return (0.5 * (mps_1 + mps_2)).toMPS()


def join_list(tn_list: List[Union[MPS, MPO]]) -> Union[MPS, MPO]:
    """
    Join a list of tensor networks (MPS or MPO) on their extremes.

    Parameters:
        tn_list (List[Union[MPS, MPO]]): A list of tensor networks to be joined.
    """
    nested_sites = [tn._data for tn in tn_list]
    flattened_sites = [site for sites in nested_sites for site in sites]
    if all(isinstance(tn, MPS) for tn in tn_list):
        return MPS(flattened_sites)
    elif all(isinstance(tn, MPO) for tn in tn_list):
        return MPO(flattened_sites)
    else:
        raise ValueError("All the tensor networks must be of the same type (either MPS or MPO).")


# def mps_position(start: float, stop: float, sites: int, mesh_type: str = "o") -> MPS:
#     """Returns a MPS corresponding to an interval x of a certain type given by
#     the mesh_type parameter. Options:
#         - "o": Half-open interval [start, stop).
#         - "c": Closed interval [start, stop].
#         - "z": Affine map between the zeros of the N-th Chebyshev polynomial
#                in [-1, 1] to (start, stop).
#     """
#     assert mesh_type in ["o", "c", "z"], "Invalid mesh_type"
#     if mesh_type == "o":
#         mps = mpo_position(start, stop, sites) @ mps_identity(sites)
#     elif mesh_type == "c":
#         stop += (stop - start) / (2**sites - 1)
#         mps = mpo_position(start, stop, sites) @ mps_identity(sites)
#     elif mesh_type == "z":
#         start_mapped = np.pi / (2 ** (sites + 1))
#         stop_mapped = np.pi + start_mapped
#         mps = -1.0 * mps_cosine(start_mapped, stop_mapped, sites)
#     return mps


# def join(mps_list: List[MPS]) -> MPS:
#     """
#     Combine multiple Matrix Product States (MPS) by their extremes and return a new MPS.

#     Parameters:
#         mps_list (List[MPS]): A list of Matrix Product States to be combined.
#     """
#     nested_sites = [mps._data for mps in mps_list]
#     flattened_sites = [site for sites in nested_sites for site in sites]
#     return MPS(flattened_sites)
