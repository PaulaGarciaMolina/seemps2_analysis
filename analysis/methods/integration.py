import numpy as np
from scipy.fftpack import ifft
from typing import List

from seemps.state import MPS, Strategy, DEFAULT_TOLERANCE
from seemps.expectation import scprod
from seemps.cross import Mesh
from .factories_mps import mps_identity, join_list

QUADRATURE_STRATEGY = Strategy(tolerance=DEFAULT_TOLERANCE)


def mps_midpoint(start: float, stop: float, sites: int) -> MPS:
    """
    Returns a Matrix Product State (MPS) corresponding to the midpoint quadrature rule.

    Parameters:
        start (float): The starting point of the interval.
        stop (float): The ending point of the interval.
        sites (int): The number of sites or qubits for the MPS.
    """
    step = (stop - start) / (2**sites - 1)
    return step * mps_identity(sites)


def mps_trapezoidal(start: float, stop: float, sites: int, from_vector: bool = False) -> MPS:
    """
    Returns a Matrix Product State (MPS) corresponding to the trapezoidal quadrature rule.

    Parameters:
        start (float): The starting point of the interval.
        stop (float): The ending point of the interval.
        sites (int): The number of sites or qubits for the MPS.
        from_vector (bool): Whether to construct the MPS from a vector (True) or tensors (False).
    """
    if from_vector:
        vector = np.ones(2**sites)
        vector[0] = vector[-1] = 0.5
        mps = MPS.from_vector(
            vector,
            [2 for _ in range(sites)],
            normalize=False,
            strategy=QUADRATURE_STRATEGY,
        )
    else:
        tensor_1 = np.zeros((1, 2, 3))
        tensor_1[0, 0, 0] = 1
        tensor_1[0, 1, 1] = 1
        tensor_1[0, 0, 2] = 1
        tensor_1[0, 1, 2] = 1
        tensor_bulk = np.zeros((3, 2, 3))
        tensor_bulk[0, 0, 0] = 1
        tensor_bulk[1, 1, 1] = 1
        tensor_bulk[2, 0, 2] = 1
        tensor_bulk[2, 1, 2] = 1
        tensor_2 = np.zeros((3, 2, 1))
        tensor_2[0, 0, 0] = -0.5
        tensor_2[1, 1, 0] = -0.5
        tensor_2[2, 0, 0] = 1
        tensor_2[2, 1, 0] = 1
        tensors = [tensor_1] + [tensor_bulk for _ in range(sites - 2)] + [tensor_2]
        mps = MPS(tensors)
    step = (stop - start) / (2**sites - 1)
    return step * mps


def mps_simpson(start: float, stop: float, sites: int, from_vector: bool = False) -> MPS:
    """
    Return a Matrix Product State (MPS) corresponding to the Simpson quadrature rule.

    Parameters:
        start (float): The starting point of the interval.
        stop (float): The ending point of the interval.
        sites (int): The number of sites or qubits for the MPS.
        from_vector (bool): Whether to construct the MPS from a vector (True) or tensors (False).
    """
    if sites % 2 != 0 and not sites > 2:
        raise ValueError("The sites must be divisible by 2.")
    if from_vector:
        vector = np.ones(2**sites)
        vector[1:-1:3] = 3
        vector[2:-1:3] = 3
        vector[3:-1:3] = 2
        vector[0] = vector[-1] = 1
        mps = MPS.from_vector(
            vector,
            [2 for _ in range(sites)],
            normalize=False,
            strategy=QUADRATURE_STRATEGY,
        )
    else:
        tensor_1 = np.zeros((1, 2, 4))
        tensor_1[0, 0, 0] = 1
        tensor_1[0, 1, 1] = 1
        tensor_1[0, 0, 2] = 1
        tensor_1[0, 1, 3] = 1
        if sites == 2:
            tensor_2 = np.zeros((4, 2, 1))
            tensor_2[0, 0, 0] = -1
            tensor_2[1, 1, 0] = -1
            tensor_2[2, 0, 0] = 2
            tensor_2[2, 1, 0] = 3
            tensor_2[3, 0, 0] = 3
            tensor_2[3, 1, 0] = 2
            tensors = [tensor_1, tensor_2]
        else:
            tensor_2 = np.zeros((4, 2, 5))
            tensor_2[0, 0, 0] = 1
            tensor_2[1, 1, 1] = 1
            tensor_2[2, 0, 2] = 1
            tensor_2[2, 1, 3] = 1
            tensor_2[3, 0, 4] = 1
            tensor_2[3, 1, 2] = 1
            tensor_bulk = np.zeros((5, 2, 5))
            tensor_bulk[0, 0, 0] = 1
            tensor_bulk[1, 1, 1] = 1
            tensor_bulk[2, 0, 2] = 1
            tensor_bulk[2, 1, 3] = 1
            tensor_bulk[3, 0, 4] = 1
            tensor_bulk[3, 1, 2] = 1
            tensor_bulk[4, 0, 3] = 1
            tensor_bulk[4, 1, 4] = 1
            tensor_3 = np.zeros((5, 2, 1))
            tensor_3[0, 0, 0] = -1
            tensor_3[1, 1, 0] = -1
            tensor_3[2, 0, 0] = 2
            tensor_3[2, 1, 0] = 3
            tensor_3[3, 0, 0] = 3
            tensor_3[3, 1, 0] = 2
            tensor_3[4, 0, 0] = 3
            tensor_3[4, 1, 0] = 3
            tensors = [tensor_1, tensor_2] + [tensor_bulk for _ in range(sites - 3)] + [tensor_3]
        mps = MPS(tensors)
    step = (stop - start) / (2**sites - 1)
    return (3 * step / 8) * mps


def mps_fifth_order(start: float, stop: float, sites: int, from_vector: bool = False) -> MPS:
    """
    Return a Matrix Product State (MPS) corresponding to the fifth-order quadrature rule.

    Parameters:
        start (float): The starting point of the interval.
        stop (float): The ending point of the interval.
        sites (int): The number of sites or qubits for the MPS.
        from_vector (bool): Whether to construct the MPS from a vector (True) or tensors (False).
    """
    if sites % 4 != 0:
        raise ValueError("The sites must be divisible by 4.")
    if from_vector:
        vector = np.ones(2**sites)
        vector[1:-1:5] = 75
        vector[2:-1:5] = 50
        vector[3:-1:5] = 50
        vector[4:-1:5] = 75
        vector[5:-1:5] = 38
        vector[0] = vector[-1] = 19
        mps = MPS.from_vector(
            vector,
            [2 for _ in range(sites)],
            normalize=False,
            strategy=QUADRATURE_STRATEGY,
        )
    else:
        tensor_1 = np.zeros((1, 2, 4))
        tensor_1[0, 0, 0] = 1
        tensor_1[0, 1, 1] = 1
        tensor_1[0, 0, 2] = 1
        tensor_1[0, 1, 3] = 1
        tensor_2 = np.zeros((4, 2, 6))
        tensor_2[0, 0, 0] = 1
        tensor_2[1, 1, 1] = 1
        tensor_2[2, 0, 2] = 1
        tensor_2[2, 1, 3] = 1
        tensor_2[3, 0, 4] = 1
        tensor_2[3, 1, 5] = 1
        tensor_3 = np.zeros((6, 2, 7))
        tensor_3[0, 0, 0] = 1
        tensor_3[1, 1, 1] = 1
        tensor_3[2, 0, 2] = 1
        tensor_3[2, 1, 3] = 1
        tensor_3[3, 0, 4] = 1
        tensor_3[3, 1, 5] = 1
        tensor_3[4, 0, 6] = 1
        tensor_3[4, 1, 2] = 1
        tensor_3[5, 0, 3] = 1
        tensor_3[5, 1, 4] = 1
        tensor_bulk = np.zeros((7, 2, 7))
        tensor_bulk[0, 0, 0] = 1
        tensor_bulk[1, 1, 1] = 1
        tensor_bulk[2, 0, 2] = 1
        tensor_bulk[2, 1, 3] = 1
        tensor_bulk[3, 0, 4] = 1
        tensor_bulk[3, 1, 5] = 1
        tensor_bulk[4, 0, 6] = 1
        tensor_bulk[4, 1, 2] = 1
        tensor_bulk[5, 0, 3] = 1
        tensor_bulk[5, 1, 4] = 1
        tensor_bulk[6, 0, 5] = 1
        tensor_bulk[6, 1, 6] = 1
        tensor_4 = np.zeros((7, 2, 1))
        tensor_4[0, 0, 0] = -19
        tensor_4[1, 1, 0] = -19
        tensor_4[2, 0, 0] = 38
        tensor_4[2, 1, 0] = 75
        tensor_4[3, 0, 0] = 50
        tensor_4[3, 1, 0] = 50
        tensor_4[4, 0, 0] = 75
        tensor_4[4, 1, 0] = 38
        tensor_4[5, 0, 0] = 75
        tensor_4[5, 1, 0] = 50
        tensor_4[6, 0, 0] = 50
        tensor_4[6, 1, 0] = 75
        tensors = (
            [tensor_1, tensor_2, tensor_3] + [tensor_bulk for _ in range(sites - 4)] + [tensor_4]
        )
        mps = MPS(tensors)
    step = (stop - start) / (2**sites - 1)
    return (5 * step / 288) * mps


def mps_fejer(start: float, stop: float, points: int, from_vector: bool = True) -> MPS:
    """
    Return a Matrix Product State (MPS) corresponding to the Féjer quadrature rule.

    Parameters:
        start (float): The starting point of the interval.
        stop (float): The ending point of the interval.
        points (int): The number of quadrature points.
        from_vector (bool): Whether to construct the MPS from a vector (True) or tensors (False).
    """
    if from_vector:
        d = 2**points
        N = np.arange(start=1, stop=d, step=2)[:, None]
        l = N.size
        v0 = [2 * np.exp(1j * np.pi * k / d) / (1 - 4 * k**2) for k in range(l)] + [0] * (l + 1)
        v1 = v0[0:-1] + np.conj(v0[:0:-1])
        vector = ifft(v1).flatten().real
        mps = MPS.from_vector(
            vector,
            [2 for _ in range(points)],
            normalize=False,
            strategy=QUADRATURE_STRATEGY,
        )
    else:
        # TODO: Implement (low priority as it converges so fast and not many qubits are needed)
        # In principle it can be implemented using the exponential MPS and iqft.
        raise ValueError("MPS not implemented")
    step = (stop - start) / 2
    return step * mps


def integrate_mps(mps: MPS, mesh: Mesh, integral_type: str) -> float:
    """
    Calculate the integral of an MPS encoding a multivariate function over a given mesh
    using a quadrature rule determined by the parameter 'integral_type'.

    Parameters:
        mps (MPS): The MPS representing the function to be integrated.
        mesh (Mesh): The mesh over which the function is integrated.
        integral_type (str): The type of quadrature rule to use for integration.
            Supported types: 'midpoint', 'trapezoidal', 'simpson', 'fifth_order', 'fejer'.

    Returns:
        float: The computed integral value.

    Example:
        >>> # Define an MPS and mesh representing a function
        >>> mps = MPS([...])  # Replace with your MPS data
        >>> mesh = Mesh([...])  # Replace with your mesh data
        >>> # Calculate the integral using the midpoint quadrature rule
        >>> integral_value = integrate_mps(mps, mesh, 'midpoint')

    """
    if integral_type == "midpoint":
        factory = mps_midpoint
    elif integral_type == "trapezoidal":
        factory = mps_trapezoidal
    elif integral_type == "simpson" and len(mps) % 2 == 0:
        factory = mps_simpson
    elif integral_type == "fifth_order" and len(mps) % 4 == 0:
        factory = mps_fifth_order
    elif integral_type == "fejer":
        factory = mps_fejer
    else:
        raise ValueError("Invalid integral_type")

    mps_list = []
    for interval in mesh.intervals:
        mps_list.append(factory(interval.start, interval.stop, int(np.log2(interval.size))))
    return scprod(mps, join_list(mps_list))
