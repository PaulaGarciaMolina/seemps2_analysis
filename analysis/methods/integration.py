import numpy as np
from scipy.fftpack import ifft
from typing import List

from seemps.state import MPS, Strategy, DEFAULT_TOLERANCE
from seemps.expectation import scprod
from seemps.cross import Mesh
from .factories import mps_identity

QUADRATURE_STRATEGY = Strategy(tolerance=DEFAULT_TOLERANCE)


def mps_midpoint(start: float, stop: float, sites: int) -> MPS:
    """Returns the MPS corresponding to the midpoint quadrature rule in the interval
    [start, stop] with 2**sites points."""
    step = (stop - start) / (2**sites - 1)
    return step * mps_identity(sites)


def mps_trapezoidal(
    start: float, stop: float, sites: int, from_vector: bool = True
) -> MPS:
    """Returns the MPS corresponding to the trapezoidal quadrature rule in the interval
    [start, stop] with 2**sites points."""
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
        raise ValueError("MPS not implemented")
    step = (stop - start) / (2**sites - 1)
    return step * mps


def mps_simpson(start: float, stop: float, sites: int, from_vector: bool = True) -> MPS:
    """Returns the MPS corresponding to the Simpson quadrature rule in the interval
    [start, stop] with 2**sites points."""
    if sites % 2 != 0:
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
        raise ValueError("MPS not implemented")
    step = (stop - start) / (2**sites - 1)
    return (3 * step / 8) * mps


def mps_fifth_order(
    start: float, stop: float, sites: int, from_vector: bool = True
) -> MPS:
    """Returns the MPS corresponding to the fifth-order quadrature rule in the interval
    [start, stop] with 2**sites points."""
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
        raise ValueError("MPS not implemented")
    step = (stop - start) / (2**sites - 1)
    return (5 * step / 288) * mps


def mps_fejer(start: float, stop: float, points: int, from_vector: bool = True) -> MPS:
    """Returns the MPS corresponding to the Féjer quadrature rule in the interval
    [start, stop] with N = 2**sites points located on the zeros of the N-th Chebyshev
    polynomial.
    """

    def _fejer_weights(d: int) -> np.ndarray:
        """Computes the vector of Féjer weigths of a given length d."""
        N = np.arange(start=1, stop=d, step=2)[:, None]
        l = N.size
        v0 = [2 * np.exp(1j * np.pi * k / d) / (1 - 4 * k**2) for k in range(l)] + [
            0
        ] * (l + 1)
        v1 = v0[0:-1] + np.conj(v0[:0:-1])
        wf1 = ifft(v1).flatten().real
        return wf1

    if from_vector:
        vector = _fejer_weights(2**points)
        mps = MPS.from_vector(
            vector,
            [2 for _ in range(points)],
            normalize=False,
            strategy=QUADRATURE_STRATEGY,
        )
    else:
        raise ValueError("MPS not implemented")
    step = (stop - start) / 2
    return step * mps


def integrate_mps(mps: MPS, mesh: Mesh, integral_type: str) -> float:
    """Returns the integral of a MPS that codifies a multivariate function in a given
    mesh with respect to a quadrature rule given by the parameter integration_type."""
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
        return None
        # raise ValueError("Invalid integral_type")

    mps_list = []
    for interval in mesh.intervals:
        mps_list.append(
            factory(interval.start, interval.stop, int(np.log2(interval.size)))
        )
    return scprod(mps, _join(mps_list))


def _join(mps_list: List[MPS]) -> MPS:
    """Returns a MPS that is given by the union of a list of MPS by their extremes."""
    nested_sites = [mps._data for mps in mps_list]
    flattened_sites = [site for sites in nested_sites for site in sites]
    return MPS(flattened_sites)
