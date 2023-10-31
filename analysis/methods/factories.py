import numpy as np

from seemps.cross import Interval
from seemps.state import MPS, Strategy, DEFAULT_TOLERANCE
from seemps.mpo import MPO

QUADRATURE_STRATEGY = Strategy(tolerance=DEFAULT_TOLERANCE)


def mps_identity(sites: int) -> MPS:
    """Returns a MPS corresponding to the identity vector (a vector of zeros).
    Represents an unnormalized superposition state given by (|000...> + |111...>)."""
    return MPS([np.ones((1, 2, 1))] * sites)


def mps_empty(sites: int) -> MPS:
    """Returns a MPS corresponding to a vector of zeros.
    Represents an unnormalized state given by |000...> (?)."""
    return MPS([np.zeros((1, 2, 1))] * sites)


def mps_cosine(start: float, stop: float, sites: int) -> MPS:
    """Returns a MPS corresponding to a cosine of the interval x, $\cos{\vec{x}}$."""
    mps_1 = mpo_exponential(start, stop, sites, c=1j) @ mps_identity(sites)
    mps_2 = mpo_exponential(start, stop, sites, c=-1j) @ mps_identity(sites)

    return (0.5 * (mps_1 + mps_2)).toMPS()


def mps_position(interval: Interval) -> MPS:
    sites = int(np.log2(interval.size))
    if interval.type == "open":
        mps = mpo_position(interval.start, interval.stop, sites) @ mps_identity(sites)
    elif interval.type == "closed":
        stop = interval.stop + (interval.stop - interval.start) / (2**sites - 1)
        mps = mpo_position(interval.start, stop, sites) @ mps_identity(sites)
    elif interval.type == "zeros":
        start_mapped = np.pi / (2 ** (sites + 1))
        stop_mapped = np.pi + start_mapped
        mps = -1.0 * mps_cosine(start_mapped, stop_mapped, sites)
    return mps


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


def mpo_empty(sites: int) -> MPO:
    """Returns a MPO corresponding to the identity."""

    identity = np.zeros((1, 2, 2, 1))
    return MPO([identity] * sites)


def mpo_identity(sites: int) -> MPO:
    """Returns a MPO corresponding to the identity."""

    identity = np.zeros((1, 2, 2, 1))
    identity[0, :, :, 0] = np.eye(2)
    return MPO([identity] * sites)


def mpo_position(start: float, stop: float, sites: int) -> MPO:
    """Returns a MPO corresponding to the position operator $\hat{x}$, which maps an
    identity MPS to a MPS corresponding to the mesh vector $\vec{x}$."""
    dx = (stop - start) / 2**sites

    left_tensor = np.zeros((1, 2, 2, 2))
    left_tensor[0, :, :, 0] = np.eye(2)
    left_tensor[0, :, :, 1] = np.array([[start, 0], [0, start + dx * 2 ** (sites - 1)]])

    right_tensor = np.zeros((2, 2, 2, 1))
    right_tensor[1, :, :, 0] = np.eye(2)
    right_tensor[0, :, :, 0] = np.array([[0, 0], [0, dx]])

    middle_tensors = [np.zeros((2, 2, 2, 2)) for _ in range(sites - 2)]
    for i in range(len(middle_tensors)):
        middle_tensors[i][0, :, :, 0] = np.eye(2)
        middle_tensors[i][1, :, :, 1] = np.eye(2)
        middle_tensors[i][0, :, :, 1] = np.array(
            [[0, 0], [0, dx * 2 ** (sites - (i + 2))]]
        )

    tensors = [left_tensor] + middle_tensors + [right_tensor]
    return MPO(tensors)


def mpo_exponential(start: float, stop: float, sites: int, c: float = 1.0) -> MPO:
    """Returns a MPO corresponding to the exponential operator $\exp{\hat{x}}$, which maps
    an identity MPS to a MPS corresponding to the exponential of the mesh vector $\exp{\vec{x}}$.
    """
    dx = (stop - start) / 2**sites

    left_tensor = np.zeros((1, 2, 2, 1), dtype=complex)
    left_tensor[0, :, :, 0] = np.array(
        [[np.exp(c * start), 0], [0, np.exp(c * start + c * dx * 2 ** (sites - 1))]],
        dtype=complex,
    )

    right_tensor = np.zeros((1, 2, 2, 1), dtype=complex)
    right_tensor[0, :, :, 0] = np.array([[1, 0], [0, np.exp(c * dx)]], dtype=complex)

    middle_tensors = [np.zeros((1, 2, 2, 1), dtype=complex) for _ in range(sites - 2)]
    for i in range(len(middle_tensors)):
        middle_tensors[i][0, :, :, 0] = np.array(
            [[1, 0], [0, np.exp(c * dx * 2 ** (sites - (i + 2)))]], dtype=complex
        )

    tensors = [left_tensor] + middle_tensors + [right_tensor]
    return MPO(tensors)
