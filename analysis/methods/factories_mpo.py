import numpy as np

from seemps.mpo import MPO


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
        middle_tensors[i][0, :, :, 1] = np.array([[0, 0], [0, dx * 2 ** (sites - (i + 2))]])

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
