import numpy as np


def distance_norm_1(tensor1: np.ndarray, tensor2: np.ndarray, normalize: bool = True) -> float:
    """Returns the distance in the L1 norm between two tensors of the same shape."""
    if tensor1.shape != tensor2.shape:
        raise ValueError("The tensors are of different shape")
    if normalize:
        return np.sum(np.abs(tensor1 - tensor2)) / np.prod(tensor1.shape)
    else:
        return np.sum(np.abs(tensor1 - tensor2))


def distance_norm_2(tensor1: np.ndarray, tensor2: np.ndarray, normalize: bool = True) -> float:
    """Returns the distance in the L2 norm between two tensors of the same shape."""
    if tensor1.shape != tensor2.shape:
        raise ValueError("The tensors are of different shape")
    if normalize:
        return np.sqrt(np.sum((tensor1 - tensor2) ** 2)) / np.sqrt(np.prod(tensor1.shape))
    else:
        return np.sqrt(np.sum((tensor1 - tensor2) ** 2))


def distance_norm_inf(tensor1: np.ndarray, tensor2: np.ndarray, normalize: bool = False) -> float:
    """Returns the distance in the Chebyshev norm between two tensors of the same shape."""
    if tensor1.shape != tensor2.shape:
        raise ValueError("The tensors are of different shape")
    if normalize:
        return np.max(np.abs(tensor1 - tensor2)) / np.prod(tensor1.shape)
    else:
        return np.max(np.abs(tensor1 - tensor2))
