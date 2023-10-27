import numpy as np
import math
from typing import Callable, List
from numpy.linalg import inv, det


def _rotation_matrix(
    angle: float, vector_1: List[float], vector_2: List[float]
) -> np.ndarray:
    """Returns a matrix corresponding to a rotation with a given angle along an axis
    spanned by the two given vectors. From https://pypi.org/project/mgen/"""
    vector_1 = np.asarray(vector_1, dtype=float)
    vector_2 = np.asarray(vector_2, dtype=float)
    vector_2 /= np.linalg.norm(vector_2)
    dot_value = np.dot(vector_1, vector_2)

    if abs(dot_value / np.linalg.norm(vector_1)) > 1:
        raise ValueError("Given vectors are parallel.")

    vector_1 = vector_1 - dot_value * vector_2
    vector_1 /= np.linalg.norm(vector_1)

    vectors = np.vstack([vector_1, vector_2]).T
    vector_1, vector_2 = np.linalg.qr(vectors)[0].T

    V = np.outer(vector_1, vector_1) + np.outer(vector_2, vector_2)
    W = np.outer(vector_1, vector_2) - np.outer(vector_2, vector_1)

    return np.eye(len(vector_1)) + (math.cos(angle) - 1) * V - math.sin(angle) * W


def _gaussian_base(dim: int, cov: np.ndarray, normalize: bool = True) -> Callable:
    """Helper function to produce multivariate Gaussian distributions with a given
    dimensionality, covariance matrix and normalization."""
    if cov.shape != (dim, dim):
        raise ValueError("Invalid covariance matrix")
    cov_squared = inv(cov)
    prefactor = 1 / np.sqrt((2 * np.pi) ** dim * det(cov)) if normalize else 1
    return lambda vec: prefactor * np.exp(-0.5 * np.transpose(vec) @ cov_squared @ vec)


def gaussian_product(
    dim: int, σ_vector: List[float], normalize: bool = True
) -> Callable:
    """Returns a multivariate Gaussian distribution with a diagonal covariance matrix
    (meaning that can be factorized in a tensor product of univariate distributions)
    given by the σ_vector parameter."""
    cov = np.diag(σ_vector)
    return _gaussian_base(dim, cov, normalize=normalize)


def gaussian_squeezed(
    dim: int, σ_vector: List[float], θ_vector: List[float], normalize: bool = True
) -> Callable:
    """Returns a multivariate Gaussian distribution with a covariance matrix given by
    the rotation of a diagonal matrix given by the σ_vector parameter along a collection
    of axes with magnitudes given by the θ_vector parameter."""
    cov = np.diag(σ_vector)
    for idx, θ in enumerate(θ_vector):
        if θ != 0:
            vector_1 = [0] * dim
            vector_1[idx] = 1
            vector_2 = [0] * dim
            vector_2[idx + 1] = 1
            rotation = _rotation_matrix(θ, vector_1, vector_2)
            cov = rotation @ cov @ rotation.T
    return _gaussian_base(dim, cov, normalize=normalize)


def gaussian_maxsqueezed(
    dim: int, σ: float = 0.1, axis: int = 0, normalize: bool = True
) -> Callable:
    """Returns a multivariate Gaussian distribution rotated $\pi / 4$ degrees
    along a given axis and with a variance of σ along it."""
    σ_vector = [1] * dim
    σ_vector[axis] = σ
    θ_vector = [0] * (dim - 1)
    θ_vector[axis] = np.pi / 4
    return gaussian_squeezed(dim, σ_vector, θ_vector, normalize=normalize)
