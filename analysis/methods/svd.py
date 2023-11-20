from typing import Callable
from seemps.cross import Mesh
from seemps.state import MPS, Strategy, DEFAULT_TOLERANCE


def svd(
    func: Callable,
    mesh: Mesh,
    strategy: Strategy = Strategy(tolerance=DEFAULT_TOLERANCE),
) -> MPS:
    """
    Perform Singular Value Decomposition (SVD) on a multivariate function to construct an MPS representation.

    Parameters:
        func (callable): The function to be approximated and represented as an MPS.
        mesh (Mesh): The mesh on which the function is sampled.
        strategy (Strategy, optional): The strategy used for SVD and MPS construction (default: Strategy with DEFAULT_TOLERANCE).
    """
    tensor = func(mesh.to_tensor())
    return MPS.from_tensor(tensor, strategy=strategy, normalize=False)
