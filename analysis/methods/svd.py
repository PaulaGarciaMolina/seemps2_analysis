from typing import Callable
from seemps.cross import Mesh
from seemps.state import MPS, Strategy, DEFAULT_TOLERANCE


def svd(
    func: Callable,
    mesh: Mesh,
    strategy: Strategy = Strategy(tolerance=DEFAULT_TOLERANCE),
    normalize: bool = False,
) -> MPS:
    tensor = func(mesh.to_tensor())
    return MPS.from_tensor(tensor, strategy=strategy, normalize=normalize)
