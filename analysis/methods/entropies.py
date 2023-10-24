from typing import List
from seemps.state import MPS, CanonicalMPS


def entanglement_entropies(mps: MPS) -> List[int]:
    """Returns all the bipartite entanglement entropies of a MPS."""
    return [
        CanonicalMPS(mps._data, center=i).entanglement_entropy()
        for i in range(mps.size)
    ]
