from typing import List
from seemps.state import MPS, CanonicalMPS


def entanglement_entropies(mps: MPS) -> List[int]:
    """
    Calculate and return all the bipartite entanglement entropies of a Matrix Product State (MPS).

    Parameters:
        mps (MPS): The Matrix Product State for which entanglement entropies are calculated.
    """
    return [CanonicalMPS(mps._data, center=i).entanglement_entropy() for i in range(mps.size)]
