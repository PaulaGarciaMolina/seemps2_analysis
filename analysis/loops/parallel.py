import numpy as np
from typing import Callable, Optional


def param_loop(
    func: Callable, params: dict, name: str = "", path: Optional[str] = None
) -> np.ndarray:
    """
    Evaluate a function in parallel on all combinations of parameters defined in the params
    dictionary and optionally save the result in a pickle file.

    This takes advantage of the "embarrassingly parallel" nature of evaluating a loop and
    distributes the work across multiple processes or cores to get a large speedup.

    Parameters:
        func (callable): The function to be evaluated.
        params (dict): A dictionary where keys are parameter names and values are lists of
            parameter values to be combined.
        name (str, optional): A name prefix for the pickle output files (default: "").
        path (str, optional): The path where intermediate results are saved and loaded from
            (default: None).

    Returns:
        np.ndarray: An array containing the results of function evaluations for all parameter combinations.
    """
    # TODO: Implement
    pass
