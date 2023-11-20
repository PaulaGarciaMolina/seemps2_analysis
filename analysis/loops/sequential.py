import numpy as np
import pickle
from typing import Callable, Optional
from itertools import product


def param_loop(
    func: Callable, params: dict, name: str = "", path: Optional[str] = None
) -> np.ndarray:
    """
    Evaluate a function on all combinations of parameters defined in the params
    dictionary and optionally save the result in a pickle file.

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
    param_names = list(params.keys())
    param_values = list(params.values())
    dims = [len(value) for value in param_values]
    data = np.empty(dims, dtype=object)

    def enumerated_product(*args):
        yield from zip(product(*(range(len(x)) for x in args)), product(*args))

    for idx, value in enumerated_product(*param_values):
        prefix = ["".join(map(str, i)) for i in zip(param_names, value)]
        filename = name + "_" + "_".join(prefix) + ".pkl"
        try:
            with open(path + filename, "rb") as f:
                data[idx] = pickle.load(f)
                print(f"Loading {filename}")
        except:
            print(f"Computing {filename}")
            data[idx] = func(*value)
            if path:
                with open(path + filename, "wb+") as f:
                    pickle.dump(data[idx], f, 2)
                    print(f"Saving {filename}")
    return data


# def param_loop_hdf5(func, funcdim, params, path=None, file="", group=""):
#     """
#     Runs a function on all the combinations of the parameters inside the params dictionary
#     and saves each value as a dataset inside a group of a hdf5 file.
#     """
#     param_names = list(params.keys())
#     param_values = list(params.values())
#     dims = [len(value) for value in param_values]
#     data = np.zeros(dims + [funcdim])

#     def enumerated_product(*args):
#         yield from zip(product(*(range(len(x)) for x in args)), product(*args))

#     for idx, value in enumerated_product(*param_values):
#         prefix = ["".join(map(str, i)) for i in zip(param_names, value)]
#         dataset = "_".join(prefix)
#         try:
#             hdf5_path = path + file + ".hdf5"
#             with h5py.File(hdf5_path, "r") as f:
#                 print(f"Loading {file + '/' + group + '/' + dataset}")
#                 data[idx] = f[group][dataset][:]
#         except:
#             print(f"Computing and saving {file + '/' + group + '/' + dataset}")
#             data[idx] = func(*value)
#             if path:
#                 with h5py.File(hdf5_path, "a") as f:
#                     g = f.require_group(group)
#                     g.create_dataset(dataset, data=data[idx])
#     return data
