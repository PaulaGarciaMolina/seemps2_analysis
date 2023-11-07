import numpy as np
import h5py
import pathlib

DATA_PATH = str(pathlib.Path(__file__).parent.absolute()) + "/data/"


def interval(start: float, stop: float, sites: int, mesh_type: str) -> np.ndarray:
    if mesh_type == "o":
        return np.array([start + i * (stop - start) / 2**sites for i in range(2**sites)])
    elif mesh_type == "c":
        stop += (stop - start) / (2**sites - 1)
        return np.array([start + i * (stop - start) / 2**sites for i in range(2**sites)])


def vector_chebyshev(sites: int, order: int, mesh_type: str = "o", name: str = None) -> np.ndarray:
    path = DATA_PATH + name + ".hdf5" if name is not None else None
    try:
        with h5py.File(path, "r") as file:
            Ti = file[f"order_{order}"][:]
            Tj = file[f"order_{order-1}"][:] if order > 0 else np.ones(2**sites)
    except:
        x = interval(-1, 1, sites, mesh_type)
        if order == 0:
            Ti = np.ones(2**sites)
            Tj = np.ones(2**sites)
        elif order == 1:
            Ti = x
            Tj = np.ones(2**sites)
        else:
            Tj = vector_chebyshev(sites, order - 1, name=name)
            Tk = vector_chebyshev(sites, order - 2, name=name)
            Ti = 2.0 * x * Tj - Tk
        if name is not None:
            with h5py.File(path, "a") as file:
                file.create_dataset(f"order_{order}", data=Ti)
    return Ti


def clenshaw_1d(coef_vector: np.ndarray, x: np.ndarray) -> np.ndarray:
    c = np.flip(coef_vector)
    y = [0] * (len(c) + 2)
    for i in range(2, len(y)):
        y[i] = c[i - 2] - y[i - 2] + 2 * x * y[i - 1]
    return y[-1] - x * y[-2]
