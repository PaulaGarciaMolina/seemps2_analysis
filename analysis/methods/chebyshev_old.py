import numpy as np
import h5py
import pathlib
from typing import List, Union, Optional, Tuple, Callable

from seemps.tools import log
from seemps.hdf5 import read_mps, write_mps
from seemps.state import MPS, Strategy, DEFAULT_TOLERANCE
from seemps.cross import Mesh

from .factories import mps_empty, mps_identity, mps_position


DATA_PATH = str(pathlib.Path(__file__).parent.absolute()) + "/data/"


class Chebyshev:
    """Chebyshev class.

    This implements a Chebyshev object which encodes a multivariate function
    together with a multidimensional domain into a Matrix Product State (MPS)
    by means of a multivariate truncated Chebyshev series with $d_i$ terms
    for each dimension.

    Parameters
    ----------
    func : Callable
        A multidimensional **vector-valued** function to be encoded in MPS form.
    mesh : Mesh
        A multidimensional discretized mesh on which the function is defined.
    orders : list
        A list of Chebyshev truncation orders to be assigned to each of the dimensions of the function.
    """

    def __init__(self, func: Callable, mesh: Mesh, orders: List[int]):
        self.func = func
        self.mesh = mesh
        self.orders = orders

    @staticmethod
    def cheb_at_zeros(d: int, i: Union[int, np.ndarray]) -> np.ndarray:
        """Returns a vector of length d with the evaluations of a Chebyshev
        polynomial of order i in the zeros of a Chebyshev polynomial of order d."""
        return np.array(
            [np.cos(np.pi * i * (2 * k - 1) / (2 * d)) for k in range(1, d + 1)]
        )

    @staticmethod
    def zeros(d: int) -> np.ndarray:
        """Returns a vector of length d with the zeros of an order-d Chebyshev polynomial."""
        return Chebyshev.cheb_at_zeros(d, i=1)

    @staticmethod
    def zeros_matrix(d: int, hdf5_path=None) -> np.ndarray:
        """Returns a (d x d) matrix with the evaluations of the first d Chebyshev
        polynomials evaluated at the zeros of a Chebyshev polynomial of order d."""
        name = f"zeros_matrix_d{d}"
        if hdf5_path is None:
            hdf5_path = DATA_PATH + "zeros_matrix.hdf5"
        try:
            with h5py.File(hdf5_path, "r") as file:
                zeros_matrix = file[name][:]
        except:
            zeros_matrix = Chebyshev.cheb_at_zeros(d, np.arange(d))
            zeros_matrix[:, 0] *= 0.5
            with h5py.File(hdf5_path, "a") as file:
                file.create_dataset(name, data=zeros_matrix)
        return zeros_matrix

    @staticmethod
    def cheb_mps(
        d: int,
        n: int,
        mesh_type: Optional[str] = "o",
        mps0: Optional[MPS] = None,  # TODO: Implement
        hdf5_path: Optional[str] = None,
    ) -> Tuple[MPS, MPS]:
        """Returns the MPS of an order-d Chebyshev polynomial with n qubits
        and 2**N points distributed according to a parameter mesh.

        Parameters
        ----------
        d : int
            Order of the Chebyshev polynomial.
        n : int
            Number of qubits of the MPS.
        mesh : str
            Type of mesh used:
            'o' -> regular mesh with open boundary conditions.
            'c' -> regular mesh with closed boundary conditions.
            'z' -> irregular mesh along the Chebyshev zeros.
        mps0 : MPS, optional
            Basis MPS on which to expand, resulting in function composition.
            If mps0 is the representation of a function g(a, b), then the MPS for the resulting
            Chebyshev polynomial T corresponds to T(g(a, b)).
            By default, given by the MPS for the position function x(a, b).
        """
        if mesh_type not in ["o", "c", "z"]:
            raise ValueError("The mesh_type is not valid or has not been implemented.")
        if hdf5_path is None:
            hdf5_path = DATA_PATH + "cheb_mps.hdf5"
        name_Ti = f"cheb_mps_d{d}_n{n}_{mesh_type}"
        name_Tj = f"cheb_mps_d{d-1}_n{n}_{mesh_type}"
        try:
            with h5py.File(hdf5_path, "r") as file:
                Ti = read_mps(file, name_Ti)
                Tj = read_mps(file, name_Tj) if d > 0 else mps_identity(n)
        except:
            if mps0 is None:
                mps0 = mps_position(-1, 1, n, mesh_type)
            if d == 0:
                Ti = mps_identity(n)
                Tj = mps_identity(n)
            elif d == 1:
                Ti = mps0
                Tj = mps_identity(n)
            else:
                Tj, Tk = Chebyshev.cheb_mps(d - 1, n, mesh_type, mps0, hdf5_path)
                Ti = (2.0 * mps0.wavefunction_product(Tj) - Tk).toMPS(
                    strategy=Strategy(tolerance=DEFAULT_TOLERANCE)
                )
            with h5py.File(hdf5_path, "a") as file:
                write_mps(file, name_Ti, Ti)
        return Ti, Tj

    def func_tensor(self) -> np.ndarray:
        """Returns the evaluation of the function on the tensor of all the Chebyshev
        zeros of the multidimensional domain, performing a linear transformation on
        them to change their domain from [-1, 1] to [a, b].
        """
        alist = self.func.mesh.alist
        blist = self.func.mesh.blist
        cheb_mesh = Mesh(alist, blist, self.orders, mesh_type="z", use_qubits=False)
        cheb_tensor = cheb_mesh.to_tensor()
        func_tensor = np.apply_along_axis(self.func, -1, cheb_tensor)
        return func_tensor

    def coef_tensor(self, hdf5_path: Optional[str] = None) -> np.ndarray:
        """Returns the tensor of Chebyshev coefficients of the function by
        contracting the tensor of evaluations of the function at the Chebyshev
        zeros with the Chebyshev matrices of size (d x d)."""
        func_tensor = self.func_tensor()
        prefactor = (2 ** len(self.orders)) / (np.prod(self.orders))
        coef_tensor = prefactor * np.flip(func_tensor)
        for idx, d in enumerate(self.orders):
            zeros_matrix = self.zeros_matrix(d, hdf5_path=hdf5_path)
            coef_tensor = np.swapaxes(coef_tensor, 0, idx)
            coef_tensor = np.einsum(
                "i..., ik... -> k...", coef_tensor, zeros_matrix
            )  # DO WITH MATMUL
            coef_tensor = np.swapaxes(coef_tensor, idx, 0)
        return coef_tensor

    @staticmethod
    def diff_coef_tensor(coef_tensor: np.ndarray, m: int) -> np.ndarray:
        """Returns the derivative along dimension m of the tensor of Chebyshev coefficients."""
        shape = coef_tensor.shape
        d = shape[m]
        diff_coef_shape = shape[:m] + (d + 1,) + shape[m + 1 :]
        diff_coef_tensor = np.zeros(diff_coef_shape)

        def take_coef(i):
            return np.take(coef_tensor, i, axis=m)

        for i in range(d - 2, -1, -1):
            diff_coef_tensor[put_at(i, axis=m)] = 2 * (i + 1) * take_coef(
                i + 1
            ) + np.take(diff_coef_tensor, i + 2, axis=m)
        diff_coef_tensor = np.take(diff_coef_tensor, range(d - 1), axis=m)
        return diff_coef_tensor

    @staticmethod
    def intg_coef_tensor(
        coef_tensor: np.ndarray, m: int, c_int: Optional[float] = None
    ):
        """Returns the integral along dimension m of the tensor of Chebyshev coefficients."""
        shape = coef_tensor.shape
        d = shape[m]
        intg_shape = shape[:m] + (d - 2,) + shape[m + 1 :]
        intg_coef_tensor = np.zeros(intg_shape)

        def take_coef(i):
            return np.take(coef_tensor, i, axis=m)

        intg_coef_tensor[put_at(1, axis=m)] = (2 * take_coef(0) - take_coef(2)) / 2
        for i in range(2, d - 2):
            intg_coef_tensor[put_at(i, axis=m)] = (
                take_coef(i - 1) - take_coef(i + 1)
            ) / (2 * i)
        if c_int is None:  # Constant of integration
            c_int = take_coef(0)  # Equivalent to c=0
        intg_coef_tensor[put_at(0, axis=m)] = c_int
        return intg_coef_tensor

    def run(self, coef_tensor: Optional[np.ndarray] = None):
        """Returns the MPS representation of the truncated Chebyshev series
        of the function by performing the summation of the tensor of Chebyshev
        coefficients with the Chebyshev' MPS.

        Parameters
        ----------
        coef_tensor : numpy array, optional
            Tensor of Chebyshev coefficients with which to perform the summation.
            Can be provided externally to account for differentiation and integration
            using recurrence relations.
            If not provided, computes and uses the tensor of coefficients of the
            function.
        """
        if self.func.mesh.use_qubits is False:
            raise ValueError("The mesh needs to use qubits.")
        qubits = self.func.mesh.nodes
        if coef_tensor is None:
            coef_tensor = self.coef_tensor()
        func_mps = mps_empty(np.sum(qubits))
        mesh_type = self.func.mesh.mesh_type
        for d_combination in np.ndindex(coef_tensor.shape):
            log("Computing Chebyshev MPS term of order", d_combination)
            mps_combination = [
                Chebyshev.cheb_mps(d, qubits[idx], mesh_type, self.mps0)[0]
                for idx, d in enumerate(d_combination)
            ]
            mps = wavefunction_cartesian_product(
                mps_combination, mlist=list(range(len(d_combination)))
            )
            func_mps = (func_mps + coef_tensor[d_combination] * mps).toMPS(
                strategy=self.strategy
            )

        return func_mps


def wavefunction_cartesian_product(ψlist: List[MPS], mlist: List[int]) -> MPS:
    """
    Multiplies element-wise a list of MPS ψlist with respect to the dimensions given by mlist.
    """
    nlist = [len(ψ) for ψ in ψlist]
    if mlist is None:
        mlist = [0] * len(ψlist)

    # Extend the first MPS and multiply the rest against it
    m = mlist[0]
    n = nlist[0]
    ψ = ψlist[0].extend(np.sum(nlist), list(range(m * n, (m + 1) * n)))
    for idx, φ in enumerate(ψlist[1:]):
        m = mlist[idx + 1]
        φ_extended = φ.extend(len(ψ), list(range(m * len(φ), (m + 1) * len(φ))))
        ψ = ψ.wavefunction_product(φ_extended)
    return ψ


def put_at(inds, axis=-1):
    """Auxiliary function to create an index tuple for numpy array indexing.

    Parameters
    ----------
        inds (int or slice or tuple of ints and/or slices): Indices to be placed at the specified axis.
        axis (int, optional): The axis where the indices are to be placed. Default is the last axis (-1).

    Returns
    ----------
        tuple: Index tuple suitable for numpy array indexing.
    """
    if isinstance(inds, (int, slice)):
        inds = (inds,)
    index = [slice(None)] * abs(axis) + list(inds) + [slice(None)] * (axis - 1)
    return tuple(index)
