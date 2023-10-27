import numpy as np
import matplotlib.pyplot as plt

from analysis.methods.chebyshev import _coef_tensor, mps_chebyshev
from analysis.methods.factories import mps_position
from analysis.methods import chebyshev_expand, chebyshev_filter
from analysis.functions import step

from seemps.state import MPS
from seemps.cross import Mesh, RegularHalfOpenInterval

from .tools import TestCase


def exponential_setup(m):
    a = -1
    b = 1
    n = 6
    func = lambda vec: vec[0]  # np.exp(vec[0])
    mesh = Mesh([RegularHalfOpenInterval(a, b, 2**n) for _ in range(m)])
    mesh_tensor = mesh.to_tensor()
    func_vector = np.apply_along_axis(func, -1, mesh_tensor).flatten()
    mps = MPS.from_vector(func_vector, [2] * (n * m), normalize=False)
    return func, mesh, mps, func_vector


class TestChebyshev(TestCase):
    def test_coef_vector(self):
        func, mesh, _, _ = exponential_setup(1)
        coef_exp = [
            1.266065877752008,
            1.130318207984970,
            0.271495339534077,
            0.044336849848664,
            0.005474240442094,
        ]
        coef_vector = _coef_tensor(func, mesh, [10])
        assert np.allclose(coef_exp, coef_vector[:5])

    def test_mps_chebyshev(self):
        interval = RegularHalfOpenInterval(-1.0, 1.0, 2**5)
        cheb_mps = mps_chebyshev(mps_position(interval), 4)
        cheb_func = lambda x: 8 * x**4 - 8 * x**2 + 1
        cheb_vector = cheb_func(interval.to_vector())
        self.assertSimilar(cheb_vector, cheb_mps.to_vector())

    def test_expansion_1d(self):
        func, mesh, _, func_vector = exponential_setup(1)
        mps = chebyshev_expand(func, mesh, [10])
        self.assertSimilar(func_vector, mps.to_vector())

    def test_expansion_2d(self):
        func, mesh, _, func_vector = exponential_setup(2)
        mps = chebyshev_expand(func, mesh, [10, 10])
        self.assertSimilar(func_vector, mps.to_vector())

    def test_filter_mps(self):
        func, mesh, mps_0, func_vector = exponential_setup(1)
        x = mesh.to_tensor()
        filter = step(0)
        mps = chebyshev_filter(mps_0, filter, -2.0, 2.0, 10)
        plt.plot(x, func_vector, label="Function")
        plt.plot(x, filter(x), label="Filter")
        plt.plot(x, mps.to_vector(), "-o", label="Filtered MPS")
        plt.legend()
        plt.show()

    def test_filter_mpo(self):
        pass


class TestChebyshevVector(TestCase):
    def test_vector_expansion_1d(self):
        pass

    def test_vector_expansion_2d(self):
        pass

    def test_vector_filter(self):
        pass
