import numpy as np
import matplotlib.pyplot as plt

from analysis.methods.chebyshev_vector import _coef_tensor, mps_chebyshev
from analysis.methods.factories_mps import mps_position
from analysis.methods.vector import vector_chebyshev, clenshaw_1d
from analysis.methods import chebyshev_expand, chebyshev_compose
from analysis.functions import rectifier

from seemps.state import MPS
from seemps.cross import Mesh, RegularHalfOpenInterval

from .tools import TestCase


# Hay 3 posibles fuentes de error:
# 1. Las constantes
#    -> Comparar constantes con constantes analíticas para órdenes grandes d > 100.
#    Posible solución: usar método alternativo como FFT o integración.
# 2. La construcción de los MPS de Chebyshev.
#    -> Comparar MPS y vector para órdenes grandes d > 100.
#    Posible solución: usar método alternativo de evaluación (interpolación baricéntrica o Clenshaw).
# 3. La evaluación de la serie de Chebyshev.
#    -> Si las otras dos fuentes no son, tiene que ser esta.
#    Posible solución: usar método alternativo de evaluación (interpolación baricéntrica o Clenshaw).


def func_setup(func, m, a=-1, b=1, n=5):
    mesh = Mesh([RegularHalfOpenInterval(a, b, 2**n) for _ in range(m)])
    mesh_tensor = mesh.to_tensor()
    func_vector = func(*mesh_tensor.T).flatten()
    # np.apply_along_axis(func, -1, mesh_tensor).flatten()
    mps = MPS.from_vector(func_vector, [2] * (n * m), normalize=False)
    return func, mesh, mps, func_vector


class TestChebyshev(TestCase):
    def test_coef_vector(self):
        exponential = lambda x: np.exp(x)
        func, mesh, _, _ = func_setup(exponential, 1)
        coef_exp = [
            1.266065877752008,
            1.130318207984970,
            0.271495339534077,
            0.044336849848664,
            0.005474240442094,
            0.000542926311914,
            0.000044977322954,
            0.000003198436462,
            0.000000199212481,
            0.000000011036772,
            0.000000000550590,
            0.000000000024980,
            0.000000000001039,
            0.000000000000040,
            0.000000000000001,
        ]
        coef_vector = _coef_tensor(func, mesh, [15])
        # Small absolute errors BUT LARGE RELATIVE ERRORS!
        # This may be the issue. Test: use the correct constants in the integration of the exponential.
        # Plot the errors
        assert np.allclose(coef_exp, coef_vector)

    def test_mps_chebyshev(self):
        sites = 10
        order = 12
        interval = RegularHalfOpenInterval(-1.0, 1.0, 2**sites)
        name = f"mps_chebyshev-type_open-sites_{sites}"
        cheb_mps = mps_chebyshev(mps_position(interval), order, name=name)
        cheb_func = lambda order, x: np.cos(order * np.arccos(x))
        cheb_vector = cheb_func(order, interval.to_vector())
        self.assertSimilar(cheb_vector, cheb_mps.to_vector())

    def test_chebyshev_expand_1d(self):
        gaussian = lambda x: np.exp(-(x**2))
        func, mesh, _, func_vector = func_setup(gaussian, 1)
        mps = chebyshev_expand(func, mesh, [20])
        self.assertSimilar(func_vector, mps.to_vector())

    def test_chebyshev_expand_2d(self):
        gaussian = lambda x, y: np.exp(-(x**2 + y**2))
        func, mesh, _, func_vector = func_setup(gaussian, 2)
        mps = chebyshev_expand(func, mesh, [20, 20])
        self.assertSimilar(func_vector, mps.to_vector())

    def test_chebyshev_compose_mps(self):
        gaussian = lambda vec: np.exp(-np.sum(vec**2))
        func, mesh, mps_0, func_vector = func_setup(gaussian, 1, a=-2, b=2)
        x = mesh.to_tensor().flatten()
        filter = rectifier(0.1, cutoff="bottom")
        mps = chebyshev_compose(filter, mps_0, 50, -1.0, 1.0)
        plt.plot(x, func_vector, label="func(x)")
        plt.plot(x, filter(x), ".", label="filter(x)")
        plt.plot(x, filter(func_vector), label="filter(func(x))")
        plt.plot(x, mps.to_vector(), label="ChebMPS filter(func(x))")
        plt.legend()
        plt.show()

    def test_chebyshev_compose_mpo(self):
        pass


class TestChebyshevVector(TestCase):
    def test_vector_with_recurrence(self):
        sites = 10
        order = 10
        interval = RegularHalfOpenInterval(-1.0, 1.0, 2**sites)
        x = interval.to_vector()
        cheb_func = lambda x: np.cos(order * np.arccos(x))
        vector_exact = cheb_func(x)
        vector_recurrent = vector_chebyshev(sites, order)
        assert np.allclose(vector_exact, vector_recurrent)

    def test_vector_with_clenshaw(self):
        pass

    def test_vector_expansion_1d(self):
        pass

    def test_vector_expansion_2d(self):
        pass

    def test_vector_filter(self):
        pass
