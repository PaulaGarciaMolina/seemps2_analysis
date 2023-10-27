from unittest import TestCase
from analysis.functions import piecewise


class TestFunctions(TestCase):
    def test_cython_speedup(self):
        pass

    def test_gaussian_product(self):
        pass

    def test_gaussian_squeezed(self):
        pass

    def test_filter_piecewise(self):
        linear = lambda x: 2 * x
        quadratic = lambda x: x**2
        x_cusps = [1.0, 2.0]
        funcs = [linear, quadratic, linear]
        piecewise_func = piecewise(funcs, x_cusps)
        self.assertEqual(piecewise_func(0.5), 1)
        self.assertEqual(piecewise_func(1), 1)
        self.assertEqual(piecewise_func(1.5), 2.25)
        self.assertEqual(piecewise_func(2), 4)
        self.assertEqual(piecewise_func(2.5), 5)

    def test_norm_distances(self):
        pass
