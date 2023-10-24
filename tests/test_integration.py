import numpy as np
from unittest import TestCase

from analysis.methods import *


class TestIntegration(TestCase):
    def test_quadrature_from_vector(self):
        """Test that the quadratures MPS loaded from vectors have small error."""
        pass

    def test_integral_error_with_vectors(self):
        """Compare the integration errors of MPS with default tolerance and with vectors."""
        pass

    def test_convergence_midpoint(self):
        """Test the asymptotic convergence of the midpoint quadrature."""
        pass

    def test_convergence_trapezoidal(self):
        """Test the asymptotic convergence of the trapezoidal quadrature."""
        pass

    def test_convergence_simpson(self):
        """Test the asymptotic convergence of the Simpson quadrature."""
        pass

    def test_convergence_fifth_order(self):
        """Test the asymptotic convergence of the fifth order quadrature."""
        pass

    def test_convergence_fejer(self):
        """Test the asymptotic convergence of the Féjer quadrature."""
        pass

    def test_norm_distances(self):
        pass
