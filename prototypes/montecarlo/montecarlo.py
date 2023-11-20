import numpy as np
from seemps.cross import Mesh, RegularClosedInterval


def integrate_montecarlo(func, mesh, samples=int(1e6)):
    """
    Perform Monte Carlo integration to estimate the integral of a multivariate function.

    Parameters:
        func (callable): The function to be integrated, accepting an array of points as input.
        mesh (Mesh): The mesh representing the integration domain.
        samples (int, optional): The number of random samples to generate for the Monte Carlo integration.

    Returns:
        float: The estimated integral value.

    Example:
        >>> # Define the function to be integrated and mesh defining the integration domain
        >>> def my_function(x, y):
        ...     return x**2 + y**2
        >>> mesh = Mesh([...])  # Replace with your Mesh data
        >>> # Estimate the integral using Monte Carlo integration
        >>> estimated_integral = integrate_montecarlo(my_function, mesh, samples=10000)
        >>> print(estimated_integral)
        5.0  # Replace with the actual result

    """
    random_points = np.random.uniform(
        low=[interval.start for interval in mesh.intervals],
        high=[interval.stop for interval in mesh.intervals],
        size=(samples, mesh.dimension),
    )
    func_values = func(*random_points.T)
    volume = np.prod([interval.stop - interval.start for interval in mesh.intervals])
    return np.mean(func_values) * volume


if __name__ == "__main__":
    a = -1.0
    b = 1.0
    func = lambda x, y: x + y
    mesh = Mesh([RegularClosedInterval(a, b, 2) for _ in range(2)])
    intg_exact = 0.5 * (b**2 - a**2) * 2
    intg_mc = integrate_montecarlo(func, mesh)
    print(intg_exact, intg_mc, intg_exact - intg_mc)
