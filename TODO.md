# TODO LIST

## High priority
- Finish the basic implementation of the `analysis` module, for example the Python functions, the plotting routines or the sequential loop function. DONE
- Implement the `analysis` module in a branch of my fork of SeeMPS2 with all the required algorithms. DONE
- Implement the simulations for the Chebyshev and Cross algorithms for one-dimensional functions. DONE
- Benchmark the Chebyshev, Cross and SVD algorithms for 1d functions and get some plots. DONE
- Analyze the unexpected results for the Chebyshev algorithm. DONE
    - Develop the Chebyshev vector method.
    - Compare its convergences with the MPS version and find out why it does not converge to 10^-16.

## Medium priority
- Optimize the `analysis` module:
    - Implement the Cython functions.
    - Implement the parallel loop.
    - Develop the cluster scripts and start sending jobs.
- Do some secondary analysis of the algorithms:
    - Entanglement entropy of the MPS.
    - Comparison against MonteCarlo integration.

- Benchmark the Chebyshev, Cross and SVD algorithms for multivariate functions.
- Develop the Cross-based global optimization method proposed by Juanjo.

## Low priority
- Develop prototypes (Padé, SPC, CrossDMRG, extrapolation, circuit method...).
- Further optimize the libraries.
    - JAX / PyTorch port for GPU.
    - C++ ports. 