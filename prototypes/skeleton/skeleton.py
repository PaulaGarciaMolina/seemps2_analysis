"""
Practice for the matrix cross-interpolation algorithm and the max-vol algorithm.
Follows the 'How to Find a Good Submatrix' paper
"""

import numpy as np
from numpy.linalg import inv, det, qr

np.random.seed(42)

import matplotlib.pyplot as plt


def maxvol(A, δ=1.0e-2, verbose=True):
    """
    Computes the maximal-volume sobmatrix of a matrix A with tolerance δ
    """
    # Assign first rows of A to random nonsingular submatrix
    A0 = A
    n = A.shape[0]
    r = A.shape[1]
    rows = np.random.choice(n - 1, r, replace=False)
    for idx, row in enumerate(rows):
        A[[idx, row], :] = A[[row, idx], :]

    i = 0
    while True:
        As = A[0:r]
        vol = abs(det(As))
        B = A @ inv(As)
        b = max(B.min(), B.max(), key=abs)
        if verbose:
            print(f"i = {i} | Vol = {vol} | abs(b) = {abs(b)}")
        # Check if the submatrix is dominant
        if abs(b) < (1 + δ):
            # TODO: Figure out how to get the indices directly from the algorithm
            idx = []
            for row in As:
                idx.append(np.where(np.all(np.isclose(A0, row), axis=1))[0][0])
            return idx

        bpos = np.where(B == b)
        B[[bpos[0][0], bpos[1][0]], :] = B[[bpos[1][0], bpos[0][0]], :]  # Swap rows of B
        A = B @ As  # Recompute A and start over
        i += 1


def skeleton(A, δ=1.0e-2, χ=5, iter=5):
    """
    Computes the skeleton decomposition of a matrix A with tolerance δ
    """

    columns = np.random.choice(A.shape[1], χ, replace=False)

    for _ in range(iter):
        rows = maxvol(A[columns, :].T, δ)
        columns = maxvol(A[:, rows], δ)

    C = A[:, columns]
    As = A[rows, :][:, columns]
    R = A[rows, :]

    return C, As, R


def skeleton_osedelets(A, δ=1.0e-2, χ=5, iter=5):
    """
    Computes the skeleton decomposition of a matrix A with tolerance δ
    Follows Algorithm 2 of 'Osedelets - TT-cross approximation for multidimensional arrays'
    """

    columns = np.random.choice(A.shape[1], χ, replace=False)

    for k in range(iter):
        # Row cycle
        R = A[:, columns]
        Q, _ = qr(R)
        # Quasi-maximal volume submatrix
        rows = maxvol(Q, δ, verbose=False)
        # Column cycle
        C = A[rows, :].T
        Q, _ = qr(C)
        # Quasi-maximal volume submatrix
        columns = maxvol(Q, δ, verbose=False)
        # New approximation
        Qhat = Q[columns, :]
        Aprev = A
        A = A[:, columns] @ (Q @ inv(Qhat)).T
        frobdist = np.max(np.abs(A - Aprev))
        print(f"k: {k} | Error: {frobdist}")

    C = A[:, columns]
    As = A[rows, :][:, columns]
    R = A[rows, :]

    return C, As, R


if __name__ == "__main__":
    # Define initial nxr matrix A
    # TODO: Ver como el algoritmo puede aceptar matrices con una de las dimensiones exponencialmente grandes
    # (para permitir el row-column alternating algorithm)
    # n = int(1.E10); r = 5

    # region Test 1: Maxvol algorithm for tall matrices
    # n = 100; r = 5
    # A = np.random.rand(n, r)
    # δ = 1.E-2
    # As, idx = maxvol(A, δ)
    # endregion

    # region Test 2: Skeleton decomposition for arbitrary matrices
    def plot_surface(X, Y, Z1, Z2=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z1)
        if Z2 is not None:
            ax.plot_surface(X, Y, Z2)
        plt.show()

    a = -1
    b = 1
    m = 100
    n = 100

    x = np.linspace(a, b, m)
    y = np.linspace(a, b, n)
    X, Y = np.meshgrid(x, y)
    A = np.exp(X * Y)  # Solo funciona para esta función
    # A = np.random.rand(m, n)

    δ = 1.0e-4
    C, As, R = skeleton(A, δ, iter=20, χ=5)
    A_approx = C @ inv(As) @ R
    print(A - A_approx)
    plot_surface(X, Y, A, A_approx)
    # endregion
