import numpy as np
from numpy.linalg import inv, det
from timeit import default_timer as timer

np.random.seed(42)


def maxvol(A, δ=1.0e-2):
    n = A.shape[0]
    r = A.shape[1]

    # Assign first rows of A to random nonsingular submatrix
    random_rows = np.random.choice(n - 1, r, replace=False)  # r random row indices
    remaining_rows = list(set(range(n)) - set(random_rows))
    rows = np.concatenate((random_rows, remaining_rows))
    A = A[rows]

    i = 0
    while True:
        As = A[0:r]
        vol = abs(det(As))
        B = A @ inv(As)
        b = max(B.min(), B.max(), key=abs)
        print(f"i = {i} | Vol = {vol} | abs(b) = {abs(b)}")
        if abs(b) < (1 + δ):  # Check if the submatrix is dominant
            return rows[:r]

        b_i, b_j = np.argwhere(B == b)[0]
        B[[b_i, b_j]] = B[[b_j, b_i]]  # Swap rows of B
        rows[[b_i, b_j]] = rows[[b_j, b_i]]
        A = B @ As  # Recompute A and start over
        i += 1


def maxvol_refined(A, δ=0.01):
    n = A.shape[0]
    r = A.shape[1]

    # Assign first rows of A to random nonsingular submatrix
    random_rows = np.random.choice(n - 1, r, replace=False)  # r random row indices
    remaining_rows = list(set(range(n)) - set(random_rows))
    rows = np.concatenate((random_rows, remaining_rows))
    A = A[rows]

    # Initialization
    As = A[0:r]
    B = A @ inv(As)
    b = max(B.min(), B.max(), key=abs)

    def e(n, k):
        e = np.zeros(n)
        e[k] = 1
        return e

    i = 0
    while abs(b) > (1 + δ):
        b_i, b_j = np.argwhere(B == b)[0]
        B -= np.outer(B[:, b_j] - e(n, b_j) + e(n, b_i), (B[b_i] - e(r, b_j)) / b)
        # Figure out how to do it for Z = B[r:]
        b = max(B.min(), B.max(), key=abs)
        print(f"i = {i} | abs(b) = {abs(b)}")
        i += 1

    return rows[:r]


if __name__ == "__main__":
    # region Test 1: Maxvol algorithm for tall matrices
    n = 100
    r = 5
    A = np.random.rand(n, r)
    δ = 1.0e-2
    idx = maxvol(A, δ)
    # endregion

    # region Test 2: Refined maxvol algorithm for tall matrices
    n = 100
    r = 5
    A = np.random.rand(n, r)
    δ = 1.0e-2
    idx = maxvol_refined(A, δ)
    # endregion

    # region Test 3: Benchmark standard maxvol against refined maxvol
    n = int(1e5)
    r = 10
    A = np.random.rand(n, r)
    δ = 1.0e-2

    tick1 = timer()
    idx_std = maxvol(A, δ)
    tick2 = timer()
    idx_ref = maxvol_refined(A, δ)
    tick3 = timer()

    t_std = tick2 - tick1
    print("Standard maxvol: ", t_std)
    t_ref = tick3 - tick2
    print("Refined maxvol: ", t_ref)
    speedup = (t_std / t_ref - 1) * 100
    print("Speedup: ", speedup, "%")
    # For small matrices (n,r) there is a speedup but gets slower for large n with the intersection at around (n, r) = (1E5, 10).
    # There is an asymptotic bottleneck with n somewhere in maxvol_refined.

    # endregion
