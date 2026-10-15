import numpy as np


def plu_decomposition(A):
    """Computes P, L, U such that PA = LU (partial pivoting)."""
    A = A.astype(float).copy()
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)

    for k in range(n - 1):
        p = k
        for i in range(k + 1, n):
            if abs(U[i, k]) > abs(U[p, k]):
                p = i

        if p != k:
            U[[k, p]] = U[[p, k]]
            P[[k, p]] = P[[p, k]]
            if k > 0:
                L[[k, p], :k] = L[[p, k], :k]

        for i in range(k + 1, n):
            m = U[i, k] / U[k, k]
            L[i, k] = m
            U[i, k:] -= m * U[k, k:]

    return P, L, U


def forward_substitution(L, b):
    n = len(b)
    z = np.zeros(n)
    for i in range(n):
        z[i] = (b[i] - L[i, :i] @ z[:i]) / L[i, i]
    return z


def backward_substitution(U, z):
    n = len(z)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (z[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def lu_inverse(A):
    """Computes A^-1 by solving A x_i = e_i for every column e_i of the
    identity, reusing a single PA = LU factorization."""
    n = A.shape[0]
    P, L, U = plu_decomposition(A)

    Ainv = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        Pe = P @ e                       # PAx = Pe -> LUx = Pe
        z = forward_substitution(L, Pe)
        x = backward_substitution(U, z)
        Ainv[:, i] = x                   # x is column i of A^-1

    return Ainv, P, L, U


if __name__ == "__main__":
    A = np.array([[2, 1, 1],
                  [4, 3, 3],
                  [8, 7, 9]], dtype=float)

    Ainv, P, L, U = lu_inverse(A)

    print("P (Permutation Matrix):\n", np.round(P, 4))
    print("\nL (Lower Triangular Matrix):\n", np.round(L, 4))
    print("\nU (Upper Triangular Matrix):\n", np.round(U, 4))
    print("\nA inverse:\n", np.round(Ainv, 4))

    print("\n=== Verification ===")
    print("np.linalg.inv(A):\n", np.round(np.linalg.inv(A), 4))
    print("||A @ Ainv - I||_F =",
          np.linalg.norm(A @ Ainv - np.eye(A.shape[0])))
