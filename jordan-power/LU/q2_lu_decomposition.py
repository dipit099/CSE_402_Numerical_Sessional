import numpy as np


def plu_decomposition(A):
    """Computes P, L, U such that PA = LU (partial pivoting)."""
    A = A.astype(float).copy()
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)

    for k in range(n - 1):
        # partial pivoting: row with largest |entry| in column k, at/below k
        p = k
        for i in range(k + 1, n):
            if abs(U[i, k]) > abs(U[p, k]):
                p = i

        if p != k:
            U[[k, p]] = U[[p, k]]
            P[[k, p]] = P[[p, k]]
            # already-computed multipliers must move with their row too
            if k > 0:
                L[[k, p], :k] = L[[p, k], :k]

        for i in range(k + 1, n):
            m = U[i, k] / U[k, k]
            L[i, k] = m
            U[i, k:] -= m * U[k, k:]

    return P, L, U


def forward_substitution(L, b):
    """Solves Lz = b for lower triangular L (unit diagonal)."""
    n = len(b)
    z = np.zeros(n)
    for i in range(n):
        z[i] = (b[i] - L[i, :i] @ z[:i]) / L[i, i]
    return z


def backward_substitution(U, z):
    """Solves Ux = z for upper triangular U."""
    n = len(z)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (z[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def solve_system_lu(A, b):
    """Solves Ax = b using PA = LU decomposition."""
    P, L, U = plu_decomposition(A)

    print("\nP (Permutation Matrix):")
    print(np.round(P, 4))
    print("\nL (Lower Triangular Matrix):")
    print(np.round(L, 4))
    print("\nU (Upper Triangular Matrix):")
    print(np.round(U, 4))

    # PAx = Pb -> LUx = Pb
    Pb = np.dot(P, b)
    print(f"\nPermuted b (Pb): {np.round(Pb, 4)}")

    # Solve Lz = Pb
    z = forward_substitution(L, Pb)
    print(f"\nz (Lz = Pb): {np.round(z, 4)}")

    # Solve Ux = z
    x = backward_substitution(U, z)
    print(f"\nx (Ux = z): {np.round(x, 4)}")

    return x


if __name__ == "__main__":
    A = np.array([[2, 1, 1],
                  [4, 3, 3],
                  [8, 7, 9]], dtype=float)
    b = np.array([4, 10, 24], dtype=float)

    x = solve_system_lu(A, b)

    print("\n=== Verification ===")
    x_np = np.linalg.solve(A, b)
    print("np.linalg.solve :", np.round(x_np, 4))
    print("||Ax - b||_2    :", np.linalg.norm(A @ x - b))
