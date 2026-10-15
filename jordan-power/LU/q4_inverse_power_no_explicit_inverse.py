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


def apply_A_inverse(P, L, U, x):
    """Computes A^-1 @ x via two triangular solves -- A^-1 is never formed."""
    Pb = P @ x
    z = forward_substitution(L, Pb)     # Lz = Pb
    y = backward_substitution(U, z)     # Uy = z   ->   y = A^-1 x
    return y


def inverse_power_method(A, x0, tol=1e-8, max_iter=500, verbose=True):
    P, L, U = plu_decomposition(A)      # factor ONCE, reuse every iteration

    x = x0.astype(float).copy()
    eigen_estimates = []                # estimates of the DOMINANT eigenvalue of A^-1

    for it in range(max_iter):
        y = apply_A_inverse(P, L, U, x)
        eigen_estimated = y[np.argmax(np.abs(y))]   # largest-magnitude entry (sign kept)
        eigen_estimates.append(eigen_estimated)
        x_new = y / eigen_estimated

        if verbose:
            print(f"  iter {it + 1:3d}:  eig(A^-1) est = {eigen_estimated:10.6f}"
                  f"   ->  eig(A) est = {1 / eigen_estimated:10.6f}")

        if np.max(np.abs(x_new - x)) < tol:
            x = x_new
            break
        x = x_new

    x = x / np.linalg.norm(x)
    lam_min = 1.0 / eigen_estimates[-1]     # convert: eigenvalue of A^-1 -> eigenvalue of A
    return lam_min, x, eigen_estimates


if __name__ == "__main__":
    A = np.array([[8, 2, 0, 0],
                  [2, 8, 0, 0],
                  [0, 0, 3, 1],
                  [0, 0, 1, 3]], dtype=float)
    # NOTE: x0 = [1,1,1,1] (the "textbook" starting guess) has ZERO component
    # along the eigenvalue-2 eigenvector for this particular A -- see the
    # eigenbasis check below. A generic asymmetric x0 avoids that blind spot.
    x0 = np.array([1, 2, 3, -1], dtype=float)

    print("=== Inverse power method (A^-1 never formed, only L,U + 2 solves/iter) ===")
    lam_min, v, history = inverse_power_method(A, x0)

    print("\nsmallest eigenvalue :", lam_min)
    print("eigenvector         :", v)
    print("iterations          :", len(history))

    print("\n=== NumPy verification ===")
    eigvals, eigvecs = np.linalg.eig(A)
    k = np.argmin(np.abs(eigvals))          # index of the smallest |eigenvalue|
    lam_np = eigvals[k]
    v_np = eigvecs[:, k]
    print("smallest eigenvalue :", lam_np)
    print("eigenvector         :", v_np)

    if np.dot(v, v_np) < 0:                 # v and -v are equally valid
        v_np = -v_np
        print("(sign flipped for comparison)")

    print("\neigenvalue difference      :", abs(lam_min - lam_np))
    print("eigenvector difference     :", np.linalg.norm(v - v_np))
    print("residual ||Av - lam*v||_2  :", np.linalg.norm(A @ v - lam_min * v))
