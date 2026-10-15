import numpy as np


def plu_decomposition(A):
    """Computes P, L, U such that PA = LU (partial pivoting).
    Also returns the number of row swaps performed, needed for the
    determinant's sign."""
    A = A.astype(float).copy()
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)
    swap_count = 0

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
            swap_count += 1

        for i in range(k + 1, n):
            m = U[i, k] / U[k, k]
            L[i, k] = m
            U[i, k:] -= m * U[k, k:]

    return P, L, U, swap_count


def lu_determinant(A):
    """det(A) = (-1)^(#row swaps) * product(diagonal of U).
    Derivation: PA = LU -> det(P)*det(A) = det(L)*det(U).
    det(L) = 1 (unit diagonal); det(P) = (-1)^(#swaps); and det(P) is its
    own inverse (+-1), so det(A) = (-1)^(#swaps) * det(U)."""
    P, L, U, swaps = plu_decomposition(A)
    sign = (-1) ** swaps
    det = sign
    for i in range(U.shape[0]):
        det *= U[i, i]
    return det, P, L, U, swaps


if __name__ == "__main__":
    A = np.array([[2, 1, 1],
                  [4, 3, 3],
                  [8, 7, 9]], dtype=float)

    det, P, L, U, swaps = lu_determinant(A)

    print("P (Permutation Matrix):\n", np.round(P, 4))
    print("\nL (Lower Triangular Matrix):\n", np.round(L, 4))
    print("\nU (Upper Triangular Matrix):\n", np.round(U, 4))
    print(f"\nnumber of row swaps : {swaps}")
    print(f"diagonal of U       : {np.round(np.diag(U), 4)}")
    print(f"det(A)              : {det}")

    print("\n=== Verification ===")
    print("np.linalg.det(A) =", np.linalg.det(A))
