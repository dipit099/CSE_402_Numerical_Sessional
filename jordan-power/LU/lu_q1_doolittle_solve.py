import numpy as np


def doolittle_lu(A, verbose=True):
    """Doolittle LU decomposition (L has 1s on the diagonal), hand-written.
    Prints L and U after each column is fully processed."""
    A = A.astype(float).copy()
    n = A.shape[0]

    L = np.eye(n)
    U = A.copy()

    for k in range(n - 1):
        for i in range(k + 1, n):
            m = U[i, k] / U[k, k]      # multiplier, hand-computed
            L[i, k] = m                 # hand-stored into L
            U[i, k:] -= m * U[k, k:]

        if verbose:
            print(f"\n--- after processing column {k} ---")
            print("L =\n", np.round(L, 4))
            print("U =\n", np.round(U, 4))

    return L, U


def forward_substitution(L, b):
    """Solves Lz = b (L unit lower triangular)."""
    n = len(b)
    z = np.zeros(n)
    for i in range(n):
        z[i] = (b[i] - L[i, :i] @ z[:i]) / L[i, i]
    return z


def backward_substitution(U, z):
    """Solves Ux = z (U upper triangular)."""
    n = len(z)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (z[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


if __name__ == "__main__":
    A = np.array([[4,  3,  2,  1],
                  [8, 11,  6,  4],
                  [4,  9, 11,  7],
                  [12, 15, 16, 20]], dtype=float)
    b = np.array([10, 29, 31, 63], dtype=float)

    print("=== Doolittle LU decomposition ===")
    L, U = doolittle_lu(A)

    print("\n=== Factorization check ===")
    fro_err = np.linalg.norm(A - L @ U, 'fro')
    print("||A - L@U||_F =", fro_err)

    print("\n=== Solve Ax = b via Lz = b, then Ux = z ===")
    z = forward_substitution(L, b)
    print("z =", np.round(z, 4))
    x = backward_substitution(U, z)
    print("x =", np.round(x, 4))

    print("\n=== Verification ===")
    x_np = np.linalg.solve(A, b)
    print("np.linalg.solve :", np.round(x_np, 4))
    print("||Ax - b||_2    :", np.linalg.norm(A @ x - b))


# ----------------------------------------------------------------------
# Written question:
# "You need to solve Ax = b for 500 different vectors b (same A). Explain
#  why LU is the right tool and compare the total cost with running full
#  Gaussian elimination 500 times. (Give the O(.) counts.)"
#
# Answer:
#   LU factorization depends only on A, never on b. So factor ONCE:
#       cost of factoring an n x n matrix  ~ O(n^3)   (~ 2n^3/3 flops)
#
#   Each new b then costs only two triangular solves:
#       forward substitution (Lz=b)  ~ O(n^2)
#       backward substitution (Ux=z) ~ O(n^2)
#       -> ~2n^2 flops per b
#
#   For k = 500 right-hand sides:
#       LU total      = O(n^3)          +  k * O(n^2)
#                      = O(n^3 + k n^2)
#
#   Naive approach -- rerunning full Gaussian elimination (which redoes the
#   O(n^3) elimination work on A AND b jointly) for every b:
#       naive total   = k * O(n^3) = O(k n^3)
#
#   Concretely for this problem (n = 4, k = 500):
#       LU factor        : ~ (2/3)*4^3            ≈   43 flops   (once)
#       500 solves       : ~ 500 * 2*4^2          ≈ 16,000 flops
#       LU total         :                        ≈ 16,043 flops
#
#       naive (500 full eliminations, each ~2n^3/3 + 2n^2)
#                         : ~ 500 * (43 + 32)      ≈ 37,500 flops
#
#   Even at this tiny n the LU approach is already cheaper, and the gap
#   widens fast as n grows: naive scales as k*n^3, LU scales as n^3 + k*n^2.
#   Once k > n (500 > 4 here, overwhelmingly), the k*n^2 term is what
#   dominates the LU cost, and it is a full factor of n cheaper per solve
#   than repeating elimination from scratch. This is exactly why LU is
#   the standard tool whenever the SAME matrix must be solved against many
#   different right-hand sides (e.g. simulations, repeated load cases).
# ----------------------------------------------------------------------
