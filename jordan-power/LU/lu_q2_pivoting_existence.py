import numpy as np


# ------------------------------------------------------------------------
# Task 1: run the Q1 (no-pivoting) Doolittle LU on A1 and see it break
# ------------------------------------------------------------------------

def doolittle_lu_naive(A, verbose=True):
    """Same code as Q1 -- no pivoting. Included here to demonstrate the
    failure on A1 (A1[0][0] = 0)."""
    A = A.astype(float).copy()
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()

    for k in range(n - 1):
        for i in range(k + 1, n):
            m = U[i, k] / U[k, k]          # <-- divides by U[k,k]
            L[i, k] = m
            U[i, k:] -= m * U[k, k:]
        if verbose:
            print(f"  after column {k}: U =\n{U}")
    return L, U


A1 = np.array([[0, 2, 1],
               [1, 1, 1],
               [2, 1, 3]], dtype=float)

print("=" * 60)
print("Task 1: naive (no-pivot) Doolittle LU on A1")
print("=" * 60)
print("A1 =\n", A1)
print("\nA1[0][0] = 0  ->  running the Q1 code:\n")
import warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    L1_broken, U1_broken = doolittle_lu_naive(A1)
    for w in caught:
        print(f"  [numpy warning] {w.message}")

print("\nresulting (garbage) U:\n", U1_broken)
print("resulting (garbage) L:\n", L1_broken)
print("""
WHAT HAPPENS AND WHY:
  At k=0, the pivot U[0,0] = 0. The multiplier
      m = U[1,0] / U[0,0] = 1 / 0
  is a division by zero. NumPy does not raise an exception for this --
  it silently produces +inf (with a RuntimeWarning), so L[1,0] becomes
  inf instead of crashing outright.

  The corruption then spreads on the very next line:
      U[i, k:] -= m * U[k, k:]
  Here m = inf is multiplied against U[0,0:] = [0, 2, 1]. The first
  entry is  inf * 0 = nan  (infinity times zero is a genuinely
  indeterminate quantity in IEEE floating point -- this is the
  "quantity that becomes undefined"). From that point on nan/inf
  propagate through every remaining row and column: the factorization
  is garbage, not just "a little off". No exception is thrown, so a
  program that doesn't explicitly check for this would silently return
  nonsense.
""")


# ------------------------------------------------------------------------
# Tasks 2-4: PA = LU with partial pivoting (hand-written)
# ------------------------------------------------------------------------

def plu_decomposition(A, verbose=True):
    """Computes P, L, U such that P @ A = L @ U, using hand-written
    partial pivoting. Prints every pivot chosen and every swap performed."""
    A = A.astype(float).copy()
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)

    for k in range(n - 1):
        # ---- hand-written pivot selection: largest |entry| in column k ----
        p = k
        for i in range(k + 1, n):
            if abs(U[i, k]) > abs(U[p, k]):
                p = i
        if verbose:
            print(f"  column {k}: pivot = {U[p, k]:g}  (row {p})")

        # ---- hand-written row swap (U, P, AND the already-built part of L) ----
        if p != k:
            if verbose:
                print(f"  swap: R{k} <-> R{p}")
            U[[k, p]] = U[[p, k]]
            P[[k, p]] = P[[p, k]]
            # CRITICAL: multipliers already stored in L for columns < k
            # belong to specific ROWS, and those rows just moved -- the
            # multipliers must move with them, or L will describe the
            # wrong elimination history.
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


A2 = np.array([[1, 2, 3],
               [2, 4, 7],
               [3, 5, 3]], dtype=float)

for name, A in [("A1", A1), ("A2", A2)]:
    print("=" * 60)
    print(f"Task 2-4: PA = LU with partial pivoting on {name}")
    print("=" * 60)
    print(f"{name} =\n", A)
    P, L, U = plu_decomposition(A)
    print("\nP =\n", P)
    print("L =\n", np.round(L, 4))
    print("U =\n", np.round(U, 4))
    err = np.linalg.norm(P @ A - L @ U, 'fro')
    print(f"\n||P@{name} - L@U||_F = {err}")


# ------------------------------------------------------------------------
# Task 5: solve A1 x = [5, 3, 6] using P, L, U
# ------------------------------------------------------------------------

print("=" * 60)
print("Task 5: solve A1 x = [5, 3, 6] via P, L, U")
print("=" * 60)

b = np.array([5, 3, 6], dtype=float)
P, L, U = plu_decomposition(A1, verbose=False)

Pb = P @ b                       # NOTE: solve Lz = P@b, NOT Lz = b
print("P@b =", Pb)
z = forward_substitution(L, Pb)
print("z (Lz = P@b) =", z)
x = backward_substitution(U, z)
print("x (Ux = z)   =", x)

print("\n=== Verification ===")
x_np = np.linalg.solve(A1, b)
print("np.linalg.solve :", x_np)
print("||A1@x - b||_2  :", np.linalg.norm(A1 @ x - b))


# ------------------------------------------------------------------------
# Written question:
# "Does every invertible matrix have an LU decomposition without
#  pivoting? State the exact condition for existence, and give a 2x2
#  counterexample. Does adding partial pivoting fix existence for every
#  invertible matrix?"
#
# Answer:
#   No. A = LU (no pivoting, L unit lower triangular) exists AND is
#   unique if and only if every LEADING PRINCIPAL MINOR of A is nonzero:
#       det(A[0:1, 0:1]) != 0
#       det(A[0:2, 0:2]) != 0
#       ...
#       det(A[0:n-1, 0:n-1]) != 0
#   (the full det(A) != 0, i.e. A invertible, is not itself sufficient --
#   it's the leading SUBMATRICES that must each be nonsingular, since
#   elimination needs a nonzero pivot at every one of the first n-1 steps).
#
#   2x2 counterexample:
#       A = [[0, 1],
#            [1, 0]]
#   det(A) = -1, so A is invertible. But its leading 1x1 minor is
#   det([0]) = 0, so there is NO way to write A = LU with L unit lower
#   triangular and U upper triangular (the first pivot would have to be
#   0, and no legal row operation is allowed to fix a "pivot", only
#   pivoting/row-swapping can).
#
#   Does partial pivoting fix existence for every invertible matrix?
#   Yes. For ANY invertible (in fact any square) matrix A, there exists
#   a permutation matrix P such that PA = LU exists with L unit lower
#   triangular and U upper triangular. Partial pivoting's row-swap rule
#   (always promote the largest-|entry| row into the pivot position) is
#   exactly the constructive procedure that finds such a P: it guarantees
#   a nonzero pivot is available at every step whenever A is nonsingular
#   (if an entire column below and including the diagonal were zero, A
#   would be singular). So PA = LU always exists for invertible A -- only
#   the pivot-free A = LU form can fail to exist.
# ------------------------------------------------------------------------
