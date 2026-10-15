import numpy as np

EPS = 1e-9   # tolerance: anything smaller in magnitude is treated as 0


def forward_elimination(A, b):
    """Gaussian elimination with partial pivoting.
    Pivot selection and row swapping are written by hand (no np.argmax / fancy indexing).
    Prints every pivot chosen and every row swap performed."""
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)

    for k in range(n - 1):
        # ---- hand-written pivot selection: largest |entry| in column k, rows k..n-1 ----
        p = k
        for i in range(k + 1, n):
            if abs(A[i, k]) > abs(A[p, k]):
                p = i
        print(f"  column {k}: pivot chosen = {A[p, k]:g} (from row {p})")

        # ---- hand-written row swap ----
        if p != k:
            print(f"  row swap: R{k} <-> R{p}")
            for j in range(n):
                A[k, j], A[p, j] = A[p, j], A[k, j]
            b[k], b[p] = b[p], b[k]

        # if the whole column (at and below k) is zero, there is nothing to eliminate
        if abs(A[k, k]) < EPS:
            print(f"  pivot in column {k} is 0 -> skip elimination for this column")
            continue

        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            b[i]     -= m * b[k]

    return A, b


def back_substitution(U, c):
    n = len(c)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (c[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def classify_and_solve(A, b, name):
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print("Augmented matrix before elimination:\n", np.column_stack([A, b]), "\n")

    U, c = forward_elimination(A, b)
    n = len(c)

    print("\nAugmented matrix after forward elimination:\n",
          np.column_stack([U, c]))

    # rows of the form  0.x + 0.y + 0.z = c
    zero_rows = [i for i in range(n) if np.all(np.abs(U[i]) < EPS)]

    # ---------- NO SOLUTION:  0.x + 0.y + 0.z = c,  c != 0 ----------
    inconsistent = [i for i in zero_rows if abs(c[i]) > EPS]
    if inconsistent:
        print("\n>>> NO SOLUTION (inconsistent system)")
        for i in inconsistent:
            print(f"    row {i} reads:  0*x1 + 0*x2 + 0*x3 = {c[i]:g}   (impossible)")
        return

    # ---------- INFINITE SOLUTIONS:  0.x + 0.y + 0.z = 0 ----------
    if zero_rows:
        print("\n>>> INFINITE SOLUTIONS")
        for i in zero_rows:
            print(f"    row {i} reads:  0*x1 + 0*x2 + 0*x3 = 0")
        # choose the free variable x3 at will (hardcoded x3 = 5)
        x = np.zeros(n)
        x[n - 1] = 5.0
        print(f"    choosing free variable x3 = {x[n - 1]:g}")
        # back-substitute through the non-zero rows (bottom-up)
        nonzero_rows = [i for i in range(n) if i not in zero_rows]
        for i in reversed(nonzero_rows):
            x[i] = (c[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
        print("    one particular solution:", x)
        print("    check ||Ax - b||_2 =", np.linalg.norm(A @ x - b))
        return

    # ---------- UNIQUE SOLUTION ----------
    x = back_substitution(U, c)
    print("\n>>> UNIQUE SOLUTION")
    print("    x (our solver)      =", x)
    x_np = np.linalg.solve(A.astype(float), b.astype(float))
    print("    np.linalg.solve     =", x_np)
    print("    ||Ax - b||_2        =", np.linalg.norm(A @ x - b))


# ----------------------------------------------------------------------
# Three 3-variable systems (one of each kind)
# ----------------------------------------------------------------------

# System 1: unique solution  (answer: x = [2, 3, -1])
A1 = np.array([[ 2,  1, -1],
               [-3, -1,  2],
               [-2,  1,  2]], dtype=float)
b1 = np.array([8, -11, -3], dtype=float)

# System 2: no solution  (row2 = 2*row1 but RHS is inconsistent)
A2 = np.array([[1, 1, 1],
               [2, 2, 2],
               [1, 2, 3]], dtype=float)
b2 = np.array([1, 3, 4], dtype=float)

# System 3: infinite solutions  (row2 = 2*row1 and RHS is consistent)
A3 = np.array([[1, 1, 1],
               [2, 2, 2],
               [1, 2, 3]], dtype=float)
b3 = np.array([6, 12, 10], dtype=float)

classify_and_solve(A1, b1, "System 1")
classify_and_solve(A2, b2, "System 2")
classify_and_solve(A3, b3, "System 3")


# ----------------------------------------------------------------------
# Written question:
# "If a diagonal entry becomes 0, is it necessary that this situation
#  results in no solution? In which case will it be no solution /
#  infinite solutions exactly?"
#
# Answer:
#   No, a zero diagonal (pivot) entry does NOT necessarily mean no solution.
#
#   Case 1: some row BELOW still has a nonzero entry in that column.
#       Partial pivoting simply swaps that row up and elimination
#       continues. The system can still have a perfectly unique
#       solution -- the zero pivot was just an unlucky row ordering.
#
#   Case 2: the pivot AND every entry below it in that column are zero.
#       Then the matrix is singular (rank < n) and there is no unique
#       solution. Which of the two remaining cases occurs depends on
#       the right-hand side of the resulting zero rows:
#         - a row  0*x1 + 0*x2 + 0*x3 = c  with c != 0
#               -> contradiction -> NO SOLUTION
#         - every zero row has  0*x1 + 0*x2 + 0*x3 = 0
#               -> that equation is always true -> one variable is
#                  free -> INFINITE SOLUTIONS
# ----------------------------------------------------------------------
