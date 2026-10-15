"""
Lab Assignment 2 (A1/B1/C1)
Solution of Systems of Equations & Eigenvalue Decomposition
-------------------------------------------------------------
Covers:
 1. Gauss Elimination (without partial pivoting -- naive)
 2. Gauss Elimination (with partial pivoting -> fixes divide-by-zero
    and reduces round-off error)
 3. Gauss-Jordan Elimination (without partial pivoting -- naive)
 4. Gauss-Jordan Elimination (with partial pivoting)
 5. LU Decomposition (Doolittle's method)
 6. Matrix Inverse using LU Decomposition
 7. Power Method            -> largest eigenvalue/eigenvector
 8. Inverse Power Method    -> smallest eigenvalue/eigenvector
 9. Deflation               -> intermediate eigenvalues/eigenvectors
10. Full Eigenvalue Decomposition (A = P D P^-1)
11. Coordinate transform between standard basis and eigenbasis
12. Computing A^100 using the eigen decomposition

NOTE: numpy is used only as an array container (indexing, +, -, *, @).
None of numpy's linear-algebra routines (np.linalg.solve / inv / eig / norm)
are used -- every algorithm below is implemented from first principles.
"""

import numpy as np


# ===========================================================
# Small helper functions
# ===========================================================

def vec_norm(v):
    """Euclidean norm or L2 norm of a vector, computed manually (no np.linalg.norm)."""
    s = 0.0
    for x in v:
        s += x * x
    return s ** 0.5
    

def normalize_vector(v):
    summation = 0.0
    print("normalize vector")
    # v = np.array([3,4,5],dtype=float)
    size= np.shape(v)
   
    print("==== Size ", size)
    for i in range(size[0]):
        summation+=v[i]*v[i]
    summation=summation**0.5
    v=v/summation
    print("v vector",v)
    


# print(help(np.linalg.eig))






def identity(n):
    I = np.zeros((n, n))
    for i in range(n):
        I[i][i] = 1.0
    return I


def dominant_entry(v):
    """
    Index and value of the largest-MAGNITUDE entry of v (manual, no
    np.argmax). This is the quantity the power method normalizes by --
    and that value itself is the eigenvalue estimate at each iteration.
    COnsiders absolutvr value during comparing
    """
    idx = 0
    best = abs(v[0])
    for i in range(1, len(v)):
        if abs(v[i]) > best:
            best = abs(v[i])
            idx = i
    return idx, v[idx]


# ===========================================================
# 1. GAUSS ELIMINATION (without partial pivoting -- naive)
# ===========================================================
def gauss_elimination_naive(A, b):
    """
    Solve Ax = b using plain (naive) Gauss elimination -- no pivoting.

    Same forward-elimination / back-substitution formulas as
    gauss_elimination below, but the pivot at each step k is simply
    A[k][k], whatever it happens to be. This is the version that
    divides by zero if a pivot is (or becomes) zero, and that can lose
    accuracy to round-off if a pivot is very small -- partial pivoting
    (see gauss_elimination) is the fix for both problems.
    """
    A = A.astype(float).copy() #typecast to float must else int dhore copy korbe by defult
    b = b.astype(float).copy()
    n = len(b)

    # ---- Forward elimination (no pivoting) ----
     # eliminate entries below the pivot
    for k in range(n - 1):
        for i in range(k + 1, n):
            factor = A[i][k] / A[k][k]
            A[i, k:] -= factor * A[k, k:] # all elements in row i considering column k as pivot
            b[i] -= factor * b[k]

    # ---- Back substitution ----
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = b[i]
        for j in range(i + 1, n):
            s -= A[i][j] * x[j]
        x[i] = s / A[i][i]

    return x


# ===========================================================
# 2. GAUSS ELIMINATION (with partial pivoting)
# ===========================================================
def gauss_elimination(A, b):
    """
    Solve Ax = b using Gauss elimination.

    Partial pivoting: at every step, swap in the row with the largest
    |value| in the current column before eliminating. This is the fix
    for the "divide by zero" problem (a zero pivot) and also reduces
    round-off error (dividing by a very small pivot blows up errors).
    """
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)

    # ---- Forward elimination ----
    for k in range(n - 1):
        # find best pivot row in column k
        pivot_row = k
        pivot_val = abs(A[k][k])
        for i in range(k + 1, n):
            if abs(A[i][k]) > pivot_val:
                pivot_val = abs(A[i][k])
                pivot_row = i
        if pivot_row != k: # we swap rows only when the best pivot row is not already the current row.
            A[[k, pivot_row]] = A[[pivot_row, k]]
            b[[k, pivot_row]] = b[[pivot_row, k]]

        # eliminate entries below the pivot
        for i in range(k + 1, n):
            factor = A[i][k] / A[k][k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # ---- Back substitution ----
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = b[i]
        for j in range(i + 1, n):
            s -= A[i][j] * x[j]
        x[i] = s / A[i][i]

    return x


# ===========================================================
# 3. GAUSS-JORDAN ELIMINATION (without partial pivoting -- naive)
# ===========================================================
def gauss_jordan_naive(A, b):
    """
    Solve Ax = b using Gauss-Jordan elimination -- no pivoting.

    Same normalize-then-eliminate-everywhere process as gauss_jordan
    below, but the pivot at each step k is simply A[k][k], whatever it
    happens to be -- so it inherits the same divide-by-zero /
    round-off pitfalls as gauss_elimination_naive.
    """
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)

    for k in range(n):
        # normalize pivot row so pivot becomes 1 (no pivoting)
        pivot = A[k][k]
        A[k, :] = A[k, :] / pivot
        b[k] = b[k] / pivot

        # eliminate this column from EVERY other row (above and below)
        for i in range(n):
            if i != k:
                factor = A[i][k]
                A[i, :] -= factor * A[k, :]
                b[i] -= factor * b[k]

    return b  # b has become the solution vector x!!!!!!


# ===========================================================
# 4. GAUSS-JORDAN ELIMINATION (with partial pivoting)
# ===========================================================
def gauss_jordan(A, b):
    """
    Solve Ax = b using Gauss-Jordan elimination.
    Reduces [A|b] all the way to [I|x] -- no separate back-substitution step.
    """
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)

    # FIRST EXCHANGE ROWS THEN NORMALIZE RULE
    for k in range(n):
        # partial pivoting (same reasoning as gauss_elimination)
        pivot_row = k
        pivot_val = abs(A[k][k])
        for i in range(k + 1, n):
            if abs(A[i][k]) > pivot_val: # only consider the nicher rows for pivoting 
                pivot_val = abs(A[i][k])
                pivot_row = i
        if pivot_row != k:
            A[[k, pivot_row]] = A[[pivot_row, k]]
            b[[k, pivot_row]] = b[[pivot_row, k]]

        # normalize pivot row so pivot becomes 1
        pivot = A[k][k]
        A[k, :] = A[k, :] / pivot
        b[k] = b[k] / pivot

        # eliminate this column from EVERY other row (above and below)
        for i in range(n):
            if i != k:
                factor = A[i][k]
                A[i, :] -= factor * A[k, :]
                b[i] -= factor * b[k]

    return b  # b has become the solution vector x


# ===========================================================
# 5. LU DECOMPOSITION (Doolittle's method)
# ===========================================================

def lu_decomposition(A):
    # no "b" HEREE
    """
    LU decomposition by performing Gaussian elimination.

    L stores the elimination multipliers.
    U is obtained by eliminating entries below each pivot.

    No pivoting (assumes nonzero pivots).
    """
    A = A.astype(float).copy()
    n = A.shape[0] # no of rows

    U = A.copy()
    L = np.eye(n)

    # Gaussian elimination
    for k in range(n - 1):

        # Eliminate entries below pivot
        for i in range(k + 1, n):

            # Same multiplier as Gaussian elimination
            factor = U[i, k] / U[k, k]

            # Store it instead of forgetting it
            L[i, k] = factor

            # Perform the row operation
            U[i, k:] -= factor * U[k, k:]




    return L, U


def forward_substitution(L, b):
    """Solve Ly = b, where L is lower triangular."""
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        s = b[i]
        for j in range(i):
            s -= L[i][j] * y[j]
        y[i] = s / L[i][i]
    return y


def backward_substitution(U, y):
    """Solve Ux = y, where U is upper triangular."""
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= U[i][j] * x[j]
        x[i] = s / U[i][i]
    return x


def solve_using_lu(A, b):
    """Solve Ax = b by first decomposing A = LU, then two substitutions."""
    L, U = lu_decomposition(A)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x


# ===========================================================
# 6. MATRIX INVERSE USING LU DECOMPOSITION
# ===========================================================
def inverse_using_lu(A):
    """
    A^-1 is built column by column: solve A x_i = e_i for every
    standard basis vector e_i, using the SAME L,U (computed once).
    """
    n = A.shape[0]
    L, U = lu_decomposition(A)
    Ainv = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        y = forward_substitution(L, e)
        x = backward_substitution(U, y)
        Ainv[:, i] = x
    return Ainv


# ===========================================================
# 7. POWER METHOD -> largest eigenvalue / eigenvector
# ===========================================================
def power_method(A, num_iter=500, tol=1e-12):
    """
    Start with any nonzero vector x. Each iteration:
        y = A @ x
        eigval_estimate = the largest-MAGNITUDE entry of y (taken BEFORE
                           normalizing -- this is the eigenvalue estimate,
                           not a Rayleigh quotient)
        x = y / eigval_estimate    (normalize by that entry)
    Repeat until the eigenvalue estimate stops changing (relative error).
    """
    n = A.shape[0]
    # NOTE on choosing the starting vector:
    # A vector like [1,1,1,...] can accidentally be EXACTLY ORTHOGONAL to the
    # eigenvector we're trying to find (this happens for the deflated matrix
    # in section 8 below) -- the iteration then has nothing of that
    # eigen-direction to amplify and converges to the WRONG eigenvalue.
    # Using varying entries [1,2,3,...] makes this essentially never happen.
    # (relevant once we deflate the matrix in section 9 below)
    x = np.arange(1, n + 1, dtype=float) # random initial vector
    eigval = 0.0

    for _ in range(num_iter):
        y = A @ x # matrix multiplication sign
        _, eigval_new = dominant_entry(y)
        x_new = y / eigval_new

        if eigval_new != 0 and abs((eigval_new - eigval) / eigval_new) < tol:
            x, eigval = x_new, eigval_new
            break
        x, eigval = x_new, eigval_new

    return eigval, x


# ===========================================================
# 8. INVERSE POWER METHOD -> smallest eigenvalue / eigenvector
# ===========================================================
def inverse_power_method(A, num_iter=500, tol=1e-12):
    """
    Power method applied implicitly to A^-1 (since 1/lambda is an
    eigenvalue of A^-1 whenever lambda is an eigenvalue of A). A is
    factored into L, U ONCE; every iteration solves A y = x via cheap
    forward/back substitution instead of forming A^-1 explicitly.
    """
    n = A.shape[0]
    L, U = lu_decomposition(A)
    x = np.arange(1, n + 1, dtype=float)
    eigval_inv = 0.0

    for _ in range(num_iter):
        z = forward_substitution(L, x)
        y = backward_substitution(U, z)       # solves A y = x
        _, eigval_inv_new = dominant_entry(y)
        x_new = y / eigval_inv_new

        if eigval_inv_new != 0 and abs((eigval_inv_new - eigval_inv) / eigval_inv_new) < tol:
            x, eigval_inv = x_new, eigval_inv_new
            break
        x, eigval_inv = x_new, eigval_inv_new

    eigval = 1.0 / eigval_inv
    return eigval, x


# ===========================================================
# 9. DEFLATION -> intermediate eigenvalues / eigenvectors
# ===========================================================
def deflate(A, eigval, eigvec):
    """
    Remove a known eigenvalue/eigenvector pair from a SYMMETRIC matrix A
    so the power method can be reapplied to find the next eigenvalue.
        A_deflated = A - eigval * (v v^T)      (v normalized to unit length)
    """
    v = eigvec / vec_norm(eigvec)
    v_col = v.reshape(-1,1)
    v_row = v.reshape(1,-1)
    #))
    print("changed vector ", A - eigval *(v_col @ v_row) )
    return A - eigval *(v_col @ v_row)


def find_all_eigen_symmetric(A, num_eigenvalues=None, num_iter=500, tol=1e-12):
    """
    Find eigenvalues/eigenvectors of a symmetric matrix by repeatedly
    running power_method then deflating out the eigenvalue just found.
    Returns them ordered largest -> smallest.

    num_eigenvalues: how many to find (default: all n). Since deflation
    only "mutes" the eigenvalues already found, stopping early still
    gives exact intermediate eigenpairs -- you don't have to peel off
    all n if you only need the first few.
    """
    n = A.shape[0]
    if num_eigenvalues is None:
        num_eigenvalues = n

    A_work = A.copy()
    eigvals = []
    eigvecs = []

    for _ in range(num_eigenvalues):
        val, vec = power_method(A_work, num_iter, tol)
        eigvals.append(val)
        eigvecs.append(vec)
        A_work = deflate(A_work, val, vec)

    return np.array(eigvals), np.array(eigvecs).T   # eigenvectors as columns


# ===========================================================
# 10. FULL EIGEN DECOMPOSITION  (A = P D P^-1)
# ===========================================================
def eigen_decomposition(A, num_eigenvalues=None):
    """
    P : matrix whose columns are the eigenvectors of A
    D : diagonal matrix of the corresponding eigenvalues
    such that A = P @ D @ P^-1

    num_eigenvalues: how many eigenpairs to include (default: all n).
    Pass fewer than n to get just the dominant few -- P will then be
    n x num_eigenvalues (rectangular) and A = P@D@P^-1 no longer holds,
    since P isn't invertible; use the full default when you need the
    complete reconstruction.
    """
    eigvals, P = find_all_eigen_symmetric(A, num_eigenvalues)
    n = len(eigvals)
    D = np.zeros((n, n))
    for i in range(n):
        D[i][i] = eigvals[i]
    return P, D


# ===========================================================
# 11. COORDINATE TRANSFORM: standard basis <-> eigenbasis
# ===========================================================
def standard_to_eigen_coords(P, x):
    """x is in the standard basis -> return its eigenbasis coordinates c = P^-1 x"""
    P_inv = inverse_using_lu(P)
    return P_inv @ x


def eigen_to_standard_coords(P, c):
    """c is in the eigenbasis -> return the standard-basis vector x = P c"""
    return P @ c


# ===========================================================
# 12. A^100 USING THE EIGEN DECOMPOSITION
# ===========================================================
def matrix_power_via_eigen(A, power):
    """
    A^power = P D^power P^-1
    (raising a diagonal matrix to a power = raising each entry to that power,
    so this avoids ever multiplying A by itself 100 times)
    """
    P, D = eigen_decomposition(A)
    n = D.shape[0]
    D_power = np.zeros((n, n))
    for i in range(n):
        D_power[i][i] = D[i][i] ** power
    P_inv = inverse_using_lu(P)
    return P @ D_power @ P_inv


# ===========================================================
# MAIN -- exercises every function above
# ===========================================================
def main():
    np.set_printoptions(precision=4, suppress=True)

    print("=" * 65)
    print("1. GAUSS ELIMINATION  (naive, no pivoting)")
    print("=" * 65)

    A0 = np.array([[4, 1, 0, 0],
                   [1, 3, 1, 0],
                   [0, 1, 2, 1],
                   [0,0,1, 1]],)
    #b0 = np.array([0,0,1,1], dtype=float).  # it represent (4,) array
    #x0 = gauss_elimination_naive(A0, b0)
    # print("A =\n", A0)
    # print("b =", b0)
    # print("Solution x =", x0)
    #print("Check A@x  =", A0 @ x0, " (should equal b)\n")

    # print("=" * 65)
    # print("2. GAUSS ELIMINATION  (partial pivoting handles a zero pivot)")
    # print("=" * 65)
    # A1 = np.array([[0, 2, 1],
    #                [1, 1, 4],
    #                [3, -1, 1]], dtype=float)
    # b1 = np.array([7, 15, 4], dtype=float)
    # x1 = gauss_elimination(A1, b1)
    # print("A =\n", A1)
    # print("b =", b1)
    # print("Solution x =", x1)
    # print("Check A@x  =", A1 @ x1, " (should equal b)\n")

    # print("=" * 65)
    # print("3. GAUSS-JORDAN ELIMINATION  (naive, no pivoting)")
    # print("=" * 65)
    # x2n = gauss_jordan_naive(A0, b0)
    # print("Solution x =", x2n)
    # print("Check A@x  =", A0 @ x2n, "\n")

    # print("=" * 65)
    # print("4. GAUSS-JORDAN ELIMINATION  (with partial pivoting)")
    # print("=" * 65)
    # x2 = gauss_jordan(A1, b1)
    # print("Solution x =", x2)
    # print("Check A@x  =", A1 @ x2, "\n")

    # print("=" * 65)
    # print("5. LU DECOMPOSITION")
    # print("=" * 65)
    # A3 = np.array([[4, 3, 2],
    #                [2, 3, 1],
    #                [1, 1, 5]], dtype=float)
    # b3 = np.array([9, 6, 7], dtype=float)
    # L, U = lu_decomposition(A3)
    # print("L =\n", L)
    # print("U =\n", U)
    # print("Check L@U  =\n", L @ U, " (should equal A3)\n")
    # x3 = solve_using_lu(A3, b3)
    # print("Solve A3 x = b3 using LU -> x =", x3)
    # print("Check A3@x =", A3 @ x3, "\n")

    # print("=" * 65)
    # print("6. MATRIX INVERSE USING LU")
    # print("=" * 65)
    # A0_inv = inverse_using_lu(A0)
    # print("A3^-1 =\n", A0_inv)
    # print("Check A0 @ A0^-1 =\n", A0 @ A0_inv, " (should be identity)\n")

    # Symmetric matrix used for all eigen-related demos
    # Asym = np.array([[2, -1, 0],
    #                   [-1, 2, -1],
    #                   [0, -1, 2]], dtype=float)

    print("=" * 65)
    print("7. POWER METHOD -> largest eigenvalue/eigenvector")
    print("=" * 65)
    lambda_max, v_max = power_method(A0)
    print("A =\n", A0)
    print("Largest eigenvalue  =", lambda_max)
    print("Eigenvector         =", v_max)
    print("Check A@v vs lambda*v:", A0 @ v_max, "|", lambda_max * v_max, "\n")

    # print("=" * 65)
    # print("8. INVERSE POWER METHOD -> smallest eigenvalue/eigenvector")
    # print("=" * 65)
    # lambda_min, v_min = inverse_power_method(A0)
    # print("Smallest eigenvalue =", lambda_min)
    # print("Eigenvector         =", v_min)
    # print("Check A@v vs lambda*v:", A0 @ v_min, "|", lambda_min * v_min, "\n")


    print("=" * 65)
    print("8. INVERSE POWER METHOD -> smallest eigenvalue/eigenvector")   
    A0_inv = inverse_using_lu(A0)
    print("A3^-1 =\n", A0_inv)
    print("Check A0 @ A0^-1 =\n", A0 @ A0_inv, " (should be identity)\n")
    lambda_min, v_min = power_method(A0_inv)
    print("Smallest eigenvalue =", 1/lambda_min)  # 1/lamda must
    print("Eigenvector         =", v_min)
    v_min = normalize_vector(v_min)  # normalize the eigenvector
    print("Normalized Eigenvector         =", v_min)

    # print("HEREE")
    # print( np.linalg.inv(A0))

    # NumPy eigenvalue/eigenvector computation
    print("=" * 65)
    print("NumPy eigenvalue/eigenvector computation")
    eigvals, eigvecs = np.linalg.eig(A0)
    print(np.linalg.eig(A0))
    indx = np.argmin(eigvals)
    min_val = eigvals[indx]
    transpose_eigvecs = np.transpose(eigvecs)
    min_vector = transpose_eigvecs[indx]
    print("Eigenvalues ", min_val)
    print("Eigenvectors ", min_vector)
    


    print("=" * 65)
    print("9. DEFLATION -> intermediate eigenvalue/eigenvector")
    print("=" * 65)

    A_deflated = deflate(A0, lambda_max, v_max)
    lambda_mid, v_mid = power_method(A_deflated)
    print("After removing the largest eigenvalue, next eigenvalue =", lambda_mid)
    print("Eigenvector         =", v_mid)
    print("Check A@v vs lambda*v:", A0 @ v_mid, "|", lambda_mid * v_mid, "\n")

    # print("=" * 65)
    # print("10. FULL EIGEN DECOMPOSITION  (A = P D P^-1)")
    # print("=" * 65)
    # P, D = eigen_decomposition(Asym)
    # print("Eigenvalues (diag of D) =", np.diag(D))
    # print("P (eigenvectors as columns) =\n", P)
    # P_inv = inverse_using_lu(P)
    # A_rebuilt = P @ D @ P_inv
    # print("Rebuilt A = P@D@P^-1 =\n", A_rebuilt)
    # print("Original A           =\n", Asym, "\n")

    # print("=" * 65)
    # print("11. COORDINATE TRANSFORM: standard <-> eigenbasis")
    # print("=" * 65)
    # x_std = np.array([1.0, 2.0, 3.0])
    # c_eigen = standard_to_eigen_coords(P, x_std)
    # x_back = eigen_to_standard_coords(P, c_eigen)
    # print("x (standard coords)   =", x_std)
    # print("c (eigenbasis coords) =", c_eigen)
    # print("x reconstructed       =", x_back, "\n")

    print("=" * 65)
    print("12. A^100 USING EIGEN DECOMPOSITION")
    print("=" * 65)
    A_100 = matrix_power_via_eigen(A0, 100)
    print("A^100 =\n", A_100)


if __name__ == "__main__":
    main()


    # lamda inverse o kora lage !! in Inverse Power Method