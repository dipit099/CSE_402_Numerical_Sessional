import numpy as np

def gauss_naive(A, b, verbose=True):
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)

    # Forward elimination
    for k in range(n - 1):
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]          # multiplier
            A[i, k:] -= m * A[k, k:]
            b[i]      -= m * b[k]
    if verbose:
        print("Upper triangular A:\n", A)
        print("Transformed b:", b)

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i+1:] @ x[i+1:]) / A[i, i]
    return x


t = np.array([5, 8, 12])
v = np.array([106.8, 177.2, 279.2])
A = np.column_stack([t**2, t, np.ones_like(t)])   # [t^2  t  1]

x = gauss_naive(A, v)
print("a1, a2, a3 =", x)

# sanity check
print("numpy check:", np.linalg.solve(A, v))