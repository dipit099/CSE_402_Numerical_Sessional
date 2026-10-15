import numpy as np

def gauss_partial_pivot(A, b):
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)

    # forword substitution
    for k in range(n - 1):
        # find the row with the largest |value| in column k, at/below row k
        p = np.argmax(np.abs(A[k:, k])) + k
        if p != k:
            A[[k, p]] = A[[p, k]]
            b[[k, p]] = b[[p, k]]

        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            b[i]      -= m * b[k]

    # backward substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i+1:] @ x[i+1:]) / A[i, i]
    return x






A = np.array([[20, 15, 10], [-3, -2.249, 7], [5, 1, 3]])
b = np.array([45, 1.751, 9])
print(gauss_partial_pivot(A, b))   # -> [1. 1. 1.]