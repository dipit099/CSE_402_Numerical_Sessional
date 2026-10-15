import numpy as np


def power_iteration(A, x0, tol=1e-6, max_iter=1000):
    """Power iteration: returns the dominant eigenvalue, the (normalized)
    dominant eigenvector, and the list of eigenvalue estimates per iteration."""
    x = x0.astype(float).copy()
    eigen_estimates = []

    for _ in range(max_iter):
        y = A @ x
        # eigenvalue estimate = entry of largest MAGNITUDE (sign preserved)
        eigen_estimated = y[np.argmax(np.abs(y))]
        eigen_estimates.append(eigen_estimated)

        # scale x so its largest-magnitude entry is 1 (keeps iteration stable)
        x_new = y / eigen_estimated

        # converge on the eigenVECTOR, not just the eigenvalue: the value
        # estimate can settle several iterations before the vector does
        if np.max(np.abs(x_new - x)) < tol:
            x = x_new
            break
        x = x_new

    # normalize to unit 2-norm before returning, so it is comparable
    # with NumPy's eigenvector output
    x = x / np.linalg.norm(x)
    return eigen_estimates[-1], x, eigen_estimates


# ----------------------------------------------------------------------
# Given 4x4 matrix
A = np.array([[8, 2, 0, 0],
              [2, 8, 0, 0],
              [0, 0, 3, 1],
              [0, 0, 1, 3]], dtype=float)
x0 = np.array([1, 1, 1, 1], dtype=float)

lam, v, history = power_iteration(A, x0)

print("=== Our power iteration ===")
print("dominant eigenvalue :", lam)
print("dominant eigenvector:", v)
print("iterations          :", len(history))

# ----------------------------------------------------------------------
# Verification with NumPy
eigvals, eigvecs = np.linalg.eig(A)
k = np.argmax(np.abs(eigvals))          # index of the dominant eigenvalue


k = np.argmax(np.abs(eigvals))
lam_np = eigvals[k]
v_np = eigvecs[:, k]                    # eigenvectors are the COLUMNS

print("\n=== NumPy verification ===")
print("dominant eigenvalue :", lam_np)
print("dominant eigenvector:", v_np)

# NumPy may return the eigenvector with opposite sign: v and -v are both
# valid eigenvectors, so align the signs before comparing.
if np.dot(v, v_np) < 0:
    v_np = -v_np
    print("(sign flipped for comparison -- both v and -v are valid)")

print("\neigenvalue  difference:", abs(lam - lam_np))
print("eigenvector difference:", np.linalg.norm(v - v_np))
print("residual ||Av - lam*v||_2 :", np.linalg.norm(A @ v - lam * v))
