import numpy as np


def gauss_jordan(A, b, verbose=True):
    """Solves Ax = b by reducing the augmented matrix [A|b] all the way to
    [I|x] (Gauss-Jordan elimination with partial pivoting). Unlike plain
    Gaussian elimination, every column is cleared both ABOVE and BELOW the
    pivot, and the pivot row is normalized to 1 -- so no back substitution
    step is needed at the end."""
    n = len(b)
    Aug = np.column_stack([A.astype(float), b.astype(float)])

    for k in range(n):
        # partial pivoting: largest |entry| in column k, at/below row k
        p = k + np.argmax(np.abs(Aug[k:, k]))
        if p != k:
            Aug[[k, p]] = Aug[[p, k]]
            if verbose:
                print(f"Swap R{k} <-> R{p}")

        # normalize the pivot row so the pivot becomes exactly 1
        Aug[k] = Aug[k] / Aug[k, k]

        # eliminate this column from every OTHER row (above and below)
        for i in range(n):
            if i != k:
                Aug[i] -= Aug[i, k] * Aug[k]

        if verbose:
            print(f"\nAfter processing column {k}:\n{np.round(Aug, 4)}")

    x = Aug[:, -1]
    return x


if __name__ == "__main__":
    A = np.array([[2, 1, 1],
                  [4, 3, 3],
                  [8, 7, 9]], dtype=float)
    b = np.array([4, 10, 24], dtype=float)

    x = gauss_jordan(A, b)
    print("\nSolution x:", np.round(x, 4))

    print("\n=== Verification ===")
    x_np = np.linalg.solve(A, b)
    print("np.linalg.solve:", np.round(x_np, 4))
    print("||Ax - b||_2   :", np.linalg.norm(A @ x - b))
