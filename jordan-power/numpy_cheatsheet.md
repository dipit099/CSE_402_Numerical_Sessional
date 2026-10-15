# NumPy cheatsheet — for Gaussian elimination / LU / eigenvalue code

Built from the actual patterns used across your Gauss-Jordan, LU, and power-method files.

---

## 1. Creating & typing arrays

```python
A = np.array([[2, 1], [1, 2]], dtype=float)   # force float from the start
A = np.zeros((n, n))                          # n x n zeros, float64 by default
A = np.eye(n)                                 # identity matrix
A = np.zeros_like(A)                          # same shape/dtype as A, filled with 0
```

**Gotcha:** if the caller passes an integer array, force it before doing elimination math:

```python
A = A.astype(float).copy()
```

Without this, `A[i,k:] -= m * A[k,k:]` can silently truncate fractional results to integers, or crash outright depending on the op.

---

## 2. `.copy()` — avoid mutating the caller's array

```python
A = A.astype(float).copy()   # your function gets a private array
```

NumPy slices and `A = B` are **views/references**, not copies. If a function does row operations directly on its input without `.copy()`, it silently mutates the caller's original matrix. Always copy at the top of any function that transforms `A` in place.

---

## 3. Indexing & slicing basics

```python
A[i]        # row i (1-D array)      -- NOT column i
A[i, j]     # single element, row i col j
A[:, j]     # column j (needs the slice syntax)
A[i, k:]    # row i, from column k to the end
A[k:, j]    # column j, from row k to the end
A[i, :i]    # row i, columns 0..i-1  (used in forward substitution)
A[i, i+1:]  # row i, columns after the diagonal (used in backward substitution)
```

Rule of thumb: a bare `A[i]` on a 2-D array always means **row** `i`. To get a column you must write `A[:, i]`.

---

## 4. The "start from column k" elimination trick

```python
A[i, k:] -= m * A[k, k:]
```

Columns before `k` are already zero in both rows (from earlier steps), so there's no need to touch them — this single vectorized line replaces an inner `for j in range(k, n)` loop. Same idea appears in:

```python
U[i, k:] -= m * U[k, k:]     # LU elimination
```

---

## 5. Swapping rows (partial pivoting)

**Fancy indexing (concise):**

```python
A[[k, p]] = A[[p, k]]        # swap rows k and p in one line
b[[k, p]] = b[[p, k]]        # do the same to b / to P
```

**Hand-written loop (when an exam wants explicit code, not fancy indexing):**

```python
for j in range(n):
    A[k, j], A[p, j] = A[p, j], A[k, j]
```

**The trap that's easy to forget:** if you're building `L` incrementally during LU with pivoting, a row swap must also move the multipliers **already stored** in `L`'s earlier columns:

```python
if k > 0:
    L[[k, p], :k] = L[[p, k], :k]
```

---

## 6. Finding the pivot (partial pivoting)

**Vectorized:**

```python
p = k + np.argmax(np.abs(A[k:, k]))   # row (from k downward) with largest |entry| in column k
```

- `A[k:, k]` — candidates: column `k`, from row `k` down (rows above are already finalized)
- `np.abs(...)` — magnitude, since sign doesn't matter for pivoting
- `np.argmax(...)` — **index** of the largest value, relative to the sliced array
- `+ k` — convert that relative index back to a real row number

**Hand-written (when a task requires no `np.argmax`):**

```python
p = k
for i in range(k + 1, n):
    if abs(A[i, k]) > abs(A[p, k]):
        p = i
```

`np.max` vs `np.argmax`: `max` gives you the **value**, `argmax` gives you **where** it is. You need the index to know which row to swap.

---

## 7. Matrix multiply vs elementwise multiply

```python
A @ B          # matrix multiplication (or np.dot(A, B))
A * B          # elementwise multiplication -- NOT matmul, shapes must match
m * A[k, k:]   # scalar times a row -- broadcasts, this IS what you want in elimination
```

Using `*` where you meant `@` is a classic silent bug — it won't crash if shapes happen to broadcast, it'll just compute the wrong thing.

---

## 8. Dot products in substitution

```python
A[i, i+1:] @ x[i+1:]      # sum of (known coefficients) * (already-solved unknowns)
L[i, :i] @ z[:i]          # same idea, for forward substitution
```

This one line replaces an inner loop like `for j in range(...): sol[i] -= A[i][j]*x[j]`.

---

## 9. Reductions: max / min / sum / all / any

```python
np.max(y)              # largest value
np.argmax(y)            # index of largest value
np.max(np.abs(y))       # largest magnitude (use this for eigenvalue estimates, not np.max alone)
np.argmax(np.abs(y))    # index of largest-magnitude entry, sign preserved when you index back: y[np.argmax(np.abs(y))]

np.all(np.abs(row) < eps)   # True if EVERY entry of row is ~0 (used to detect a degenerate row)
np.any(...)                  # True if AT LEAST ONE entry satisfies the condition
```

**Gotcha:** `np.max(y)` picks the largest *signed* value, not the largest magnitude. For a vector like `[-5, 2]`, `np.max` gives `2`, silently missing the dominant `-5` entry. Use `y[np.argmax(np.abs(y))]` when you want magnitude-based selection but need to keep the sign.

---

## 10. `np.linalg` toolbox — the verification layer

```python
np.linalg.solve(A, b)        # solve Ax = b directly (uses LU + partial pivoting internally)
np.linalg.inv(A)              # matrix inverse (avoid using this to solve systems -- see below)
np.linalg.det(A)              # determinant
np.linalg.eig(A)              # (eigenvalues, eigenvectors) -- eigenvectors are COLUMNS, not rows
np.linalg.norm(x)              # ||x||_2 for a vector (default)
np.linalg.norm(A, 'fro')       # Frobenius norm for a matrix -- use for ||A - L@U||
np.linalg.norm(A @ x - b)      # residual check after solving
```

**Eigenvector gotcha:**

```python
eigvals, eigvecs = np.linalg.eig(A)
v = eigvecs[:, k]     # column k, NOT row k
```

**Sign ambiguity:** eigenvectors are only defined up to a scalar multiple. Both `v` and `-v` are equally valid for the same eigenvalue. When comparing your result to NumPy's:

```python
if np.dot(v, v_np) < 0:
    v_np = -v_np
```

**Eigenvalues come back in no particular order.** Find the dominant one explicitly:

```python
k = np.argmax(np.abs(eigvals))    # dominant (largest magnitude)
k = np.argmin(np.abs(eigvals))    # smallest magnitude
```

---

## 11. Normalizing a vector

```python
x = x / np.linalg.norm(x)     # unit length (||x||_2 = 1) -- NumPy's convention for eigenvectors
x = x / np.max(np.abs(x))     # scale so the largest-magnitude entry is 1 -- power-iteration convention
```

These are two *different* normalization conventions — if you're comparing your power-iteration result against `np.linalg.eig`, you need the first one right before returning, even if the loop itself uses the second.

---

## 12. Building a coefficient matrix from feature columns

```python
A = np.column_stack([t**2, t, np.ones_like(t)])   # [t^2  t  1] as columns
```

Turns several 1-D arrays into a 2-D matrix, one column each — the standard way to build a design/coefficient matrix for curve fitting (`np.ones_like(t)` gives a column of 1s matching `t`'s shape/dtype).

---

## 13. `None` from a missing `return` — a silent NumPy trap

```python
def backward_substitution(U, Z):
    sol = np.zeros(len(Z))
    ...                     # fills sol correctly
    # forgot `return sol`   <-- implicitly returns None

col = backward_substitution(U, Z)
sol[i] = col                 # assigns None into a float array...
```

NumPy does **not** raise an error here — it silently converts `None` to `nan`:

```python
>>> arr = np.zeros(3); arr[0] = None
>>> arr
array([nan,  0.,  0.])
```

If a result matrix comes out all-`NaN` with no traceback, check for a missing `return` before anything else.

---

## 14. Quick reference: shapes that trip people up

| Code              | What it actually is                                                                                       |
| ----------------- | --------------------------------------------------------------------------------------------------------- |
| `A[i]`          | row`i`                                                                                                  |
| `A[:, i]`       | column`i`                                                                                               |
| `A[i, j]`       | single scalar                                                                                             |
| `A[i:i+1, :]`   | row`i` **as a 2-D array** (shape `(1, n)`, not `(n,)`)                                        |
| `eigvecs[:, k]` | eigenvector`k` (column)                                                                                 |
| `sol.T`         | flips rows ↔ columns — needed whenever you filled a result matrix row-by-row but meant column-by-column |
