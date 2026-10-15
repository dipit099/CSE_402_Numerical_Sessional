# LU Decomposition — Practice Questions (exam-style, ~30-35 min each)

## Q1 — Basic Doolittle LU + solve (the "B1 format" question)

Given:

```
A = [[ 4,  3,  2,  1],
     [ 8, 11,  6,  4],
     [ 4,  9, 11,  7],
     [12, 15, 16, 20]]        b = [10, 29, 31, 63]
```

**Tasks**

1. Implement Doolittle LU decomposition (L has 1s on the diagonal) **without**
   using any library routine. The multiplier computation and the storing of
   multipliers into L must be explicitly hand-written.
2. Print L and U after processing **each column** (i.e., show the factorization
   being built step by step).
3. Verify the factorization by printing `||A - L@U||_F` (should be ~0).
4. Solve `Ax = b` using your own forward substitution (`Lz = b`) and backward
   substitution (`Ux = z`). Print z and x.
5. Verify x against `np.linalg.solve(A, b)` and print `||Ax - b||_2`.

**Written question:** You need to solve `Ax = b` for 500 different vectors b
(same A). Explain why LU is the right tool and compare the total cost with
running full Gaussian elimination 500 times. (Give the O(·) counts.)

---

## Q2 — When naive LU breaks: pivoting and existence

Given:

```
A1 = [[0, 2, 1],          A2 = [[1, 2, 3],
      [1, 1, 1],                [2, 4, 7],
      [2, 1, 3]]                [3, 5, 3]]
```

**Tasks**

1. Try your Q1 Doolittle code on A1. Report what happens and explain **why**
   (which quantity becomes undefined, at which step).
2. Implement LU **with partial pivoting**: compute P, L, U such that
   `P @ A = L @ U`. Represent P however you like (permutation matrix or an
   index list), but pivot selection and row swapping must be hand-written.
   NOTE: when you swap a row during elimination you must also swap the
   already-computed multipliers in L — demonstrate this in code.
3. Print every pivot chosen and every swap performed (like the B1 question).
4. Verify by printing `||P@A - L@U||_F` for both A1 and A2.
5. Solve `A1 x = [5, 3, 6]` using P, L, U. (Careful: solve `Lz = P@b`, not `Lz = b`.)

**Written question:** Does every invertible matrix have an LU decomposition
without pivoting? State the exact condition for existence, and give a 2×2
counterexample. Does adding partial pivoting fix existence for every
invertible matrix?

---

## Q3 — Determinant and inverse from LU

Use the P, L, U code from Q2 on:

```
A = [[2, 1, 1],
     [4, 3, 3],
     [8, 7, 9]]
```

**Tasks**

1. Compute det(A) from the LU factors. State the formula you used, including
   how row swaps affect the sign. Verify with `np.linalg.det(A)`.
2. Compute A⁻¹ column-by-column: for each column eᵢ of the identity, solve
   `Lz = P@eᵢ` then `Ux = z`; assemble the columns. Verify with
   `np.linalg.inv(A)` and print `||A @ A_inv - I||_F`.
3. Now solve `Ax = b` for `b = [4, 10, 24]` twice: (a) via the two triangular
   solves, (b) via `A_inv @ b`. Print both results.

**Written question:** Both methods in task 3 give the same x. Why is method
(a) still preferred in practice? Give both the cost argument (flop counts for
one solve, given the factors) and the accuracy argument.

---

## Q4 — Inverse power method WITHOUT forming A⁻¹  (course tie-in)

Your earlier code computed A⁻¹ explicitly and ran power iteration on it. This
question removes that inefficiency.

Given:

```
A = [[8, 2, 0, 0],
     [2, 8, 0, 0],
     [0, 0, 3, 1],
     [0, 0, 1, 3]]        x0 = [1, 1, 1, 1]
```

**Tasks**

1. Factor A = LU **once**.
2. Implement the inverse power method where each iteration performs
   `y = A⁻¹ x` as **two triangular solves** (`Lz = x`, then `Uy = z`) instead
   of ever computing A⁻¹.
3. Print the eigenvalue estimate at each iteration. Report the smallest
   eigenvalue of A (remember: the iteration converges to the largest
   eigenvalue of A⁻¹ — convert it!) and its normalized eigenvector.
4. Verify with `np.linalg.eig(A)` (smallest |eigenvalue| and its column).
   Handle the possible sign flip of the eigenvector.

**Written question:** Per iteration, what is the cost of your triangular-solve
version vs. multiplying by a precomputed A⁻¹? Both are O(n²) — so why is the
LU version still considered better? (Think: cost of the setup phase, and
sparsity — what happens to the zeros of this block matrix in L,U vs in A⁻¹?)

---

## Q5 — Short theory bank (one-liners, typical viva/written questions)

1. Why does storing the multiplier `m = U[i][k]/U[k][k]` at position L[i][k]
   produce a matrix satisfying A = LU? (Hint: elementary matrices.)
2. Doolittle puts the 1s on L's diagonal; Crout puts them on U's. Is the
   factorization unique once you fix that convention? Under what condition?
3. During factorization U[k][k] becomes exactly 0 for every choice of pivot
   row (the whole column below is 0 too). What does this tell you about A?
   About det(A)? About solving Ax = b?
4. What is the flop count of LU factorization? Of one forward+backward solve?
5. If A is symmetric positive definite, which specialized factorization
   replaces LU, and what does it cost relative to LU?
6. True or false: partial pivoting guarantees all entries of L satisfy
   |L[i][j]| ≤ 1. Why is that numerically desirable?
7. You solved Ax = b via LU and got a small ||Ax - b|| but x looks wildly
   different from a friend's answer. What property of A explains this, and
   which quantity measures it?

---
---

# Answer key (compact)

**Q1 written:** Factor once: O(n³) (≈ 2n³/3 flops). Each extra b costs only two
triangular solves: O(n²). Total: O(n³) + 500·O(n²). Naive: 500 full
eliminations = 500·O(n³). The elimination work on A does not depend on b, so
LU caches it; Gaussian elimination throws it away each time.

**Q2.1:** A1[0][0] = 0 → the first multiplier m = A[i][0]/0 is a division by
zero. The matrix is invertible; only the *order* of rows is bad.

**Q2 written:** No. LU without pivoting exists iff all leading principal
minors are nonzero (upper-left 1×1, 2×2, ..., (n-1)×(n-1) submatrices have
nonzero determinant). Counterexample: [[0,1],[1,0]] — invertible (det = -1)
but the 1×1 leading minor is 0, so no LU. With partial pivoting, PA = LU
exists for **every** invertible (indeed every square) matrix.

**Q3.1:** det(A) = (-1)^s · Π U[i][i], where s = number of row swaps.
L contributes det = 1 (unit diagonal). For this A: det = 4... verify in code.

**Q3 written:** Cost: given the factors, two triangular solves ≈ 2n² flops;
but obtaining A⁻¹ first costs n pairs of solves ≈ 2n³ extra, and A⁻¹b is
another 2n² anyway — you pay n× more for nothing. Accuracy: forming A⁻¹
compounds rounding error over n solves and loses structure/sparsity; the
direct solve has a smaller error bound (backward stable with pivoting).
Rule of thumb: never form the inverse to solve a system.

**Q4.3:** Smallest eigenvalue of this A is 2 (blocks: {10,6} and {4,2});
the iteration converges to 1/2 on A⁻¹, so report 1/estimate = 2, eigenvector
≈ [0, 0, 1, -1]/√2.

**Q4 written:** Precomputing A⁻¹ costs ≈ 2n³ up front vs ≈ 2n³/3 for LU — 3×
more setup for the same O(n²) per-iteration cost. Sparsity: A is block
tridiagonal → L and U keep the zero blocks (bandwidth is preserved), so the
triangular solves touch few entries; A⁻¹ of a sparse matrix is generally
**dense**, destroying the savings. For large sparse matrices this is the
difference between feasible and infeasible.

**Q5:**
1. Elimination is multiplication by elementary matrices: Eₙ...E₁A = U, so
   A = (E₁⁻¹...Eₙ⁻¹)U, and that product is exactly unit lower triangular with
   the multipliers in the (i,k) slots. Inverting Eᵢ just flips the sign of the
   multiplier, and multiplying the inverses stacks them without interaction.
2. Yes — if A is invertible and the factorization exists, Doolittle (or
   Crout) form is unique. Proof sketch: L₁U₁ = L₂U₂ ⇒ L₂⁻¹L₁ = U₂U₁⁻¹;
   left side is unit lower triangular, right side upper triangular ⇒ both = I.
3. A is singular: det(A) = (-1)^s·ΠU[i][i] = 0. Ax = b then has either no
   solution or infinitely many (the B1 classification), never a unique one.
4. Factorization ≈ 2n³/3 flops; each triangular solve ≈ n², so a
   forward+backward pair ≈ 2n².
5. Cholesky: A = LLᵀ. About half the flops (≈ n³/3) and half the storage;
   needs no pivoting because SPD guarantees positive pivots.
6. True. Multipliers are A[i][k]/pivot and the pivot is the largest |·| in
   the column, so |m| ≤ 1. This prevents the row updates from amplifying
   roundoff error (keeps element growth in check).
7. A is ill-conditioned; the condition number κ(A) = ||A||·||A⁻¹|| measures
   it. A small residual does not imply a small error in x when κ is large:
   relative error in x can be ≈ κ(A) × relative residual.
