# LU Decomposition — Theory Cheat Sheet

Quick-reference theory for online exam written questions. Pairs with [lu_practice_questions.md](lu_practice_questions.md) (implementation practice) and [numpy_cheatsheet.md](numpy_cheatsheet.md) (code patterns).

---

## 1. What LU decomposition is

For a square matrix `A`, LU decomposition writes:

```
A = L @ U
```

where `L` is **lower triangular** and `U` is **upper triangular**. Two common conventions for which triangular factor carries the 1s on its diagonal:

- **Doolittle**: `L` has 1s on the diagonal (this is what all the practice code uses)
- **Crout**: `U` has 1s on the diagonal instead

`U` is exactly the matrix Gaussian elimination produces. `L` is new information: it records the elimination steps themselves (the multipliers), which Gaussian elimination normally throws away.

---

## 2. Where `L` actually comes from — the elementary-matrix view

Each elimination step (subtract `m` times row `k` from row `i`) is equivalent to left-multiplying by an **elementary matrix** `E`:

```
E_k = I - m * (e_i e_k^T)      (identity, with -m in position (i,k))
```

Applying every elimination step in order:

```
E_n ... E_2 E_1 A = U
```

so:

```
A = (E_1^-1 E_2^-1 ... E_n^-1) U = L U
```

Each `E_k^-1` just flips the sign of its multiplier, and — crucially — because elimination proceeds column by column with the multipliers stacking in strictly-lower-triangular slots that never interact, the product of all the `E_k^-1` is simply the **unit lower-triangular matrix with every multiplier dropped straight into position `(i,k)`**. This is *why* the code

```python
m = U[i,k] / U[k,k]
L[i,k] = m
```

is enough — no matrix inversion or multiplication is actually needed at runtime; the algebra behind it guarantees the multiplier lands directly in the right slot of `L`.

---

## 3. Existence and uniqueness

**`A = LU` (no pivoting) exists if and only if every leading principal minor of `A` is nonzero:**

```
det(A[0:1, 0:1]) != 0
det(A[0:2, 0:2]) != 0
...
det(A[0:n-1, 0:n-1]) != 0
```

`det(A) != 0` (i.e. `A` invertible) is **not sufficient** — it's specifically the leading (upper-left) submatrices that must be nonsingular, since elimination needs a nonzero pivot at every one of the first `n-1` steps.

**Counterexample:** `A = [[0, 1], [1, 0]]`. `det(A) = -1` (invertible), but the leading `1x1` minor is `det([0]) = 0` → no `LU` exists without reordering rows.

**Uniqueness:** if `A = L1 U1 = L2 U2` with both factorizations following the same convention (e.g. both Doolittle), then `L1 = L2` and `U1 = U2`. Proof sketch: `L2^-1 L1 = U2 U1^-1`; the left side is unit lower triangular, the right side is upper triangular — the only matrix that is both is the identity, so `L1 = L2` and `U1 = U2`.

---

## 4. Partial pivoting: `PA = LU`

Partial pivoting fixes the existence problem above. Before eliminating column `k`, swap in the row (at or below `k`) with the **largest-magnitude entry** in that column. `P` is the permutation matrix recording every swap (starts as `I`, each swap does `P[[k,p]] = P[[p,k]]` in lockstep with `A`/`U`).

**Claim: for every invertible matrix, `PA = LU` exists**, even when plain `A = LU` doesn't. Reason: partial pivoting only fails to find a nonzero pivot if an entire column (at and below the diagonal) is zero — but that would make `A` singular, contradicting invertibility. So for any invertible `A`, a valid pivot is always available at every step.

**Two numerical-stability reasons to pivot even when you technically could avoid it:**
1. A pivot of exactly 0 causes division by zero → multiplier becomes `inf`, and `inf * 0` in the next row update becomes `nan` — silent corruption, not a clean crash (numpy doesn't raise an exception for either).
2. A pivot that's merely *small* (not zero) still causes a huge multiplier `m = A[i,k]/A[k,k]`, which amplifies rounding error when subtracted into the row (catastrophic cancellation). Partial pivoting guarantees every multiplier satisfies `|m| <= 1`, which bounds error growth throughout elimination — this is what "numerically stable" means here.

**Important consequence for solving:** once you have `P, L, U`, you must solve `Lz = P @ b`, **not** `Lz = b` — `b`'s rows need the same reordering that was applied to `A`'s rows.

**Also important when implementing pivoting by hand:** if you're filling in `L`'s multipliers incrementally *while* doing elimination, a row swap at step `k` must also swap the already-computed multipliers in `L`'s earlier columns (`L[[k,p], :k] = L[[p,k], :k]`) — otherwise `L` ends up describing an elimination history that never happened.

---

## 5. Solving `Ax = b` via LU

```
A = LU  ->  Ax = b  ->  LUx = b
```

Introduce `z = Ux`. The system splits into two **triangular** solves:

```
Lz = b     (forward substitution, top-down: row 0 has 1 unknown, ..., row i uses rows 0..i-1)
Ux = z     (backward substitution, bottom-up: row n-1 has 1 unknown, ..., row i uses rows i+1..n-1)
```

Each triangular solve is `O(n^2)`. This is the entire point of factoring first: the expensive part (`O(n^3)`) happens once, and it doesn't depend on `b`.

---

## 6. Cost / complexity

| Operation | Cost |
|---|---|
| LU factorization | `O(n^3)` (~ `2n^3/3` flops) |
| One triangular solve (forward or backward) | `O(n^2)` |
| Forward + backward together | `O(n^2)` (~ `2n^2` flops) |
| Solving for `k` different right-hand sides (same `A`) via LU | `O(n^3 + k*n^2)` |
| Solving for `k` different right-hand sides via repeated Gaussian elimination | `O(k*n^3)` |

**Why LU wins when there are many right-hand sides:** elimination on `A` doesn't depend on `b`, so redoing it for every new `b` is wasted work. Once `k > n` (true almost always in practice — simulations, repeated load cases, etc.), the `k*n^2` term dominates the LU-based cost and is a full factor of `n` cheaper per solve than starting over.

---

## 7. Determinant via LU

```
PA = LU
det(P) * det(A) = det(L) * det(U)
```

- `det(L) = 1` (unit lower triangular — Doolittle convention)
- `det(P) = (-1)^(number of row swaps)`, and `det(P)` is its own inverse (it's ±1)

So:

```
det(A) = (-1)^(#swaps) * product(diagonal of U)
```

This is dramatically cheaper than cofactor expansion (`O(n!)`) — it falls out of the factorization you likely needed anyway, at `O(n^3)`.

**A zero on `U`'s diagonal after elimination means `det(A) = 0`** — the matrix is singular. This connects directly to the solution-classification question below.

---

## 8. Inverse via LU

`A^-1`'s column `i` is the solution `x` of `A x = e_i` (`e_i` = column `i` of the identity). Factor once, then for each `i`:

```
z = forward_substitution(L, P @ e_i)
x_i = backward_substitution(U, z)
```

Assemble all `x_i` as columns of `A^-1`.

**Why not just use the inverse to solve systems?** Given the factors, a direct solve costs `~2n^2` flops. Forming `A^-1` first costs `n` solves `~2n^3` flops — `n` times more expensive for no benefit, and it compounds rounding error over `n` solves instead of one. Rule of thumb taught alongside this: **never explicitly form the inverse just to solve `Ax=b`.**

---

## 9. Classifying `Ax = b`: unique / no solution / infinite solutions

After forward elimination (with or without pivoting), look at the reduced rows:

| Row after elimination | Meaning | Result |
|---|---|---|
| No fully-zero coefficient row | full rank, `rank(A) = n` | **Unique solution** |
| A zero-coefficient row `0 = c`, `c != 0` | contradiction | **No solution** |
| A zero-coefficient row `0 = 0` | redundant equation, `rank(A) < n` | **Infinite solutions** (pick free variable(s), back-substitute the rest) |

**A zero pivot during elimination does NOT automatically mean no solution.** Two distinct cases:
- Some row *below* the zero pivot still has a nonzero entry in that column → pivoting swaps it up, elimination continues normally, and a **unique solution is still possible**.
- The pivot *and* every entry below it in that column are also zero → the matrix is singular; whether the system then has no solution or infinite solutions depends on the right-hand side of the resulting zero row(s), per the table above.

This is Rouché–Capelli in disguise: compare `rank(A)` to `rank([A | b])` and to `n`.

---

## 10. Symmetric positive-definite matrices: Cholesky

If `A` is symmetric (`A = A^T`) and positive definite (`x^T A x > 0` for all `x != 0`), a specialized factorization applies:

```
A = L L^T
```

(single triangular factor, no separate `U` needed). Benefits over general LU:
- **No pivoting required** — SPD guarantees all pivots stay positive throughout elimination.
- **About half the cost**: `~n^3/3` flops instead of `~2n^3/3`, and half the storage (only one triangular factor).

Common in least-squares (`A^T A` is always symmetric, and positive definite if `A` has full column rank) and covariance matrices.

---

## 11. Condition number — why a small residual isn't the whole story

After solving, checking `||Ax - b||` small does **not** guarantee `x` is close to the true answer. The relevant quantity is the **condition number**:

```
kappa(A) = ||A|| * ||A^-1||
```

For a well-conditioned matrix, small residual implies small error in `x`. For an **ill-conditioned** matrix (large `kappa(A)`), a tiny residual can coexist with a large error:

```
relative error in x  <~  kappa(A) * relative residual
```

This is why two people solving the "same" system with slightly different pivoting/rounding can get visibly different answers even though both report a tiny `||Ax-b||`.

---

## 12. Quick answers to common written questions

- **Why does storing multipliers directly into `L` give a valid `A = LU`?** See §2 — it follows from the product of elementary matrices collapsing into a single unit-lower-triangular matrix.
- **Is the factorization unique?** Yes, given a fixed convention (Doolittle or Crout) and that it exists. See §3.
- **What if a pivot column is entirely zero (at and below the diagonal) for every possible row choice?** `A` is singular: `det(A) = 0` (§7), and `Ax=b` has either no solution or infinitely many, never a unique one (§9).
- **What does partial pivoting guarantee about `L`?** Every multiplier satisfies `|L[i][j]| <= 1`, since the pivot chosen is always the largest-magnitude entry available.
- **Cost of factor vs cost of solve?** Factor `O(n^3)` once; each solve (given the factors) `O(n^2)`. See §6.
- **When would you use Cholesky instead of general LU?** `A` symmetric positive definite → half the cost, no pivoting needed (§10).
- **Small residual but "wrong" answer?** Ill-conditioning — check `kappa(A)`, not just `||Ax-b||` (§11).

---

## 13. Bugs worth knowing (things that look right but aren't)

Real mistakes found while building this practice set — good to recognize on sight:

- **Missing `return`** at the end of a substitution function: the computed answer is discarded, the function returns `None`, and assigning `None` into a NumPy float array **silently becomes `NaN`** — no exception, no traceback.
- **Swapped parameter order** between a function's call site and its (possibly redefined/shadowed) definition — passing `L` where `U` is expected and vice versa. Forward/backward substitution don't check the triangular structure of what they're given; they'll happily run and produce a wrong-but-plausible-looking answer.
- **Forgetting `.T`** when a result matrix is built row-by-row but was supposed to represent columns (e.g. each solved `x_i` of an inverse is naturally produced as a row of a working array and must be transposed before returning).
- **Forgetting to permute `b`** when solving with `PA = LU` — must solve `Lz = P@b`, not `Lz = b`.
- **Forgetting to swap `L`'s stored multipliers** when swapping rows mid-elimination — corrupts the elimination history `L` is supposed to represent.
