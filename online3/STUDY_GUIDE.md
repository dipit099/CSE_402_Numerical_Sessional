# CSE 402 — Simulation CT Study Guide

Compact theory + worked examples + the edge cases that quietly cost marks.
Instructor said **focus on simulation**, so §1 (the DES queue) is the anchor — but every CT topic is here.

**Contents**
1. [DES: Single-Server Queue ★](#1-discrete-event-simulation-single-server-queue-main-focus)
2. [LCG & Random Number Generation](#2-lcg--random-number-generation)
3. [Middle-Square & Weyl Repair](#3-middle-square--the-weyl-repair)
4. [Testing RNGs (χ², K-S, Autocorr, Runs)](#4-testing-random-numbers)
5. [Monte Carlo Simulation](#5-monte-carlo-simulation)
6. [Inverse-Transform Variates](#6-inverse-transform-variate-generation)
7. [Exam Trap Cheat-Sheet](#7-exam-trap-cheat-sheet)

---

## 1. Discrete-Event Simulation: Single-Server Queue (MAIN FOCUS)

*The "barber shop" / M/M/1-style queue by the next-event time-advance algorithm.*

A DES only jumps the clock to the moments **something happens** (an event) — never tick-by-tick. Between events the system state is frozen. You keep an **event list** of scheduled future events, always process the **earliest** one, update statistics, then schedule the next.

### The two events
- **Arrival** — a customer shows up.
- **Departure** — the customer in service finishes.

Only these two kinds exist. The whole simulation is: repeatedly pick whichever of `time_next_arrival` / `time_next_departure` is smaller, advance the clock to it, and handle it.

### State variables you must track

| Variable | Meaning |
|---|---|
| `clock` | current simulated time |
| `server_status` | 0 = idle, 1 = busy |
| `num_in_queue` | customers *waiting* (not counting the one in service) |
| `time_arrival[]` | FIFO list of arrival times of those waiting (to compute their delay later) |
| `time_last_event` | clock value at the previous event (for the area rectangles) |

### The algorithm (the part you write / trace)

```python
while num_delayed < N_required:
    # 1. TIMING: pick next event = the smaller of the two future times
    if time_next_arrival < time_next_departure:
        clock = time_next_arrival;   event = ARRIVAL
    else:
        clock = time_next_departure; event = DEPARTURE

    # 2. ACCUMULATE AREAS  ***BEFORE*** touching state  (see trap)
    dt = clock - time_last_event
    time_last_event = clock
    area_num_in_q      += num_in_queue  * dt
    area_server_status += server_status * dt

    # 3. HANDLE the event
    if event == ARRIVAL:
        schedule next arrival (clock + next interarrival)
        if server idle:            # goes straight into service
            delay = 0; record it; num_delayed++; server_status = 1
            schedule departure (clock + next service time)
        else:                      # server busy -> wait
            num_in_queue++; time_arrival.append(clock)
    else:  # DEPARTURE
        if num_in_queue == 0:      # nobody waiting -> server goes idle
            server_status = 0; time_next_departure = INF
        else:                      # pull next customer (FIFO)
            num_in_queue--; delay = clock - time_arrival.pop(0)
            record delay; num_delayed++
            schedule departure (clock + next service time)
```

> ⚠ **Edge case #1 — the #1 exam killer.** Update the area accumulators **BEFORE you change any state variable.** The interval `dt` that just elapsed was spent at the *old* queue length / server status. If you increment `num_in_queue` first and then multiply by `dt`, the rectangle has the wrong height and every metric is off. Rule: *measure the interval you just left, then step into the new state.*

### Performance measures (the answers)

```
d(n) = total delay in queue / n            (average delay per customer)
q(n) = area_num_in_q / clock               (time-average #  in queue)
u(n) = area_server_status / clock          (server utilization, fraction busy)
```

- **d(n)** is a plain average over the *n customers*.
- **q(n)** and **u(n)** are *time-averages* = area under the step-function ÷ total time. That is why we accumulate rectangles.

> ◆ **Why time-average, not a simple average?** Queue length lives in continuous time. A queue of 3 held for 2 hours is not the same as a queue of 3 held for 2 seconds. So you integrate length × duration (Σ height × width), giving `q(n) = ∫L(t)dt / T`. Only `d(n)` — one number per customer — is a per-item mean.

### Worked trace (the classic 6-delay example)

Interarrivals A = [0.4, 1.2, 0.5, 1.7, 0.2, 1.6, …], Services S = [2.0, 0.7, 0.2, 1.1, 3.7, 0.6], stop after 6 delays. Watch how `AreaQ` / `AreaB` grow by `height × dt`, and how the clock only lands on event times.

| Clock | Event | Stat | Q | #Del | AreaQ | AreaB |
|---:|---|---:|---:|---:|---:|---:|
| 0.00 | Initialize | 0 | 0 | 0 | 0.00 | 0.00 |
| 0.40 | Arrival → service (D=0) | 1 | 0 | 1 | 0.00 | 0.00 |
| 1.60 | Arrival → joins queue | 1 | 1 | 1 | 0.00 | 1.20 |
| 2.10 | Arrival → joins queue | 1 | 2 | 1 | 0.50 | 1.70 |
| 2.40 | Departure (next served, D=0.80) | 1 | 1 | 2 | 1.10 | 2.00 |
| 3.10 | Departure (next served, D=1.00) | 1 | 0 | 3 | 1.80 | 2.70 |
| 3.30 | Departure → server idle | 0 | 0 | 3 | 1.80 | 2.90 |
| 3.80 | Arrival → service (D=0) | 1 | 0 | 4 | 1.80 | 2.90 |
| … | … |  |  |  |  |  |
| 8.60 | Departure (D=3.00) → 6th delay | 1 | 2 | 6 | 9.90 | 7.70 |

Delays collected = [0, 0.8, 1.0, 0, 0.9, 3.0], stop time T = 8.60. Then:

```
d(6) = (0+0.8+1.0+0+0.9+3.0)/6 = 5.7/6 = 0.95
q(6) = 9.90 / 8.60          = 1.15
u(6) = 7.70 / 8.60          = 0.90
```

**Sample-row check (row at clock 2.10 → 2.40):** `dt = 0.30`, queue was **2**, server was **1**. So AreaQ gains `2 × 0.30 = 0.60` (0.50 → 1.10 ✓) and AreaB gains `1 × 0.30 = 0.30` (1.70 → 2.00 ✓). This is exactly the "measure the interval you just left" rule.

### Variations they may add

> ⚠ **Edge cases in the queue**
> - **Tie (arrival time == departure time):** pick a fixed rule and state it. The code's `<` test sends equal times to the else-branch → **departure-first**. A departure-first tie frees the server so the arriving customer can enter service.
> - **Departure with empty queue:** do NOT record a delay — just set server idle and `time_next_departure = INF`. Forgetting the INF causes a phantom departure.
> - **Balking / waiting-room limit k:** if `num_in_queue >= k` on arrival, the customer *leaves without joining* (a "lost"/balk). It never contributes a delay and never enters `num_delayed`.
> - **First arrival** always enters an idle server with delay 0; `time_next_departure` starts at INF so the first event is guaranteed to be an arrival.
> - **Stopping rule:** "N delays completed" ≠ "N arrivals". Customers still waiting when you stop are not counted.

---

## 2. LCG & Random Number Generation

*Deterministic recurrence that looks random. Never use Python's `random` in these tasks.*

```
X_{n+1} = (a·X_n + c) mod m        (integers)
U_n     = X_n / m                  in [0, 1)
```

- **a** = multiplier, **c** = increment, **m** = modulus, **X₀** = seed.
- **Mixed LCG:** c ≠ 0.  **Multiplicative LCG (MCG):** c = 0.
- **Period** = how many values before the sequence repeats. Max possible is m (mixed) — never more, since only m distinct states exist.

**Worked step — X₀=27, a=17, c=43, m=100**

```
X₁ = (17·27 + 43) mod 100 = 502 mod 100 = 2   → U₁ = 0.02
X₂ = (17·2  + 43) mod 100 =  77 mod 100 = 77  → U₂ = 0.77
X₃ = (17·77 + 43) mod 100 = 1352 mod 100 = 52 → U₃ = 0.52
X₄ = (17·52 + 43) mod 100 =  927 mod 100 = 27 → back to seed! Period = 4
```

Only 4 distinct values {2, 77, 52, 27} — a terrible generator despite valid-looking parameters.

### Maximum-period conditions (memorize the 3 cases)

| Case | Conditions | Max period |
|---|---|---:|
| **1. Mixed**, m = 2ᵇ, c ≠ 0 | `gcd(c, m) = 1` **and** `a = 1 + 4k` (a−1 divisible by 4) | m |
| **2. Multiplicative**, m = 2ᵇ, c = 0 | seed **odd** **and** `a = 3+8k or 5+8k` | m / 4 |
| **3. Multiplicative**, m prime, c = 0 | a is a **primitive root** mod m | m − 1 |

*General (Hull–Dobell) for mixed LCG full period m:* (1) gcd(c,m)=1, (2) a−1 divisible by every prime factor of m, (3) a−1 divisible by 4 if m divisible by 4. Cases 1–3 are the exam-friendly specializations.

> ◆ **Standard "good" generator — Lehmer / MINSTD:** a = 16807, c = 0, m = 2³¹−1 (prime). Period = m−1 ≈ 2.1 billion. This is the reliable one used inside the Monte Carlo / inverse-transform code.

> ⚠ **Degenerate & failure cases**
> - **c=0, X₀=0** → stuck at 0 forever (period 1). A multiplicative LCG must never be seeded with 0.
> - **a=1, c=0** → X stays constant (period 1).
> - **Even seed in Case 2** → you fall into a shorter sub-cycle, not the full m/4.
> - **Passing a uniformity test ≠ good generator.** **RANDU** (a=65539, c=0, m=2³¹) passes 1-D tests but consecutive triples satisfy `9·Xᵢ − 6·Xᵢ₊₁ + Xᵢ₊₂ ≡ 0 (mod 2³¹)` — all points lie on just 15 planes in 3-D. Uniformity in 1-D can hide structure in higher dimensions.

---

## 3. Middle-Square & the Weyl Repair

*Von Neumann's method — historically important, practically broken.*

**Algorithm (4-digit):**
1. Square the current 4-digit number.
2. Pad the square to **8 digits** with leading zeros.
3. Take the **middle 4 digits** → next value. Divide by 10⁴ for U.

```
Seed 5731:  5731² = 32844361  → middle 4 = 8443
            8443² = 71284249  → middle 4 = 2842  → ...
```

> ⚠ **Edge cases — where it dies**
> - **Zero trap:** once the middle becomes 0000, it's stuck forever (absorbing state).
> - **Short cycles:** seed 5731 collapses into a repeating 4-cycle. Many seeds fall into tiny cycles fast.
> - **Padding is mandatory:** if the square has < 8 digits and you don't `zfill(8)`, you slice the wrong digits — a silent bug.
> - **More digits doesn't fix it:** a d-digit state has only 10ᵈ values, and squaring isn't a bijection → cycles appear around the birthday bound ~√(10ᵈ). Bigger d only delays collapse.

> ◆ **The Weyl-sequence fix (MSWS).** Keep a second running state `w` that always advances by an odd constant: `w = (w + s) mod 10ᵈ`, then `x = (middle(x²) + w) mod 10ᵈ`. The Weyl part has full period 10ᵈ and never sticks at 0, so the pair (x, w) can't collapse. The 32-bit version (B2 exam) uses `c = 0xb5ad4ec5`, mod 2³², and outputs the top bits via `x >> 16` (i.e. `floor(x/2¹⁶) / 2¹⁶`).

---

## 4. Testing Random Numbers

*Two independent questions: (A) are they **uniform**? (B) are they **independent**? You must test both.*

### 4.1 Chi-Square — uniformity (frequency)

```
Bin [0,1) into k bins.  Eᵢ = N/k (expected per bin).
χ² = Σ (Oᵢ − Eᵢ)² / Eᵢ        df = k − 1
Reject H₀ if χ² > χ²_crit   (df=9, α=0.05 → 16.919)
```

H₀: the numbers are Uniform[0,1). Big χ² = observed counts are far from flat = reject.

> ⚠ **Chi-square edge cases**
> - Rule of thumb **Eᵢ ≥ 5** per bin, so N ≥ 5k. With too many bins the test is invalid.
> - **Bin index:** `int(r*k)`, but clamp `r=1.0` (or rounding to it) down to bin k−1, else index out of range.
> - Testing a **non-uniform** target (e.g. exponential variates) needs **equiprobable bins** — choose cutpoints from the inverse CDF so each bin has prob 1/k. Equal-width bins leave tail bins with ~0 expected → invalid.
> - **p < α → reject.** Don't flip it. Large p = looks uniform.

### 4.2 Kolmogorov–Smirnov — uniformity (CDF distance)

Sort the sample R₍₁₎ ≤ … ≤ R₍ₙ₎. Compare the empirical CDF to the straight line of Uniform[0,1].

```
D⁺ = maxᵢ ( i/N − R₍ᵢ₎ )
D⁻ = maxᵢ ( R₍ᵢ₎ − (i−1)/N )
D  = max(D⁺, D⁻)
Large-N critical (α=0.05): D_crit = 1.36 / √N
Reject H₀ if  D > D_crit
```

> ⚠ **K-S edge cases**
> - **You must SORT first** — indices i run over the sorted order.
> - Mind the off-by-one: D⁺ uses `i/N`, D⁻ uses `(i−1)/N`. Mixing them is the classic slip.
> - Small N uses a **table** value (N=5 → 0.565); only large N uses 1.36/√N.
> - **K-S is blind to ordering/period.** A generator that cycles (period p ≪ N) just repeats the same set of values — the *marginal* distribution still looks perfectly uniform, so K-S happily accepts. That's the whole point of the C2 exam: a broken MCG passes K-S. Catch it with a **period check** or an **independence** test.

### 4.3 Autocorrelation — independence

Tests whether values `ℓ` apart are correlated. H₀: ρ = 0 (independent).

```
M = largest int with  i + (M+1)·ℓ ≤ N
ρ̂ = [ 1/(M+1) · Σ_{k=0..M} R_{i+kℓ}·R_{i+(k+1)ℓ} ] − 0.25
σ = √(13M + 7) / (12(M+1))
Z₀ = ρ̂ / σ      Reject H₀ if |Z₀| > 1.96  (α=0.05)
```

> ⚠ **Autocorrelation edge cases**
> - The `−0.25` comes from E[RᵢRⱼ] = ¼ for independent uniforms. Don't drop it.
> - **Choosing lag ℓ matters:** a hidden period is only caught at a lag that lines up with it. A middle-square 4-cycle sails through lag 1 and lag 5 but is caught at **lag 4**. If you test one lag, you can miss the flaw — test several.
> - Watch the index base: `i` is 1-based in the formula; in code, offset by −1.

### 4.4 Runs (up/down) — independence

```
a = number of runs of rising/falling
mean = (2N − 1)/3      var = (16N − 29)/90
Z₀ = (a − mean)/√var   Reject if |Z₀| > 1.96
```

> ◆ **Which test catches which defect (memorize this table)**
>
> | Stream | χ² / K-S (uniform?) | Runs / Autocorr (independent?) |
> |---|---|---|
> | Good LCG (Lehmer) | Pass | Pass |
> | Ramp 0.000,0.001,… (sorted) | **Pass** (perfectly flat) | **FAIL** (totally ordered) |
> | Middle-square 4-cycle | Fail | Fail at lag 4 (passes 1 & 5) |
> | √(uniform) | **FAIL** (density not flat) | Pass (order is fine) |
>
> Takeaway: uniformity and independence are *separate* — a stream can ace one and fail the other. Always test both.

---

## 5. Monte Carlo Simulation

*Estimate a quantity by averaging over many random samples.*

### 5.1 Sample-mean integration

```
I = ∫ₐᵇ g(x) dx  ≈  (b − a) · (1/n) · Σ g(Xᵢ),   Xᵢ ~ Uniform(a,b)
```

Draw x uniformly in [a,b], average the heights, multiply by the width. Example: ∫₀^π sin x dx (true = 2): sample x's in [0,π], average sin values, × π.

### 5.2 Hit-or-miss estimation of π

```
Drop (x,y) in the unit square.  Hit if x² + y² ≤ 1.
π̂ = 4 · (hits / n)
```

The quarter-circle has area π/4; the fraction of hits estimates that.

### 5.3 Buffon's Needle (the assignment)

Parallel lines distance D apart, needle length L, with L ≤ D. Probability of crossing:

```
P = 2L / (π·D)      ⇒      π̂ = 2L / (D · P̂)
For each needle:  x ~ Uniform(0, D/2),  θ ~ Uniform(0, π/2)
Crosses a line  ⇔  x ≤ (L/2)·sin θ
P̂ = (# crossings) / N
```

> ⚠ **Buffon / MC edge cases**
> - You need **two independent uniforms per needle** — one for position x, one for angle θ. Don't reuse the same U.
> - With L=1, D=2 the true crossing prob is 1/π ≈ 0.318. If P̂ comes out 0 (tiny N), π̂ blows up — needs enough N.
> - The estimate is only as good as the RNG. Q3 of the assignment: going 100k → 1M does **not** guarantee more accuracy if the LCG's *period* is short — you just recycle the same numbers, so the error plateaus. This is the key insight they want.

### 5.4 Error, convergence & confidence

```
Standard error ∝ 1/√n      (the "√n law")
95% CI:  estimate ± 1.96 · SE
Hit-or-miss:  SE = 4·√( p̂(1−p̂)/n )
```

> ◆ **The 1/√n law — expect this question.** To **halve** the error you need **4×** the samples. So `|error|·√n ≈ constant`. This is slow, but the crucial win: it's independent of dimension. Simpson's rule is O(n⁻⁴) in 1-D (far better) but needs nᵈ grid points in d dimensions, becoming hopeless past d≈4. Monte Carlo stays O(n⁻¹ᐟ²) in *any* dimension — that's why we use it.

### 5.5 Reliability networks (B2 exam)

Simulate system survival by combining component lifetimes Xᵢ (drawn via inverse-transform):

```
Series  (both needed):    T = min(Tₐ, T_b)
Parallel(either works):   T = max(Tₐ, T_b)
p̂ = (1/N) · Σ 1[ Yₖ ≥ θ ]      (fraction surviving past θ days)
```

> ⚠ **Network edge cases**
> - **Series = min** (weakest link fails first), **parallel = max** (survives as long as any branch lives). Swapping these is the common mistake.
> - Reduce the topology inside-out: collapse each series/parallel block into one lifetime, then combine.
> - Adding a spare in parallel helps most on the component that is currently the system's **bottleneck** (the min in the dominating series path / the weakest branch) — reason from structure, not by re-simulating.
> - Use a fixed **stream-assignment scheme** (e.g. Uᵢ = U(4k+i)) so each replication uses its own dedicated uniforms — keeps replications independent and reproducible.

---

## 6. Inverse-Transform Variate Generation

*Turn a Uniform(0,1) into any distribution: solve U = F(x) for x.*

```
Method:  draw U ~ Uniform(0,1),  return X = F⁻¹(U)
```

| Distribution | CDF F(x) | Inverse X = F⁻¹(U) |
|---|---|---|
| Uniform(a,b) | (x−a)/(b−a) | `a + (b−a)·U` |
| Exponential(mean β) | 1 − e^(−x/β) | `−β·ln(1−U)` |
| Weibull(k, c) | 1 − e^(−(x/c)ᵏ) | `c·(−ln(1−U))^(1/k)` |
| Triangular(a,b,mode c) | piecewise | if U ≤ (c−a)/(b−a): `a+√(U(b−a)(c−a))`; else `b−√((1−U)(b−a)(b−c))` |
| Discrete | staircase | smallest value whose cumulative prob ≥ U |

> ⚠ **Inverse-transform edge cases**
> - **Use 1−U, and never let U=0** in the log: `ln(0)` is −∞. Since U∈[0,1), `1−U` stays in (0,1] and is safe; `ln(U)` directly would blow up at U=0. (Both 1−U and U are uniform, but 1−U dodges the endpoint.)
> - Inverse-transform is a **deterministic map** — it adds no randomness. If the input uniform stream has a short period or correlation, the variates *inherit* the same flaws (e.g. service times repeat in a cycle).
> - Verify by **moments**: Exponential(β) has mean β, variance β². Uniform(a,b) has variance (b−a)²/12.

---

## 7. Exam Trap Cheat-Sheet

*The ones that quietly lose marks — scan this last before the CT.*

1. **DES:** accumulate areas *before* mutating state. Every metric depends on it.
2. **DES:** departure on empty queue → server idle + departure time = ∞, no delay recorded.
3. **DES:** "N delays" ≠ "N arrivals"; declare your tie-breaking rule.
4. **q(n), u(n)** are area÷time (time-averages); only **d(n)** is a per-customer mean.
5. **LCG:** c=0 with seed 0, or a=1, gives period 1. Multiplicative needs odd seed.
6. **Passing χ²/K-S ≠ good RNG** — RANDU, ramp, and short-period MCGs all prove it.
7. **K-S:** sort first; D⁺ uses i/N, D⁻ uses (i−1)/N; blind to period/order.
8. **χ²:** need Eᵢ≥5; use equiprobable bins for non-uniform targets; p<α ⇒ reject.
9. **Autocorr:** keep the −0.25; test multiple lags — a cycle hides at the wrong lag.
10. **Monte Carlo:** error ∝ 1/√n (4× samples to halve error); wins in high dimensions.
11. **Buffon:** two independent U's per needle; more samples can't beat a short RNG period.
12. **Inverse transform:** use ln(1−U); it only maps randomness, never creates it.
13. **Reliability:** series = min, parallel = max (don't swap).
14. **Middle-square:** always zero-pad the square before slicing the middle.
