# `cse401_simple/` — File & Topic Index

Which file covers which topic, the key functions, and which assignment it solves.
8 Python files total. All run standalone: `python <file>.py`.

---

## Quick map: topic → file

| Topic | File | Key function(s) |
|---|---|---|
| LCG (mixed & multiplicative), R = X/m | `rng_simple.py` | `lcg_step`, `generate_lcg_uniforms` |
| Period detection / cycle length | `rng_simple.py` | `find_lcg_period` |
| Max-period conditions (Cases 1, 2, **3 prime**) | `rng_simple.py` | `check_max_period_conditions`, `is_primitive_root` |
| Middle-Square (d-digit) | `middle_square_simple.py` | `middle_square`, `find_cycle` |
| Middle-Square Weyl (MSWS, 32-bit) | `middle_square_simple.py` | `msws_generator` |
| Chi-Square uniformity test | `rng_tests_simple.py` | `chi_square_test` |
| Kolmogorov–Smirnov (K-S) test | `rng_tests_simple.py` | `kolmogorov_smirnov_test` |
| Autocorrelation test | `rng_tests_simple.py` | `autocorrelation_test` |
| **Runs test** (up/down) | `q3_test_battery_simple.py` | `test_runs` |
| Test battery / which test catches which defect | `q3_test_battery_simple.py` | `stream_A..D`, all `test_*` |
| Monte Carlo integration (sample-mean) | `monte_carlo_simple.py` | `monte_carlo_integrate` |
| Hit-or-miss π estimation | `monte_carlo_simple.py` | `estimate_pi` |
| **Buffon's Needle** π estimation | `monte_carlo_simple.py` | `buffon_needle_pi` |
| Confidence intervals / 1-over-√n error | `monte_carlo_simple.py` | (inside both estimators) |
| Inverse-transform variates (Unif/Exp/Weibull/Tri/Discrete) | `variate_generation_simple.py` | `inverse_*` |
| Series / parallel reliability (min / max) | `variate_generation_simple.py` | `series`, `parallel`, `b2_system_survival` |
| Monte Carlo reliability network | `variate_generation_simple.py` | `reliability_simulation`, `reliability_with_spare` |
| Single-server queue (next-event DES) | `des_single_server_simple.py` | `simulate_single_server_queue` |
| Queue with **balking / capacity k** | `q6_queue_simulation_simple.py` | `simulate_barber_shop` |
| Performance metrics d(n), q(n), u(n) | both DES files | (returned dict) |

---

## File → topics (detail)

### 1. `rng_simple.py` — Random number generation (LCG)
- LCG `X_{n+1}=(aX+c) mod m`, uniforms `R=X/m`; period/cycle detection.
- Max-period checker for all three cases: **Case 1** (mixed, m=2ᵇ), **Case 2** (multiplicative, m=2ᵇ), **Case 3** (m prime → primitive-root check).
- Helpers: `_is_prime`, `_prime_factors`, `is_primitive_root`.
- Demos: Examples 1 (mixed), 2 (multiplicative), 4 (Lehmer a=16807, m=2³¹−1).

### 2. `middle_square_simple.py` — Middle-Square + Weyl repair
- `middle_square(seed, n, d=4)` — general **d-digit** (d=4 = classic).
- `find_cycle` — zero-trap / short-cycle detection.
- `msws_generator` — 32-bit Middle-Square Weyl Sequence (the repaired generator).
- `perform_chi_square_test` — scipy χ², p-value decision.
- **Solves assignment C1** (middle-square) and **B2 Task 1** (MSWS).

### 3. `rng_tests_simple.py` — Statistical tests (uniformity + independence)
- `chi_square_test(numbers, k=10, alpha=0.05, chi2_crit=None)` — pass the given critical value, or it auto-computes for df=k−1.
- `kolmogorov_smirnov_test` — D⁺, D⁻, D vs `1.36/√N`.
- `autocorrelation_test(start, lag)` — ρ̂, σ, Z₀ vs 1.96.
- **Solves K-S tasks (C2 Task 2)**; χ² and autocorrelation for all `q*`.
- ⚠️ Note: the **Runs test lives in `q3_test_battery_simple.py`**, not here.

### 4. `monte_carlo_simple.py` — Monte Carlo estimation
- `monte_carlo_integrate(func, a, b, n)` — sample-mean integration + 95% CI.
- `estimate_pi(n)` — circle hit-or-miss (x²+y²≤1 → π≈4·hits/n).
- `buffon_needle_pi(n, L, D)` — needle drop; **two uniforms per drop**, hit if `x ≤ (L/2)·sinθ`, `π̂ = 2L/(D·p̂)`.
- **Solves assignment A1** (Buffon's needle) and general integration/π tasks.

### 5. `variate_generation_simple.py` — Variate generation & reliability
- `inverse_uniform/exponential/weibull/triangular/discrete` — inverse-transform.
- `series`=min, `parallel`=max; `b2_system_survival` = `max(T2, min(T1, max(T3,T4)))`.
- `reliability_simulation` (stream scheme U(4k+i)); `reliability_with_spare` answers **B2 Task 3** (best spare = component 2).
- **Solves B2 Task 2** and inverse-transform tasks.

### 6. `des_single_server_simple.py` — Discrete-event single-server queue
- `simulate_single_server_queue(interarrivals, services, num_delays)` — next-event loop, area accumulators (updated **before** state change), returns d(n), q(n), u(n) + full trace.
- Verified 6-delay example: d=0.95, q=1.15, u=0.90.

### 7. `q3_test_battery_simple.py` — Test battery (copied from practice)
- 4 streams (good LCG / ramp / middle-square / √uniform) × 5 tests → pass/fail matrix.
- Adds the **Runs test** (`test_runs`) and multi-lag autocorrelation.
- Teaches: uniformity vs independence are separate; which test catches which defect.

### 8. `q6_queue_simulation_simple.py` — Queue with balking (copied from practice)
- `simulate_barber_shop(..., capacity=k)` — single-server DES **with balking**: arrivals leave if the queue is full; tracks max queue length and lost customers.
- Extends file 6 with the capacity/finite-waiting-room variation.

---

## Coverage vs. previous assignments

| Assignment | Fully solvable here? | Files used |
|---|---|---|
| **C1** (Middle-Square) | ✅ | `middle_square_simple.py` |
| **C2** (K-S on MCG) | ✅ | `rng_simple.py` + `rng_tests_simple.py` |
| **A1** (Buffon's Needle) | ✅ | `monte_carlo_simple.py` |
| **B2** (MSWS + reliability) | ✅ | `middle_square_simple.py` + `variate_generation_simple.py` |

Every pipeline topic — **generate → test → use (variates / Monte Carlo / DES)** — now has a home in this folder.
