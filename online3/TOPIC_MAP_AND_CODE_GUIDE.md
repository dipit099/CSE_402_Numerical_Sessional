 

# CSE 402 — Topic Map, Concepts & Code Reading Guide

Companion to `STUDY_GUIDE.md`. This file answers four things:

1. **Where** each topic lives in the two PDFs.
2. **Basics you must know beforehand** (the prerequisites the slides assume).
3. **Which technique to use when.**
4. **Which `.py` file to read for what** — since this is a lab project, code first.

> **Honesty note on the PDFs**
>
> - `6_rng_mcs.pdf` (90 MB, 330 pages) — I scanned it fully. It's **Beamer slides with animation overlays**, so one logical slide = several PDF pages. The **footer slide number** (bottom-right of each slide) is the reliable anchor; PDF page numbers drift (~5× the slide number early on, ~4× later). Ranges below are approximate; trust the slide number.
> - `5_simu_des_ssqs.pdf` (149 MB, 445 pages) — **too large for the reader (>100 MB limit), so I could not open it.** Its topics are mapped by section below (they match `des_single_server_simple.py` one-to-one), but without page numbers. This is the same "too big" problem you have — consider splitting it with a PDF tool if you need to open it (see note at the bottom).

---

## 1. Topic Location Map

### PDF 6 — `6_rng_mcs.pdf` (RNG + Monte Carlo). Deck runs slides 1–71.

| Topic                                                    | Slide # (footer) | ≈ PDF page | Also coded in                           | Study § |
| -------------------------------------------------------- | ---------------- | ----------- | --------------------------------------- | -------- |
| Why randomness / simulation motivation                   | 1–6             | 1–25       | —                                      | —       |
| **Two properties: Uniformity + Independence**      | **7**      | ~30         | (tests verify these)                    | §4      |
| Pseudo-random, repeatability (seed), defects             | 12–14           | ~55–65     | —                                      | §2      |
| **LCG worked example** (X₀=27, a=17, c=43, m=100) | **18**     | ~90         | `rng_simple.py`, `q1_lcg…`         | §2      |
| Period, maximum density                                  | 19–24           | ~100–125   | `rng_simple.py` (`find_lcg_period`) | §2      |
| Max-period conditions / Lehmer / RANDU defect*           | ~25–34          | ~125–160   | `rng_simple.py`, `q1_lcg…`         | §2      |
| Multiple testing / Bonferroni (why testing is subtle)    | 30–31           | ~148–155   | —                                      | §4      |
| **K-S test procedure** (sort, D⁺, D⁻)            | **35**     | ~165        | `rng_tests_simple.py`                 | §4.2    |
| **Chi-Square frequency test**                      | **37**     | ~175        | `rng_tests_simple.py`, most `q*`    | §4.1    |
| **Autocorrelation test** ("why subtract 0.25", σ) | **45–51** | ~180–208   | `rng_tests_simple.py`, `q3…`       | §4.3    |
| **Monte Carlo Simulation** (section divider)       | **58**     | ~208        | `monte_carlo_simple.py`, `q5…`     | §5      |
| Estimate I = E(Y): sample-mean idea                      | 62               | ~285        | `monte_carlo_simple.py`               | §5.1    |
| Sample-mean integration (∫sin, Ȳ(5)=1.83)              | 65               | ~250–290   | `monte_carlo_simple.py`, `q5…`     | §5.1    |
| "Does more sampling help?" error table + 1/√n           | 67–68           | ~310–318   | `q5_monte_carlo…`                    | §5.4    |
| **Estimating π (hit-or-miss)**                    | **69–70** | ~320–330   | `monte_carlo_simple.py`, `q5…`     | §5.2    |

*RANDU / max-period conditions fall in the slides-19–34 block; I confirmed the surrounding slides (18 LCG example, 24 max density, 35 K-S) but the deck's overlay density makes the exact page fuzzy — look in that block.

**NOT in this deck** (they come from assignments / your practice files, not the slides):

- **Middle-Square method** → assignment **C1** (`previous_ques/C1.docx`) + `middle_square_simple.py`, `q2_middle_square…`
- **Middle-Square Weyl (MSWS)** → assignment **B2** (`previous_ques/B2.pdf`) + `q2_middle_square…`
- **Buffon's Needle** → the **π/Buffon assignment** (your photos) — Monte Carlo applied; concept = §5.3
- **Inverse-transform variate generation** (exponential/Weibull/triangular) → `q4_inverse_transform…` + used in B2; concept = §6
- **Reliability networks** (series=min, parallel=max) → assignment **B2**; concept = §5.5
- **K-S deep dive with period-collapse** → assignment **C2** (`previous_ques/C2_kstest.pdf`)

### PDF 5 — `5_simu_des_ssqs.pdf` (Discrete-Event Simulation of Single-Server Queue)

*(Could not open — 149 MB. Section order below is standard for this deck and matches `des_single_server_simple.py`.)*

| Section (in order)                                                            | Coded in                                       | Study §  |
| ----------------------------------------------------------------------------- | ---------------------------------------------- | --------- |
| Systems, models, why simulate                                                 | —                                             | §1 intro |
| Discrete vs continuous;**next-event vs fixed-increment** time advance   | `des_single_server_simple.py`                | §1       |
| Components: clock,**state variables, event list, statistical counters** | `des_single_server_simple.py` (init routine) | §1       |
| Single-server queue: problem setup, arrival/departure events                  | both DES files                                 | §1       |
| **Arrival routine** & **Departure routine** logic                 | `des_single_server_simple.py` (main loop)    | §1       |
| **Hand-simulation example** (interarrivals 0.4,1.2,…; 6 delays)        | `des_single_server_simple.py` `__main__`   | §1       |
| Performance measures**d(n), q(n), u(n)** (area accumulators)            | both DES files                                 | §1       |
| Report / interpretation; extensions (balking, capacity)                       | `q6_queue_simulation…`                      | §1       |

---

## 2. Basics You Must Know Beforehand

The slides assume these. If any feels shaky, review it first — everything else builds on them.

**Probability & statistics**

- **Uniform(0,1):** every value in [0,1) equally likely. Mean = 0.5, variance = 1/12. All RNG output aims for this.
- **CDF F(x) = P(X ≤ x):** always rises 0→1. Inverse-transform (§6) literally inverts this.
- **Expected value E[X] = mean.** Monte Carlo works because *an average of samples estimates an expected value* (law of large numbers).
- **Variance / standard deviation:** spread. Standard **error** = std-dev/√n — shrinks as you add samples. This is the whole reason more samples help.

**Hypothesis testing** (needed for every RNG test)

- **H₀ (null):** "the numbers ARE uniform / independent." Tests try to *disprove* it.
- **α = 0.05:** significance level = allowed false-alarm rate.
- **Decision rules (memorize both forms):**
  - p-value form: **p < α → reject H₀** (numbers look bad). p ≥ α → don't reject (look fine).
  - statistic form: **test-stat > critical value → reject H₀.**
- **Critical values you'll reuse:** Z = 1.96 (two-sided, α=0.05); χ² = 16.919 (df=9); K-S = 1.36/√N (large N).

**Discrete math / CS**

- **Modular arithmetic:** `a mod m` = remainder. The heart of LCG. Know that only `m` distinct remainders exist → that caps the period.
- **gcd** (for the max-period condition `gcd(c,m)=1`).
- **FIFO queue:** first in, first out — customers served in arrival order (the DES uses `list.pop(0)`).

**Calculus (light)**

- An **integral = area under a curve.** Monte Carlo integration (§5.1) estimates that area by "average height × width."

---

## 3. Which Technique to Use When

**Generating uniform random numbers → LCG** (`X_{n+1}=(aX+c) mod m`). Default to the good one: **a=16807, c=0, m=2³¹−1** (Lehmer). Use middle-square only if the question explicitly asks (it's the "bad" historical one).

**Checking a generator — pick by what you suspect:**

| You want to check…                                      | Use                                         | Why                                   |
| -------------------------------------------------------- | ------------------------------------------- | ------------------------------------- |
| Are values spread evenly? (uniformity)                   | **Chi-Square** or **K-S**       | compare counts / CDF to flat          |
| Uniformity with**few** samples or exact CDF        | **K-S**                               | works on the sorted sample directly   |
| Uniformity with many samples, binned                     | **Chi-Square**                        | needs Eᵢ ≥ 5 per bin                |
| Are consecutive values related? (independence)           | **Autocorrelation**                   | catches periodic/serial patterns      |
| Is there an up/down trend pattern?                       | **Runs test**                         | order-based                           |
| Testing a**non-uniform** target (e.g. exponential) | **Chi-Square with equiprobable bins** | equal-width bins break on skewed data |

> **Golden rule:** uniformity and independence are *separate*. Always run at least one of each kind. A stream can pass χ²/K-S and still fail autocorrelation (see the §4.4 defect table).

**Turning uniforms into other distributions → Inverse transform** `X = F⁻¹(U)` (§6). Exponential: `−β·ln(1−U)`.

**Estimating a number (π, an integral, a probability) → Monte Carlo:**

- Integral of g(x) → **sample-mean**: `(b−a)·avg(g(Xᵢ))`.
- π, or an area/probability → **hit-or-miss**: fraction of points that land inside.
- System reliability → simulate component lifetimes, combine **series=min / parallel=max**, count survivors.

**Modeling a queue / waiting line over time → Discrete-Event Simulation** (§1). Use when events happen at irregular times and you need averages like delay, queue length, utilization.

---

## 4. Which `.py` File to Read for What

All files are runnable standalone (`python <file>`). Read them in this order for the lab.

### `cse401_simple/` — the clean reference implementations (learn from these first)

| File                                         | Read it to learn                                                     | Key functions to copy in an exam                                                              |
| -------------------------------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| **`rng_simple.py`**                  | LCG generation, period detection, the 3 max-period cases             | `lcg_step`, `generate_lcg_uniforms`, `find_lcg_period`, `check_max_period_conditions` |
| **`middle_square_simple.py`**        | Middle-square + its zero/cycle failure, χ² test                    | `middle_square` (note the `zfill(8)` + `[2:6]` slice), `find_cycle`                   |
| **`rng_tests_simple.py`**            | **The 3 core tests** with full formulas & table traces         | `kolmogorov_smirnov_test`, `chi_square_test`, `autocorrelation_test`                    |
| **`monte_carlo_simple.py`**          | Sample-mean integration, hit-or-miss π, confidence intervals        | `monte_carlo_integrate`, `estimate_pi`, `simple_lcg`                                    |
| **`des_single_server_simple.py`** ★ | **The DES queue** — next-event loop, area accumulators, d/q/u | `simulate_single_server_queue` (study the loop + the "areas BEFORE state" comment)          |

### `practice_simple/` — exam-style problems (read after the reference files)

| File                                              | Problem it solves                                                 | Extra thing it teaches                                                            |
| ------------------------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **`q1_lcg_investigation_simple.py`**      | LCG periods, degeneracies, χ² on several generators             | **RANDU** 3D-plane defect (`9X−6X'+X''≡0`) — passing χ² ≠ good RNG  |
| **`q2_middle_square_variants_simple.py`** | d-digit middle-square survey;**Weyl repair (MSWS)**         | why more digits can't fix it (birthday bound); the (x,w) state pair               |
| **`q3_test_battery_simple.py`**           | Runs 4 streams × 5 tests →**pass/fail matrix**            | **which test catches which defect** (the most exam-relevant file for tests) |
| **`q4_inverse_transform_simple.py`**      | Inverse-transform for Uniform/Exp/Weibull/Triangular/Discrete     | equiprobable bins; verify variates by**moments** (mean/variance)            |
| **`q5_monte_carlo_simple.py`**            | Sample-mean integration + CIs, π,**1/√n law**, vs Simpson | why MC beats Simpson in high dimensions                                           |
| **`q6_queue_simulation_simple.py`** ★    | Barber-shop DES with**balking (capacity k)**                | max queue length, lost customers; the "areas before state" reasoning              |

### Fast lookup — "I need code for X"

- **Generate uniforms** → `rng_simple.py: generate_lcg_uniforms` (or the 4-line `simple_lcg` in `monte_carlo_simple.py`).
- **Find a period / detect a cycle** → `rng_simple.py: find_lcg_period` (works for LCG; `find_cycle` for middle-square).
- **Chi-square test** → `rng_tests_simple.py: chi_square_test` (from scratch) or `q1…: chi_square_test_10bins` (via SciPy).
- **K-S test** → `rng_tests_simple.py: kolmogorov_smirnov_test`.
- **Autocorrelation** → `rng_tests_simple.py: autocorrelation_test` (mind the M calc & −0.25).
- **Runs test** → `q3_test_battery_simple.py: test_runs`.
- **Monte Carlo integral** → `monte_carlo_simple.py: monte_carlo_integrate`.
- **Estimate π** → `monte_carlo_simple.py: estimate_pi`.
- **Buffon's needle** → adapt `estimate_pi`: two uniforms per needle (x, θ), cross if `x ≤ (L/2)·sin θ`, then `π̂ = 2L/(D·P̂)`.
- **Inverse-transform variate** → `q4_inverse_transform_simple.py` (pick the distribution's function).
- **Single-server queue** → `des_single_server_simple.py: simulate_single_server_queue`.
- **Queue with balking / capacity** → `q6_queue_simulation_simple.py: simulate_barber_shop`.

---

## 5. The One-Line Concept Flow

```
GENERATE           TEST                         USE
LCG / middle-sq -> chi²/K-S (uniform?)      -> inverse transform (make any distribution)
                   autocorr/runs (indep?)   -> Monte Carlo (integral, π, reliability)
                                            -> Discrete-Event Simulation (queues, delay/util)
```

Every topic in this course is one box in that pipeline. If you know which box a question is in, you know which file and which formula to reach for.

---

## Appendix: opening the 149 MB DES PDF yourself

If you need the DES slides open and they're too big/slow, split them once with a PDF tool. On your Mac:

```bash
# needs qpdf (brew install qpdf) — splits into 50-page chunks
qpdf --split-pages=50 "5_simu_des_ssqs.pdf" "des_part-%d.pdf"
```

Or in Preview: open → Print → save a page range as PDF. The DES content maps to the sections in §1's table above.
