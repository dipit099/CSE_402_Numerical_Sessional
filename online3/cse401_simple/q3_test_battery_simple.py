"""
Practice Q3: RNG Test Battery (Which Test Catches Which Defect?)
Simple, Student-Friendly Implementation
"""

import math
from scipy.stats import chisquare

# ==============================================================================
# Streams Under Test (N = 1000)
# ==============================================================================
def stream_A(n=1000):
    """Good LCG: a = 16807, c = 0, m = 2^31 - 1"""
    m, x, out = 2**31 - 1, 123457, []
    for _ in range(n):
        x = (16807 * x) % m
        out.append(x / m)
    return out

def stream_B(n=1000):
    """The Ramp: 0.000, 0.001, ..., 0.999 (Uniform, but totally dependent!)"""
    return [(i % 1000) / 1000.0 for i in range(n)]

def stream_C(n=1000):
    """Middle-Square seed 5731 (Collapses to 4-cycle)"""
    x, out = 5731, []
    for _ in range(n):
        x = int(str(x * x).zfill(8)[2:6])
        out.append(x / 10000.0)
    return out

def stream_D(n=1000):
    """sqrt(Stream A): Independent, but NOT uniform (density is 2x)"""
    return [math.sqrt(r) for r in stream_A(n)]


# ==============================================================================
# Statistical Tests
# ==============================================================================
def test_chi_square(numbers):
    observed = [0] * 10
    for r in numbers:
        observed[min(int(r * 10), 9)] += 1
    stat, p = chisquare(observed)
    return "X (Reject)" if p < 0.05 else "Pass", stat

def test_ks(numbers):
    data = sorted(numbers)
    n = len(data)
    d_plus = max((i + 1) / n - r for i, r in enumerate(data))
    d_minus = max(r - i / n for i, r in enumerate(data))
    D = max(d_plus, d_minus)
    d_crit = 1.36 / math.sqrt(n)
    return "X (Reject)" if D > d_crit else "Pass", D

def test_autocorr(numbers, lag):
    n = len(numbers)
    start = 1
    M = (n - start) // lag - 1
    idx = [start + k * lag for k in range(M + 2)]
    sub = [numbers[j - 1] for j in idx]
    rho = sum(sub[k] * sub[k + 1] for k in range(M + 1)) / (M + 1) - 0.25
    sigma = math.sqrt(13 * M + 7) / (12 * (M + 1))
    Z0 = rho / sigma
    return "X (Reject)" if abs(Z0) > 1.96 else "Pass", Z0

def test_runs(numbers):
    n = len(numbers)
    # Signs of consecutive differences: True if rising, False if falling
    signs = [numbers[i + 1] > numbers[i] for i in range(n - 1)]
    runs = 1 + sum(1 for i in range(len(signs) - 1) if signs[i] != signs[i + 1])
    mean_runs = (2 * n - 1) / 3.0
    var_runs = (16 * n - 29) / 90.0
    Z0 = (runs - mean_runs) / math.sqrt(var_runs)
    return "X (Reject)" if abs(Z0) > 1.96 else "Pass", Z0


# ==============================================================================
# MAIN SCRIPT
# ==============================================================================
if __name__ == "__main__":
    streams = [("A (Good LCG)", stream_A(1000)),
               ("B (Ramp)", stream_B(1000)),
               ("C (Middle Square)", stream_C(1000)),
               ("D (sqrt LCG)", stream_D(1000))]

    print("=" * 75)
    print("PASS / FAIL MATRIX FOR THE TEST BATTERY (alpha = 0.05)")
    print("=" * 75)
    print(f"{'Test':<20} | {'Stream A':<11} | {'Stream B':<11} | {'Stream C':<11} | {'Stream D'}")
    print("-" * 75)

    test_names = [
        ("Chi-Square", lambda s: test_chi_square(s)),
        ("K-S Test", lambda s: test_ks(s)),
        ("Autocorr (Lag 1)", lambda s: test_autocorr(s, lag=1)),
        ("Autocorr (Lag 4)", lambda s: test_autocorr(s, lag=4)),
        ("Autocorr (Lag 5)", lambda s: test_autocorr(s, lag=5)),
        ("Runs Test", lambda s: test_runs(s)),
    ]

    for t_name, t_func in test_names:
        row_res = []
        for _, data in streams:
            verdict, _ = t_func(data)
            row_res.append(verdict)
        print(f"{t_name:<20} | {row_res[0]:<11} | {row_res[1]:<11} | {row_res[2]:<11} | {row_res[3]}")

    print("\n" + "=" * 75)
    print("KEY TAKEAWAYS FOR EXAMS:")
    print("=" * 75)
    print("1. Stream B (The Ramp) passes Chi-Square & K-S (Uniformity) perfectly,")
    print("   but fails Runs and Autocorrelation (Order is completely deterministic).")
    print("2. Stream C (Middle-Square 4-cycle) passes Autocorrelation at Lag 1 and Lag 5,")
    print("   but is CAUGHT at Lag 4 where the cycle repeats!")
    print("3. Stream D (sqrt(LCG)) passes Runs (Order is fine),")
    print("   but fails Chi-Square and K-S because its probability density is non-uniform.")
