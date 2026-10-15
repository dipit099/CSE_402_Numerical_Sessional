"""
CSE 401: Monte Carlo Simulation (Simple Implementation)
Includes:
  1. Monte Carlo Integration (Sample-Mean Method)
  2. Hit-or-Miss Estimation of Pi
  3. Buffon's Needle Estimation of Pi
  4. Error & Confidence Interval Analysis
"""

import math

# Simple standard LCG (Lehmer Generator)
def simple_lcg(seed=123457):
    x = seed
    a = 16807
    m = 2**31 - 1
    while True:
        x = (a * x) % m
        yield x / m

# ==============================================================================
# 1. Monte Carlo Integration (Sample-Mean Estimator)
# ==============================================================================
def monte_carlo_integrate(func, a, b, n, seed=123457):
    """
    Estimates the integral I = integral_a^b g(x) dx using n random samples:
        I ~= (b - a) * (1 / n) * sum( g(X_i) ), where X_i ~ Uniform(a, b)
    """
    rng = simple_lcg(seed)
    width = b - a
    
    total = 0.0
    total_sq = 0.0
    
    for _ in range(n):
        r = next(rng)
        x = a + width * r
        y = width * func(x)
        total += y
        total_sq += y * y
        
    mean = total / n
    variance = (total_sq - n * (mean ** 2)) / (n - 1) if n > 1 else 0.0
    std_dev = math.sqrt(max(0.0, variance))
    std_error = std_dev / math.sqrt(n)
    
    # 95% Confidence Interval: mean +/- 1.96 * SE
    ci_lower = mean - 1.96 * std_error
    ci_upper = mean + 1.96 * std_error
    
    return {
        "estimate": mean,
        "std_dev": std_dev,
        "std_error": std_error,
        "confidence_interval": (ci_lower, ci_upper)
    }


# ==============================================================================
# 2. Hit-or-Miss Estimation of Pi
# ==============================================================================
def estimate_pi(n, seed=123457):
    """
    Estimates pi by generating (X, Y) in unit square [0, 1]x[0, 1]
    and checking if X^2 + Y^2 <= 1.
        pi ~= 4 * (hits / n)
    """
    rng = simple_lcg(seed)
    hits = 0
    
    for _ in range(n):
        x = next(rng)
        y = next(rng)
        if x * x + y * y <= 1.0:
            hits += 1
            
    p_hat = hits / n
    pi_estimate = 4.0 * p_hat
    
    std_error = 4.0 * math.sqrt(p_hat * (1.0 - p_hat) / n)
    ci_lower = pi_estimate - 1.96 * std_error
    ci_upper = pi_estimate + 1.96 * std_error
    
    return {
        "estimate": pi_estimate,
        "hits": hits,
        "std_error": std_error,
        "confidence_interval": (ci_lower, ci_upper)
    }


# ==============================================================================
# 3. Buffon's Needle Estimation of Pi
# ==============================================================================
def buffon_needle_pi(n, L=1.0, D=2.0, seed=123457):
    """
    Estimates pi with Buffon's Needle: a needle of length L is dropped onto a
    floor ruled with parallel lines spaced D apart (requires L <= D so the
    needle can cross at most one line).

    Each drop needs TWO independent uniforms (this is the key point vs. a plain
    1-D estimate):
      x     = distance from the needle's CENTER to the nearest line,
              x ~ Uniform(0, D/2)       ->  x     = (D/2)  * R1
      theta = acute angle to the lines,
              theta ~ Uniform(0, pi/2)  ->  theta = (pi/2) * R2

    A needle CROSSES a line when its half-length's vertical reach covers x:
              x <= (L/2) * sin(theta)

    The crossing probability is P = 2L / (pi*D). Inverting for pi gives the
    estimator (with p_hat = hits / n):
              pi ~= 2L / (D * p_hat)
    """
    rng = simple_lcg(seed)
    hits = 0

    for _ in range(n):
        x = (D / 2.0) * next(rng)             # center distance  (first uniform)
        theta = (math.pi / 2.0) * next(rng)   # orientation      (second uniform)
        if x <= (L / 2.0) * math.sin(theta):
            hits += 1

    p_hat = hits / n
    # Guard: if no needle crossed, p_hat = 0 and the estimator would divide by 0.
    pi_estimate = (2.0 * L) / (D * p_hat) if p_hat > 0 else float("inf")

    return {
        "estimate": pi_estimate,
        "hits": hits,
        "p_hat": p_hat,
    }


# ==============================================================================
# Main: Verification & Demos
# ==============================================================================
if __name__ == "__main__":
    # ----- 1. Monte Carlo Integration of sin(x) over [a, b] -----
    integrand = math.sin
    integrand_name = "sin(x)"
    int_a = 0.0
    int_b = math.pi
    lower_label = "0"       # display label for int_a
    upper_label = "pi"      # display label for int_b
    int_width = int_b - int_a
    width_label = "(pi - 0)"   # display label for the interval width (int_b - int_a)
    true_integral = 2.0
    n_values_int = (10, 100, 1000, 10000, 100000)
    print("=" * 70)
    print(f"1. Monte Carlo Integration: integral_{lower_label}^{upper_label} {integrand_name} dx "
          f"(Exact = {true_integral:g})")
    print("=" * 70)
    # Worked example with fixed points from slides
    slide_samples = [0.52, 1.19, 2.83, 0.21, 1.77, 2.50, 0.89, 1.53, 2.97, 2.11]
    heights = [integrand(x) for x in slide_samples]
    avg_height = sum(heights) / len(heights)
    slide_est = int_width * avg_height
    print(f"Slide Samples: {slide_samples}")
    print(f"Average Height = {avg_height:.3f}")
    print(f"Ybar({len(slide_samples)}) = {width_label} * {avg_height:.3f} = {slide_est:.3f} "
          f"(True = {true_integral:g}, Error = {abs(slide_est - true_integral):.3f})\n")

    print(f"{'n':<10} | {'Estimate':<10} | {'Error':<10} | {'95% Confidence Interval'}")
    print("-" * 60)
    for n in n_values_int:
        res = monte_carlo_integrate(integrand, int_a, int_b, n)
        lo, hi = res["confidence_interval"]
        err = res["estimate"] - true_integral
        print(f"{n:<10} | {res['estimate']:<10.4f} | {err:<10.4f} | [{lo:.4f}, {hi:.4f}]")

    # ----- 2. Hit-or-Miss Estimation of Pi -----
    true_pi = math.pi
    n_values_pi = (100, 1000, 10000, 100000, 1000000)
    print("\n" + "=" * 70)
    print(f"2. Hit-or-Miss Estimation of Pi (Exact = {true_pi:.5f}...)")
    print("=" * 70)
    print(f"{'n':<10} | {'Pi Estimate':<12} | {'|Error|':<10} | {'95% Confidence Interval'}")
    print("-" * 60)
    for n in n_values_pi:
        res_pi = estimate_pi(n)
        lo, hi = res_pi["confidence_interval"]
        err = abs(res_pi["estimate"] - true_pi)
        print(f"{n:<10} | {res_pi['estimate']:<12.5f} | {err:<10.5f} | [{lo:.5f}, {hi:.5f}]")

    # ----- 3. Buffon's Needle Estimation of Pi (L = 1, D = 2) -----
    buffon_L = 1.0
    buffon_D = 2.0
    n_values_buffon = (100, 1000, 10000, 100000, 1000000)
    print("\n" + "=" * 70)
    print(f"3. Buffon's Needle Estimation of Pi (L = {buffon_L:g}, D = {buffon_D:g}, Exact = {true_pi:.5f}...)")
    print("=" * 70)
    print(f"{'n':<10} | {'Hits':<10} | {'p_hat':<10} | {'Pi Estimate':<12} | {'|Error|'}")
    print("-" * 60)
    for n in n_values_buffon:
        res_b = buffon_needle_pi(n, L=buffon_L, D=buffon_D)
        err = abs(res_b["estimate"] - true_pi)
        print(f"{n:<10} | {res_b['hits']:<10} | {res_b['p_hat']:<10.5f} | {res_b['estimate']:<12.5f} | {err:.5f}")
