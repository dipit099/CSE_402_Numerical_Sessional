"""
CSE 401: Statistical Testing of Random Numbers (Simple Implementation)
Includes:
  1. Kolmogorov-Smirnov (K-S) Test for Uniformity
  2. Chi-Square Frequency Test for Uniformity
  3. Autocorrelation Test for Independence
"""

import math

# ==============================================================================
# 1. Kolmogorov-Smirnov (K-S) Test
# ==============================================================================
def kolmogorov_smirnov_test(numbers, alpha=0.05):
    """
    Performs K-S test for H0: R_i ~ Uniform[0, 1].
    
    Formulas:
      D+ = max_i ( i/N - R_(i) )
      D- = max_i ( R_(i) - (i-1)/N )
      D  = max(D+, D-)
    """
    sorted_r = sorted(numbers)
    N = len(sorted_r)
    
    d_plus_max = -math.inf
    d_minus_max = -math.inf
    
    table_rows = []
    for i, r in enumerate(sorted_r, start=1):
        d_plus = (i / N) - r
        d_minus = r - ((i - 1) / N)
        
        d_plus_max = max(d_plus_max, d_plus)
        d_minus_max = max(d_minus_max, d_minus)
        
        table_rows.append((i, r, i / N, d_plus, (i - 1) / N, d_minus))
        
    D = max(d_plus_max, d_minus_max)
    
    # Critical value table for alpha = 0.05
    ks_table_005 = {1: 0.975, 2: 0.842, 3: 0.708, 4: 0.624, 5: 0.565, 10: 0.410, 20: 0.294}
    if N in ks_table_005:
        d_crit = ks_table_005[N]
    else:
        # Large sample approximation for alpha = 0.05: 1.36 / sqrt(N)
        d_crit = 1.36 / math.sqrt(N)
        
    decision = "Reject H0" if D > d_crit else "Do not reject H0"
    
    return {
        "D": D,
        "D+": d_plus_max,
        "D-": d_minus_max,
        "critical_value": d_crit,
        "decision": decision,
        "table": table_rows
    }


# ==============================================================================
# Helper: a random-number generator (returns the numbers themselves, in [0,1)).
# Pass this (or your own middle_square / lcg function) into the chi-square tests.
# ==============================================================================
def lcg_uniforms(n, seed=123457, a=16807, c=0, m=2**31 - 1):
    """
    Generate n uniforms in [0, 1) with an LCG:  X = (a*X + c) mod m,  R = X/m.
    Defaults are the Lehmer generator (a=16807, c=0, m=2^31-1).
    """
    x = seed
    numbers = []
    for _ in range(n):
        x = (a * x + c) % m
        numbers.append(x / m)
    return numbers


# ==============================================================================
# 2. Chi-Square Goodness-of-Fit Test
#    Statistic is computed BY HAND in both; only the decision rule differs.
#      chi2 = sum_i ( (O_i - E_i)^2 / E_i ),  E_i = N / k,  df = k - 1
# ==============================================================================
def chi_square_test_pvalue(numbers, k=10, alpha=0.05):
    """
    Chi-Square uniformity test -- decide using the P-VALUE.
      p = chi2.sf(chi2_stat, df)   (df = k - 1),   reject H0 if p < alpha
    `numbers` is a list of values in [0, 1) (build it in the caller).
    Returns (chi2_stat, p_value, decision, observed).
    """
    from scipy.stats import chi2 as chi2_dist

    # Step 1: count how many numbers land in each bin.
    N = len(numbers)
    observed = [0] * k
    for r in numbers:
        idx = int(r * k)          # r in [0,1) -> bin 0..k-1
        if idx == k:              # only happens if r == 1.0
            idx = k - 1
        observed[idx] = observed[idx] + 1

    # Step 2: add up the chi-square terms one by one.
    expected = N / k              # E = N / k, same for every bin
    chi2_stat = 0.0
    for o in observed:
        chi2_stat = chi2_stat + (o - expected) ** 2 / expected

    # Step 3: decide using the p-value.
    df = k - 1
    p_value = chi2_dist.sf(chi2_stat, df)     # upper-tail probability (1 - CDF)

    if p_value < alpha:
        decision = "Reject H0"
    else:
        decision = "Do not reject H0"

    return chi2_stat, p_value, decision, observed


def chi_square_test_critical(numbers, k=10, alpha=0.05, chi2_crit=None):
    """
    Chi-Square uniformity test -- decide using the CRITICAL VALUE.
      reject H0 if chi2_stat > chi2_crit
    Pass chi2_crit if the question gives it, else it is looked up for df = k - 1.
    `numbers` is a list of values in [0, 1) (build it in the caller).
    Returns (chi2_stat, chi2_crit, decision, observed).
    """
    # Step 1: count how many numbers land in each bin.
    N = len(numbers)
    observed = [0] * k
    for r in numbers:
        idx = int(r * k)          # r in [0,1) -> bin 0..k-1
        if idx == k:              # only happens if r == 1.0
            idx = k - 1
        observed[idx] = observed[idx] + 1

    # Step 2: add up the chi-square terms one by one.
    expected = N / k              # E = N / k, same for every bin
    chi2_stat = 0.0
    for o in observed:
        chi2_stat = chi2_stat + (o - expected) ** 2 / expected

    # Step 3: decide using the critical value.
    if chi2_crit is None:                     # not given -> compute it
        from scipy.stats import chi2 as chi2_dist
        df = k - 1
        chi2_crit = chi2_dist.ppf(1 - alpha, df)

    if chi2_stat > chi2_crit:
        decision = "Reject H0"
    else:
        decision = "Do not reject H0"

    return chi2_stat, chi2_crit, decision, observed


# ==============================================================================
# 3. Autocorrelation Test
# ==============================================================================
def autocorrelation_test(numbers, start=3, lag=5, alpha=0.05):
    """
    Performs Autocorrelation test for H0: rho_{i,l} = 0.
    
    Formulas:
      M = largest integer such that start + (M + 1) * lag <= N
      rho_hat = [ 1 / (M + 1) * sum_{k=0}^M ( R_{i+kl} * R_{i+(k+1)l} ) ] - 0.25
      sigma = sqrt(13M + 7) / (12 * (M + 1))
      Z0 = rho_hat / sigma
      Reject H0 if |Z0| > 1.96 (for alpha = 0.05)
    """
    N = len(numbers)
    # 0-indexed positions: index = (start - 1) + k * lag
    # Find M:
    M = -1
    while start + (M + 2) * lag <= N:
        M += 1
        
    indices = [start + k * lag for k in range(M + 2)]  # 1-based indices
    subsequence = [numbers[idx - 1] for idx in indices]
    
    products = [subsequence[k] * subsequence[k + 1] for k in range(M + 1)]
    sum_products = sum(products)
    
    rho_hat = (sum_products / (M + 1)) - 0.25
    sigma = math.sqrt(13 * M + 7) / (12 * (M + 1))
    z0 = rho_hat / sigma
    
    z_crit = 1.96  # for alpha = 0.05
    decision = "Reject H0" if abs(z0) > z_crit else "Do not reject H0"
    
    return {
        "M": M,
        "indices": indices,
        "subsequence": subsequence,
        "products": products,
        "rho_hat": rho_hat,
        "sigma": sigma,
        "Z0": z0,
        "critical_value": z_crit,
        "decision": decision
    }


# ==============================================================================
# Main: Verification with Slide Examples
# ==============================================================================
if __name__ == "__main__":
    # ----- Example 6: Kolmogorov-Smirnov Test -----
    alpha_ks = 0.05
    sample_ks = [0.44, 0.81, 0.14, 0.05, 0.93]
    print("=" * 70)
    print(f"Example 6: Kolmogorov-Smirnov Test (Data: {', '.join(f'{v}' for v in sample_ks)})")
    print("=" * 70)
    res_ks = kolmogorov_smirnov_test(sample_ks, alpha=alpha_ks)
    print(f"{'i':<3} | {'R_(i)':<8} | {'i/N':<8} | {'D+':<8} | {'(i-1)/N':<8} | {'D-'}")
    print("-" * 55)
    for row in res_ks["table"]:
        print(f"{row[0]:<3} | {row[1]:<10.6f} | {row[2]:<10.6f} | {row[3]:<10.6f} | {row[4]:<10.6f} | {row[5]:.6f}")
    print(f"\nD+ = {res_ks['D+']:.6f}, D- = {res_ks['D-']:.6f} -> D = {res_ks['D']:.6f}")
    print(f"Critical D_{alpha_ks} = {res_ks['critical_value']:.6f} -> Decision: {res_ks['decision']}")

    # ----- Example 7: Chi-Square on 1000 generated numbers, 10 bins -----
    alpha_chi = 0.05
    gen_N = 1000
    gen_k = 10
    print("\n" + "=" * 70)
    print(f"Example 7: Chi-Square Uniformity Test on {gen_N} generated numbers ({gen_k} bins)")
    print("=" * 70)
    # Step 1: create the numbers list here, then pass it into the tests.
    gen_data = lcg_uniforms(gen_N)
    chi2_g, p_g, dec_p, obs_g = chi_square_test_pvalue(gen_data, k=gen_k, alpha=alpha_chi)
    chi2_g, crit_g, dec_c, _ = chi_square_test_critical(gen_data, k=gen_k, alpha=alpha_chi)

    # Per-bin table: interval, observed O_i, expected E_i, (O-E)^2, (O-E)^2/E
    expected = gen_N / gen_k
    print(f"{'Interval':<12} | {'O_i':<5} | {'E_i':<6} | {'(O-E)^2':<10} | {'(O-E)^2/E'}")
    print("-" * 55)
    for i, o in enumerate(obs_g):
        diff_sq = (o - expected) ** 2
        term = diff_sq / expected
        print(f"[{i/gen_k:.1f}, {(i+1)/gen_k:.1f}) | {o:<5} | {expected:<6.1f} | {diff_sq:<10.4f} | {term:.4f}")

    print(f"\nChi^2 = {chi2_g:.6f}")
    print(f"  1. p-value  : p = {p_g:.6f}          -> {dec_p} (reject if p < {alpha_chi})")
    print(f"  2. critical : crit = {crit_g:.6f} (df={gen_k - 1}) -> {dec_c} (reject if Chi^2 > crit)")

    # ----- Example 8: Autocorrelation Test -----
    alpha_auto = 0.05
    start_auto = 3 #i
    lag_auto = 5 #l
    numbers_30 = [
        0.12, 0.01, 0.23, 0.28, 0.89, 0.31, 0.64, 0.28, 0.83, 0.93,
        0.99, 0.15, 0.33, 0.35, 0.91, 0.41, 0.60, 0.27, 0.75, 0.88,
        0.68, 0.49, 0.05, 0.43, 0.95, 0.58, 0.19, 0.36, 0.69, 0.87
    ]
    N_auto = len(numbers_30)
    print("\n" + "=" * 70)
    print(f"Example 8: Autocorrelation Test (i = {start_auto}, lag = {lag_auto}, N = {N_auto})")
    print("=" * 70)
    res_auto = autocorrelation_test(numbers_30, start=start_auto, lag=lag_auto, alpha=alpha_auto)
    print(f"M = {res_auto['M']}")
    print(f"Subsequence values (indices {res_auto['indices']}): {res_auto['subsequence']}")
    print(f"Products: {[round(p, 4) for p in res_auto['products']]}")
    print(f"rho_hat = {res_auto['rho_hat']:.6f}")
    print(f"sigma   = {res_auto['sigma']:.6f}")
    print(f"Z0      = {res_auto['Z0']:.6f} (Critical = +/-{res_auto['critical_value']})")
    print(f"Decision: {res_auto['decision']}")


